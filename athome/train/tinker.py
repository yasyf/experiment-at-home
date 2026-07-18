from __future__ import annotations

import importlib.util
import os
import random
import tarfile
from dataclasses import dataclass
from functools import partial
from time import perf_counter
from typing import TYPE_CHECKING, ClassVar

import httpx
from anyio import to_thread

from athome.concurrency import gather_bounded
from athome.config import load
from athome.errors import AthomeError
from athome.llm.spend import SpendGuard
from athome.train import data, sidecar
from athome.train.backend import TINKER_ENV
from athome.train.engine import (
    CrossEntropy,
    Custom,
    ScoreOp,
    SnapshotDone,
    SnapshotOp,
    TrainDone,
    TrainOp,
    execute,
    projection,
    token_count,
)
from athome.train.spec import (
    Adapter,
    Checkpoint,
    CheckpointPolicy,
    InsufficientData,
    LoraSpec,
    OverlongEvalRows,
    SavedCheckpoint,
    ScoredSequence,
    StepRecord,
    TinkerSettings,
    TrainReport,
    UnservableBase,
    UnsupportedLoraShape,
    lora_keys,
    spend_cap,
    std_lora_keys,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import tinker
    import torch

    from athome.progress import RunSink
    from athome.train.data import TinkerPreference
    from athome.train.engine import Loss, Op
    from athome.train.spec import (
        BackendName,
        BaseModelSpec,
        EvalRow,
        Method,
        TinkerModelId,
        TinkerPrice,
        TrainSpec,
    )

BETA = 0.1
DOWNLOAD_TIMEOUT = httpx.Timeout(30.0, read=600.0)
TORCH_HINT = (
    "tinker DPO needs torch locally (its custom loss backprops in-process): "
    "install `experiment-at-home[train-dpo]`. SFT needs no torch, and modal DPO runs torch in the Modal image."
)


class TorchRequired(AthomeError):
    """Raised when Tinker's DPO path runs without torch, which its custom loss backprops in."""


def require_torch() -> None:
    if importlib.util.find_spec("torch") is None:
        raise TorchRequired(TORCH_HINT)


def load_key() -> None:
    if "TINKER_API_KEY" not in os.environ:
        os.environ.update(
            env_pair(entry)
            for line in TINKER_ENV.read_text().splitlines()
            if (entry := line.strip()) and not entry.startswith("#")
        )


def env_pair(entry: str) -> tuple[str, str]:
    key, _, value = entry.partition("=")
    return key.strip(), value.strip().strip("\"'")


def tinker_model(base: BaseModelSpec) -> TinkerModelId:
    if base.tinker is None:
        raise UnservableBase(f"{base.mlx} has no Tinker base model")
    return base.tinker


def tinker_lora(lora: LoraSpec) -> tuple[str, ...]:
    """The modules Tinker will train for ``lora``, or a refusal if it cannot train that shape.

    Tinker's trainer takes a rank and the three toggles — nothing else. It picks the alpha, the
    dropout, and the exact module list itself, so a request that names its own is not a request
    Tinker can carry out; accepting it would train one adapter and fuse it as another. What
    Tinker did choose is read back off the archive at conversion time.

    Args:
        lora: The requested LoRA shape.

    Returns:
        The modules Tinker trains: every standard module the toggles select.

    Raises:
        UnsupportedLoraShape: The request names an alpha, a dropout, or a target list Tinker
            cannot honor. Modal takes all three; local takes them too.
    """
    if (lora.alpha, lora.dropout) != ((default := LoraSpec()).alpha, default.dropout):
        raise UnsupportedLoraShape(
            f"tinker chooses the adapter's alpha and dropout itself — its trainer takes only a rank and the "
            f"train_mlp/train_attn/train_unembed toggles — so alpha={lora.alpha}, dropout={lora.dropout} cannot "
            f"be honored. Train that shape on modal or local."
        )
    if (requested := lora_keys(lora)) != (trained := std_lora_keys(lora)):
        raise UnsupportedLoraShape(
            f"tinker trains every standard module its toggles select ({', '.join(trained)}); it cannot narrow "
            f"to {', '.join(requested)}, and the modules it trained anyway would be dropped by the fuse. "
            f"Train that target list on modal or local."
        )
    return trained


def batches[T](pool: Sequence[T], *, size: int, steps: int, seed: int) -> list[tuple[T, ...]]:
    rng = random.Random(seed)
    order: list[T] = []
    plan: list[tuple[T, ...]] = []
    while len(plan) < steps:
        if len(order) < size:
            order.extend(rng.sample(pool, len(pool)))
        plan.append(tuple(order[:size]))
        del order[:size]
    return plan


def fits(datum: tinker.Datum, max_seq_len: int) -> bool:
    return datum.model_input.length + 1 <= max_seq_len


def weights_of(datum: tinker.Datum) -> Sequence[float]:
    return datum.loss_fn_inputs["weights"].data


def sequence_logprob(output: dict[str, tinker.TensorData], datum: tinker.Datum) -> float:
    return sum(logprob * weight for logprob, weight in zip(output["logprobs"].data, weights_of(datum), strict=True))


def score_sequence(logprobs: Sequence[float], weights: Sequence[float]) -> ScoredSequence:
    return ScoredSequence(
        logprob=sum(logprob * weight for logprob, weight in zip(logprobs, weights, strict=True)),
        weight=sum(weights),
    )


def masked_loss(outputs: Sequence[dict[str, tinker.TensorData]], batch: Sequence[tinker.Datum]) -> float:
    scored = sum(sequence_logprob(output, datum) for output, datum in zip(outputs, batch, strict=True))
    return -scored / sum(sum(weights_of(datum)) for datum in batch)


def pair_datums(prefs: Sequence[TinkerPreference]) -> list[tinker.Datum]:
    return [datum for pref in prefs for datum in (pref.chosen, pref.rejected)]


def dpo_loss(
    datums: Sequence[tinker.Datum],
    logprobs: Sequence[torch.Tensor],
    *,
    reference: Sequence[float],
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """The DPO objective over interleaved chosen/rejected sequence logprobs.

    ``datums`` and ``logprobs`` arrive from Tinker's custom-loss path as
    ``[chosen_0, rejected_0, chosen_1, ...]``; ``reference`` holds the frozen policy's
    sequence logprobs in the same order. Backward is the SDK's: it reads
    ``-d(loss)/d(logprobs)`` off these tensors and ships it as the surrogate's weights.

    Args:
        datums: The chosen/rejected ``Datum`` s, whose ``weights`` select the completion tokens.
        logprobs: The trained policy's per-token logprobs, one differentiable tensor per datum.
        reference: The frozen policy's sequence logprobs, one per datum.
        beta: The KL penalty strength on the implicit reward.

    Returns:
        The mean ``-log_sigmoid`` preference loss, and the loss/margin/accuracy metrics.
    """
    import torch

    policy = torch.stack(
        [
            (logprob * torch.tensor(weights_of(datum), dtype=logprob.dtype)).sum()
            for datum, logprob in zip(datums, logprobs, strict=True)
        ]
    )
    frozen = torch.tensor(reference, dtype=policy.dtype)
    margin = beta * ((policy[0::2] - frozen[0::2]) - (policy[1::2] - frozen[1::2]))
    loss = -torch.nn.functional.logsigmoid(margin).mean()
    return loss, {
        "loss": loss.item(),
        "margin": margin.mean().item(),
        "accuracy": (margin > 0).to(margin.dtype).mean().item(),
    }


def eval_datum(row: EvalRow) -> tinker.Datum:
    """One scoring ``Datum`` from a pre-tokenized eval row, shifted like a masked completion.

    The full sequence is the input up to its last token; the targets are the sequence shifted
    by one, weighted so only the row's marked positions are scored.
    """
    import tinker

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(row.tokens[:-1])),
        loss_fn_inputs={
            "weights": tinker.TensorData(data=list(row.weights[1:]), dtype="float32", shape=[len(row.weights) - 1]),
            "target_tokens": tinker.TensorData(data=list(row.tokens[1:]), dtype="int64", shape=[len(row.tokens) - 1]),
        },
    )


def step_metrics(loss: Loss, output: tinker.ForwardBackwardOutput, datums: Sequence[tinker.Datum]) -> dict[str, float]:
    match loss:
        case CrossEntropy():
            return {"loss": masked_loss(output.loss_fn_outputs, datums)}
        case Custom():
            return {name: output.metrics[name] for name in ("loss", "margin", "accuracy")}


def reduce_eval(
    outputs: Sequence[dict[str, tinker.TensorData]], eval_datums: Sequence[tinker.Datum]
) -> tuple[ScoredSequence, ...]:
    return tuple(
        score_sequence(output["logprobs"].data, weights_of(datum))
        for output, datum in zip(outputs, eval_datums, strict=True)
    )


def assemble(
    spec: TrainSpec,
    steps: Sequence[tuple[tuple[tinker.Datum, ...], Loss]],
    *,
    checkpoints: CheckpointPolicy,
    eval_datums: tuple[tinker.Datum, ...],
) -> tuple[tuple[Op, ...], list[tuple[int, bool]]]:
    """The schedule for ``steps``: a train op per step, snapshots at the policy's cadence and the end.

    Returns the ops in submission order and the ``(step, final)`` metadata for each snapshot in
    that same order, so the fold can label each :class:`SavedCheckpoint` as its result drains.
    """
    total = spec.hyperparams.steps
    intermediate = set(checkpoints.steps_for(total))
    ops: list[Op] = []
    meta: list[tuple[int, bool]] = []
    for step, (datums, loss) in enumerate(steps, start=1):
        ops.append(TrainOp(datums, loss, spec.hyperparams.learning_rate))
        if step in intermediate:
            ops.append(SnapshotOp(f"{spec.name}-step{step:05d}", checkpoints.ttl_seconds, eval_datums))
            meta.append((step, False))
    ops.append(SnapshotOp(f"{spec.name}-{spec.method}-{total}", None, eval_datums))
    meta.append((total, True))
    return tuple(ops), meta


async def reference_pass(
    frozen: tinker.TrainingClient, datums: Sequence[tinker.Datum], *, guard: SpendGuard, price: TinkerPrice
) -> list[tuple[float, float]]:
    """The frozen policy's chosen/rejected sequence logprobs, scored on the batched scorer.

    Runs as a single :class:`ScoreOp` fully consumed before the training schedule is compiled,
    since its values are baked into the DPO loss closures.
    """
    [scored] = [result async for result in execute(frozen, (ScoreOp(tuple(datums)),))]
    spent = price.prefill * token_count(datums) / 1e6
    await guard.record(spent, spent)
    scores = [sequence_logprob(output, datum) for output, datum in zip(scored.outputs, datums, strict=True)]
    return list(zip(scores[0::2], scores[1::2], strict=True))


async def run_schedule(
    spec: TrainSpec,
    client: tinker.TrainingClient,
    schedule: Sequence[Op],
    meta: Sequence[tuple[int, bool]],
    dropped: int,
    *,
    guard: SpendGuard,
    price: TinkerPrice,
    sink: RunSink,
    eval_datums: Sequence[tinker.Datum],
) -> TrainReport:
    """Fold the executor's result stream into a :class:`TrainReport`, recording spend and journaling.

    Every side effect — spend reconciliation, the journal line — lands at a drain point, so the
    journal stays step-ordered and the executor tops the queue up before each yield.
    """
    records: list[StepRecord] = []
    saved: list[SavedCheckpoint] = []
    snapshots = iter(meta)
    start = perf_counter()
    async for result in execute(client, schedule):
        match result:
            case TrainDone(op=op, output=output):
                tokens = token_count(op.datums)
                cost = price.train * tokens * op.loss.passes / 1e6
                await guard.record(cost, cost)
                metrics = step_metrics(op.loss, output, op.datums)
                step = len(records) + 1
                await sink.append(
                    {"step": step, "method": spec.method, "tokens": tokens, "cost_usd": guard.spent} | metrics
                )
                records.append(StepRecord(step, tokens, cost, metrics))
            case SnapshotDone(op=op, path=path, outputs=outputs):
                step, final = next(snapshots)
                if outputs is not None:
                    eval_cost = price.prefill * token_count(op.eval) / 1e6
                    await guard.record(eval_cost, eval_cost)
                scores = reduce_eval(outputs, eval_datums) if outputs is not None else None
                saved.append(SavedCheckpoint(step=step, path=path, final=final, scores=scores))
                await sink.append({"event": "checkpoint", "step": step, "path": path, "final": final})
    return TrainReport(
        method=spec.method,
        steps=tuple(records),
        checkpoints=tuple(saved),
        dropped=dropped,
        wall_s=perf_counter() - start,
        train_cost_usd=guard.spent,
    )


def extract(archive: Path, out_dir: Path) -> None:
    with tarfile.open(archive) as tar:
        tar.extractall(out_dir, filter="data")


async def download_adapter(service: tinker.ServiceClient, tinker_path: str, out_dir: Path) -> Path:
    """Download and unpack a Tinker sampler checkpoint (a PEFT adapter) into ``out_dir``."""
    signed = await service.create_rest_client().get_checkpoint_archive_url_from_tinker_path_async(tinker_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / "checkpoint.tar"
    async with (
        httpx.AsyncClient(follow_redirects=True, timeout=DOWNLOAD_TIMEOUT) as http,
        http.stream("GET", signed.url) as response,
    ):
        response.raise_for_status()
        with archive.open("wb") as file:
            async for chunk in response.aiter_bytes():
                file.write(chunk)
    await to_thread.run_sync(extract, archive, out_dir)
    archive.unlink()
    return out_dir


@dataclass(frozen=True, slots=True)
class TinkerBackend:
    """LoRA fine-tuning on Tinker's managed trainers, converging on a fused local MLX model.

    SFT trains against Tinker's ``cross_entropy`` loss over prompt-masked ``Datum`` s. DPO has
    no named Tinker loss, so it runs through ``forward_backward_custom``: the trained policy's
    logprobs come back as torch tensors, :func:`dpo_loss` scores them against a frozen
    reference policy's cached logprobs, and the SDK backpropagates the client-side gradient.

    Every op runs through the pipelined :func:`~athome.train.engine.execute` stream, which keeps
    the server's turnstile full so a step costs one clock cycle instead of three, and lets
    checkpoint saves and their eval scoring ride between steps without stalling. Every run is
    projected against ``spend_cap_usd`` before its first billable call and charged step by step,
    so a run that cannot fit the cap aborts having spent nothing.

    Example:
        >>> checkpoint = await TinkerBackend.from_settings().train(spec, sink=sink)
    """

    settings: TinkerSettings
    name: ClassVar[BackendName] = "tinker"

    @staticmethod
    def available() -> bool:
        """Whether the Tinker SDK is installed and keyed: a key in the env, or ``~/.athome/tinker.env``.

        The SDK is half of it. Selection treats an available backend as one that can actually run,
        so claiming availability without ``tinker`` on the path picks this backend and then dies
        importing it, instead of falling through to one that would have worked.
        """
        return importlib.util.find_spec("tinker") is not None and (
            "TINKER_API_KEY" in os.environ or TINKER_ENV.exists()
        )

    @staticmethod
    def supports(method: Method) -> bool:
        """Whether Tinker trains ``method`` *here*: SFT always, DPO only where torch can back it.

        Tinker has no named preference loss, so DPO runs through ``forward_backward_custom`` and
        :func:`dpo_loss` backprops it in-process — without torch there is nothing to run it with.
        """
        match method:
            case "sft":
                return True
            case "dpo":
                return importlib.util.find_spec("torch") is not None

    @classmethod
    def from_settings(cls) -> TinkerBackend:
        """Load ``[train.tinker]``, taking the API key from ``~/.athome/tinker.env`` when unset."""
        load_key()
        return cls(load(TinkerSettings))

    def cost(self, *, model: TinkerModelId, prefill: int = 0, sample: int = 0, train: int = 0) -> float:
        """What ``model`` bills for these token counts, each class at its own per-Mtok rate.

        Args:
            model: The Tinker base model whose price sheet applies.
            prefill: Tokens through a forward-only pass.
            sample: Tokens generated by a sampling client.
            train: Tokens through a ``forward_backward`` and its ``optim_step``.

        Returns:
            The USD total across the three classes.
        """
        price = self.settings.price_per_mtok[model]
        return (prefill * price.prefill + sample * price.sample + train * price.train) / 1e6

    async def lora_client(
        self, service: tinker.ServiceClient, spec: TrainSpec, *, model: TinkerModelId
    ) -> tinker.TrainingClient:
        return await service.create_lora_training_client_async(
            base_model=model,
            rank=spec.lora.rank,
            seed=spec.hyperparams.seed,
            train_mlp=spec.lora.train_mlp,
            train_attn=spec.lora.train_attn,
            train_unembed=spec.lora.train_unembed,
        )

    def service(self) -> tinker.ServiceClient:
        import tinker

        return tinker.ServiceClient(api_key=self.settings.api_key.get_secret_value())

    async def fit(
        self,
        spec: TrainSpec,
        *,
        sink: RunSink,
        checkpoints: CheckpointPolicy = CheckpointPolicy(),
        eval_rows: Sequence[EvalRow] | None = None,
    ) -> TrainReport:
        """Train ``spec`` on Tinker, journaling each step and saving checkpoints at the cadence.

        The pool is rendered and under-filled runs aborted before anything billable; the whole
        schedule — training, the DPO reference pass, and every checkpoint's eval scoring — is
        projected against the spend cap in one reservation. The stream then runs, spend and the
        journal reconciling at each drain point.

        Args:
            spec: The fine-tuning request: base, dataset, hyperparams, method, LoRA, spend cap.
            sink: The run journal; one record lands per step, plus a checkpoint event per save.
            checkpoints: Which fractions of the run to snapshot as intermediates; the final step
                is always snapshotted and kept forever.
            eval_rows: Pre-tokenized rows scored against every checkpoint's weights, or None.

        Returns:
            The report: per-step records, the saved checkpoints with their eval scores, the drop
            count, the wall-clock, and the total metered spend.

        Raises:
            UnservableBase: The base has no Tinker id.
            UnsupportedLoraShape: The LoRA shape asks for something Tinker cannot train.
            TorchRequired: The method is ``dpo`` and torch is not installed locally.
            InsufficientData: The surviving pool is smaller than one batch.
            OverlongEvalRows: An eval row exceeds ``max_seq_len``.
            SpendExceeded: The projected run cost crosses the spend cap.
        """
        model = tinker_model(spec.base)
        tinker_lora(spec.lora)
        if spec.method == "dpo":
            require_torch()
        price = self.settings.price_per_mtok[model]
        eval_datums = tuple(eval_datum(row) for row in (eval_rows or ()))
        if overlong := sum(not fits(datum, spec.hyperparams.max_seq_len) for datum in eval_datums):
            raise OverlongEvalRows(overlong, spec.hyperparams.max_seq_len)

        guard = SpendGuard(max_usd=spend_cap(spec, self.settings.spend_cap_usd))
        match spec.method:
            case "sft":
                schedule, meta, dropped, client = await self.prepare_sft(
                    spec, model=model, checkpoints=checkpoints, eval_datums=eval_datums, guard=guard, price=price
                )
            case "dpo":
                schedule, meta, dropped, client = await self.prepare_dpo(
                    spec, model=model, checkpoints=checkpoints, eval_datums=eval_datums, guard=guard, price=price
                )
        return await run_schedule(
            spec, client, schedule, meta, dropped, guard=guard, price=price, sink=sink, eval_datums=eval_datums
        )

    async def prepare_sft(
        self,
        spec: TrainSpec,
        *,
        model: TinkerModelId,
        checkpoints: CheckpointPolicy,
        eval_datums: tuple[tinker.Datum, ...],
        guard: SpendGuard,
        price: TinkerPrice,
    ) -> tuple[tuple[Op, ...], list[tuple[int, bool]], int, tinker.TrainingClient]:
        examples = await data.normalize(spec.dataset, method="sft")
        pool = [
            datum
            for example in examples
            if fits(datum := data.render_tinker_sft(example, spec.base.mlx), spec.hyperparams.max_seq_len)
        ]
        if len(pool) < spec.hyperparams.batch_size:
            raise InsufficientData(len(pool), spec.hyperparams.batch_size)
        plan = batches(pool, size=spec.hyperparams.batch_size, steps=spec.hyperparams.steps, seed=spec.hyperparams.seed)
        schedule, meta = assemble(
            spec, [(batch, CrossEntropy()) for batch in plan], checkpoints=checkpoints, eval_datums=eval_datums
        )
        await guard.check(projection(schedule, price))
        client = await self.lora_client(self.service(), spec, model=model)
        return schedule, meta, len(examples) - len(pool), client

    async def prepare_dpo(
        self,
        spec: TrainSpec,
        *,
        model: TinkerModelId,
        checkpoints: CheckpointPolicy,
        eval_datums: tuple[tinker.Datum, ...],
        guard: SpendGuard,
        price: TinkerPrice,
    ) -> tuple[tuple[Op, ...], list[tuple[int, bool]], int, tinker.TrainingClient]:
        examples = await data.normalize(spec.dataset, method="dpo")
        max_seq_len = spec.hyperparams.max_seq_len
        pool = [
            pref
            for example in examples
            if fits((pref := data.render_tinker_dpo(example, spec.base.mlx)).chosen, max_seq_len)
            and fits(pref.rejected, max_seq_len)
        ]
        if len(pool) < spec.hyperparams.batch_size:
            raise InsufficientData(len(pool), spec.hyperparams.batch_size)
        plan = batches(
            range(len(pool)), size=spec.hyperparams.batch_size, steps=spec.hyperparams.steps, seed=spec.hyperparams.seed
        )
        reference_datums = pair_datums(pool)
        shape, meta = assemble(
            spec,
            [(tuple(pair_datums([pool[i] for i in batch])), Custom(dpo_loss)) for batch in plan],
            checkpoints=checkpoints,
            eval_datums=eval_datums,
        )
        await guard.check(projection(shape, price) + price.prefill * token_count(reference_datums) / 1e6)

        service = self.service()
        reference = await reference_pass(
            await self.lora_client(service, spec, model=model), reference_datums, guard=guard, price=price
        )
        client = await self.lora_client(service, spec, model=model)
        schedule, _ = assemble(
            spec,
            [
                (
                    tuple(pair_datums([pool[i] for i in batch])),
                    Custom(partial(dpo_loss, reference=[score for i in batch for score in reference[i]], beta=BETA)),
                )
                for batch in plan
            ],
            checkpoints=checkpoints,
            eval_datums=eval_datums,
        )
        return schedule, meta, len(examples) - len(pool), client

    async def train(self, spec: TrainSpec, *, sink: RunSink, work_dir: Path) -> Checkpoint:
        """Train ``spec`` on Tinker and fuse the resulting final adapter into a servable MLX model.

        Args:
            spec: The fine-tuning request: base, dataset, hyperparams, method, LoRA, spend cap.
            sink: The run journal; one record lands per step with its loss, tokens, and spend.
            work_dir: This run's own directory; the adapter and the fused model are written under it.

        Returns:
            The checkpoint, whose ``mlx_path`` is the fused standalone 4-bit MLX model.

        Raises:
            UnservableBase: The base has no Tinker id, or no mlx-lm counterpart to fuse into.
            UnsupportedLoraShape: The LoRA shape asks for something Tinker cannot train.
            TorchRequired: The method is ``dpo`` and torch is not installed locally.
            InsufficientData: The surviving pool is smaller than one batch.
            SpendExceeded: The projected run cost crosses the spend cap.
        """
        if not spec.base.serves_locally:
            raise UnservableBase(
                f"{spec.base.mlx} has no mlx-lm LoRA counterpart, so a Tinker adapter cannot be fused into it"
            )
        report = await self.fit(spec, sink=sink)
        adapter = await self.materialize(report.final, spec, work_dir=work_dir, cost=report.train_cost_usd)
        return await self.fuse(adapter, spec, work_dir=work_dir)

    async def materialize(self, saved: SavedCheckpoint, spec: TrainSpec, *, work_dir: Path, cost: float) -> Adapter:
        """Download and convert one saved Tinker checkpoint into a non-fused MLX adapter.

        Args:
            saved: Any saved checkpoint from :meth:`fit`, including an intermediate.
            spec: The fine-tuning request whose base model the adapter targets.
            work_dir: This materialization's directory; PEFT and MLX adapter files land under it.
            cost: The training spend attributed to the adapter.

        Returns:
            The non-fused MLX adapter, ready for direct adapter serving or a later :meth:`fuse`.
        """
        peft = await download_adapter(self.service(), saved.path, work_dir / "peft")
        adapter_dir = await sidecar.convert_peft_to_mlx(peft, work_dir / "adapter", base=spec.base)
        return Adapter(step=saved.step, adapter_dir=adapter_dir, train_cost_usd=cost)

    async def fuse(self, adapter: Adapter, spec: TrainSpec, *, work_dir: Path) -> Checkpoint:
        """Fuse a materialized adapter into a standalone servable MLX checkpoint.

        Args:
            adapter: The non-fused MLX adapter to merge into its base model.
            spec: The fine-tuning request whose base and method describe the checkpoint.
            work_dir: This fusion's directory; the standalone model is written under it.

        Returns:
            The fused checkpoint, preserving the adapter's step and training cost.
        """
        return Checkpoint(
            base=spec.base,
            backend="tinker",
            method=spec.method,
            step=adapter.step,
            mlx_path=await sidecar.fuse(adapter.adapter_dir, work_dir / "mlx", base=spec.base),
            adapter_dir=adapter.adapter_dir,
            train_cost_usd=adapter.train_cost_usd,
        )

    async def score(
        self,
        path: str,
        rows: Sequence[EvalRow],
        *,
        base: BaseModelSpec,
        max_usd: float | None = None,
    ) -> tuple[ScoredSequence, ...]:
        """Score weighted token sequences against a saved Tinker sampling checkpoint.

        Args:
            path: The opaque ``tinker://`` sampler checkpoint path.
            rows: Pre-tokenized weighted rows to score, in return order.
            base: The base model identity used to price the sampling requests.
            max_usd: The scoring spend cap, or None for the configured Tinker cap.

        Returns:
            One weighted sequence score per row, preserving input order.

        Raises:
            UnservableBase: The base has no Tinker identity.
            SpendExceeded: The projected prefill and one sampled token per row cross the cap.
        """
        import tinker

        model = tinker_model(base)
        projected = self.cost(
            model=model,
            prefill=sum(len(row.tokens) for row in rows),
            sample=len(rows),
        )
        guard = SpendGuard(max_usd=self.settings.spend_cap_usd if max_usd is None else max_usd)
        await guard.check(projected)
        client = self.service().create_sampling_client(model_path=path)
        prompts = [tinker.ModelInput.from_ints(list(row.tokens)) for row in rows]
        outputs = await gather_bounded(
            [partial(client.compute_logprobs_async, prompt) for prompt in prompts],
            concurrency=64,
        )
        await guard.record(projected, projected)
        return tuple(
            score_sequence(
                tuple(logprob for logprob in output[1:] if logprob is not None),
                row.weights[1:],
            )
            for output, row in zip(outputs, rows, strict=True)
        )
