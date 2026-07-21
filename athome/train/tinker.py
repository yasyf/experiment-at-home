from __future__ import annotations

import importlib.util
import os
import random
import tarfile
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from time import perf_counter
from typing import TYPE_CHECKING, ClassVar

import httpx
from anyio import CancelScope, to_thread

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
from athome.train.runstate import RunState, run_key, spec_fingerprint
from athome.train.spec import (
    Adapter,
    Checkpoint,
    CheckpointPolicy,
    InsufficientData,
    LoraSpec,
    OverlongEvalRows,
    SampledSequence,
    SavedCheckpoint,
    ScoredSequence,
    StepRecord,
    TinkerSettings,
    TrainReport,
    UnservableBase,
    UnsupportedLoraShape,
    lora_keys,
    require_servable,
    spend_cap,
    std_lora_keys,
)
from athome.train.state import BackendMismatch, TinkerState

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import tinker
    import torch

    from athome.progress import RunSink
    from athome.train.data import Message, TinkerPreference
    from athome.train.engine import Loss, Op
    from athome.train.runstate import RunStateStore
    from athome.train.spec import (
        BackendName,
        BaseModelSpec,
        EvalRow,
        Method,
        TinkerModelId,
        TinkerPrice,
        TrainSpec,
    )
    from athome.train.state import Resume, StateFidelity, StateHandle

BETA = 0.1
DOWNLOAD_TIMEOUT = httpx.Timeout(30.0, read=600.0)
TORCH_HINT = (
    "tinker DPO needs torch locally (its custom loss backprops in-process): "
    "install `experiment-at-home[train-dpo]`. SFT needs no torch, and modal DPO runs torch in the Modal image."
)


class TorchRequired(AthomeError):
    """Raised when Tinker's DPO path runs without torch, which its custom loss backprops in."""


class LoraShapeDrift(AthomeError):
    """Raised when a resume seed's saved LoRA shape disagrees with the spec's requested shape.

    The SDK derives the LoRA rank and module toggles from the checkpoint's ``weights_info`` and
    ignores the caller's spec on from-state creation, so a silent disagreement would train a
    different adapter than the spec asked for. The mismatch is refused loudly instead.
    """


class ExpiredStateHandle(AthomeError):
    """Raised when a resume seed's saved training state is expired or missing at restore.

    A vanished checkpoint never silently falls back to a fresh-from-base run — that would bill a
    full run while quietly discarding the continuation the caller asked for. It is refused loudly,
    naming the handle and the explicit re-run-from-base option.
    """


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
    base_step: int = 0,
    run_tag: str,
) -> tuple[tuple[Op, ...], list[tuple[int, bool]]]:
    """The schedule for ``steps``: a train op per step, snapshots at the policy's cadence and the end.

    Returns the ops in submission order and the ``(step, final)`` metadata for each snapshot in
    that same order, so the fold can label each :class:`SavedCheckpoint` as its result drains.

    ``run_tag`` namespaces every snapshot name by this attempt's own run identity, so a fresh
    re-run of the same spec — a completed or re-proposed family, whose ``base_step`` is 0 — never
    re-emits a prior run's immortal final state name under the SDK's ``overwrite=False``.

    ``base_step`` offsets both the step labels and the snapshot names for a same-run resume: the
    remaining ``steps`` are labeled ``base_step + 1 …`` (absolute, so cadence intermediates land at
    the numbers an uninterrupted run would use) and every snapshot name carries an ``-r{base_step}``
    tag on top of ``run_tag``, so a resumed attempt never collides with the prior attempt's
    already-saved names under the SDK's ``overwrite=False``.
    """
    total = spec.hyperparams.steps
    intermediate = set(checkpoints.steps_for(total))
    run_prefix = f"-{run_tag}" if run_tag else ""
    tag = run_prefix if base_step == 0 else f"{run_prefix}-r{base_step:05d}"
    ops: list[Op] = []
    meta: list[tuple[int, bool]] = []
    for step, (datums, loss) in enumerate(steps, start=base_step + 1):
        ops.append(TrainOp(datums, loss, spec.hyperparams.learning_rate))
        if step in intermediate:
            ops.append(SnapshotOp(f"{spec.name}{tag}-step{step:05d}", checkpoints.ttl_seconds, eval_datums))
            meta.append((step, False))
    ops.append(SnapshotOp(f"{spec.name}{tag}-{spec.method}-{total}", None, eval_datums))
    meta.append((total, True))
    return tuple(ops), meta


@dataclass(slots=True)
class Reservation:
    guard: SpendGuard
    outstanding: float = 0.0

    async def reserve(self, projected: float) -> None:
        await self.guard.check(projected)
        self.outstanding += projected

    async def reconcile(self, amount: float) -> None:
        await self.guard.record(amount, amount)
        self.outstanding -= amount

    async def release(self) -> None:
        await self.guard.release(self.outstanding)
        self.outstanding = 0.0


async def reference_pass(
    frozen: tinker.TrainingClient, datums: Sequence[tinker.Datum], *, reservation: Reservation, price: TinkerPrice
) -> list[tuple[float, float]]:
    """The frozen policy's chosen/rejected sequence logprobs, scored on the batched scorer.

    Runs as a single :class:`ScoreOp` fully consumed before the training schedule is compiled,
    since its values are baked into the DPO loss closures.
    """
    [scored] = [result async for result in execute(frozen, (ScoreOp(tuple(datums)),))]
    spent = price.prefill * token_count(datums) / 1e6
    await reservation.reconcile(spent)
    scores = [sequence_logprob(output, datum) for output, datum in zip(scored.outputs, datums, strict=True)]
    return list(zip(scores[0::2], scores[1::2], strict=True))


async def run_schedule(
    spec: TrainSpec,
    client: tinker.TrainingClient,
    schedule: Sequence[Op],
    meta: Sequence[tuple[int, bool]],
    dropped: int,
    *,
    reservation: Reservation,
    price: TinkerPrice,
    sink: RunSink,
    eval_datums: Sequence[tinker.Datum],
    base_step: int = 0,
    store: RunStateStore | None = None,
    reference: StateHandle | None = None,
) -> TrainReport:
    """Fold the executor's result stream into a :class:`TrainReport`, recording spend and journaling.

    Every side effect — spend reconciliation, the journal line, and the run-state upsert — lands at
    a drain point, so the journal and the persisted resume record stay step-ordered and the executor
    tops the queue up before each yield. Each snapshot binds its sampler and training-state paths
    into one :class:`SavedCheckpoint`, and, when a ``store`` is threaded, persists the run's most
    progressed :class:`RunState` there — the one place run state is written.

    The step counter starts at ``base_step`` so a resumed fold labels its steps and its cumulative
    records from where the prior attempt left off.
    """
    records: list[StepRecord] = []
    saved: list[SavedCheckpoint] = []
    snapshots = iter(meta)
    key = run_key(spec)
    fingerprint = spec_fingerprint(spec)
    completed = base_step
    start = perf_counter()
    async for result in execute(client, schedule):
        match result:
            case TrainDone(op=op, output=output):
                tokens = token_count(op.datums)
                cost = price.train * tokens * op.loss.passes / 1e6
                await reservation.reconcile(cost)
                metrics = step_metrics(op.loss, output, op.datums)
                completed += 1
                await sink.append(
                    {"step": completed, "method": spec.method, "tokens": tokens, "cost_usd": reservation.guard.spent}
                    | metrics
                )
                records.append(StepRecord(completed, tokens, cost, metrics))
            case SnapshotDone(op=op, sampler_path=sampler_path, state_path=state_path, outputs=outputs):
                step, final = next(snapshots)
                if outputs is not None:
                    eval_cost = price.prefill * token_count(op.eval) / 1e6
                    await reservation.reconcile(eval_cost)
                scores = reduce_eval(outputs, eval_datums) if outputs is not None else None
                handle = TinkerState(state_path=state_path)
                saved.append(
                    SavedCheckpoint(step=step, sampler_path=sampler_path, state=handle, final=final, scores=scores)
                )
                if store is not None:
                    await store.put(
                        RunState(
                            run_key=key,
                            spec_fingerprint=fingerprint,
                            step=step,
                            handle=handle,
                            reference=reference,
                            cost_usd=reservation.guard.spent,
                            status="running",
                            updated_at=datetime.now(UTC),
                        )
                    )
                await sink.append({"event": "checkpoint", "step": step, "path": sampler_path, "final": final})
    return TrainReport(
        method=spec.method,
        steps=tuple(records),
        checkpoints=tuple(saved),
        dropped=dropped,
        wall_s=perf_counter() - start,
        train_cost_usd=reservation.guard.spent,
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
    state_fidelity: ClassVar[StateFidelity] = "weights+optimizer"

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
        self,
        service: tinker.ServiceClient,
        spec: TrainSpec,
        *,
        model: TinkerModelId,
        seed: StateHandle | None = None,
    ) -> tinker.TrainingClient:
        """A training client for ``spec``: fresh from ``model``, or seeded from a prior Tinker state.

        With no ``seed`` the client is a fresh LoRA over the base. A :class:`TinkerState` seed restores
        the weights *and* optimizer via ``create_training_client_from_state_with_optimizer_async`` — after
        :meth:`require_matches` refuses a saved shape that disagrees with ``spec.lora``. A foreign handle
        (``LocalState``/``ModalState``) cannot seed a Tinker client and is refused loudly.
        """
        match seed:
            case None:
                return await service.create_lora_training_client_async(
                    base_model=model,
                    rank=spec.lora.rank,
                    seed=spec.hyperparams.seed,
                    train_mlp=spec.lora.train_mlp,
                    train_attn=spec.lora.train_attn,
                    train_unembed=spec.lora.train_unembed,
                )
            case TinkerState(state_path=state_path):
                await self.require_matches(service, state_path, spec)
                return await service.create_training_client_from_state_with_optimizer_async(state_path)
            case _:
                raise BackendMismatch(f"tinker cannot seed a training client from a {type(seed).__name__}: {seed}")

    async def require_matches(self, service: tinker.ServiceClient, state_path: str, spec: TrainSpec) -> None:
        """Refuse a resume whose saved LoRA shape disagrees with what ``spec.lora`` restores to.

        The SDK reads the rank and module toggles off the checkpoint's ``weights_info`` and ignores
        ``spec.lora`` on a from-state creation, so a silent disagreement would train a different
        adapter than the spec asked for. This reads that saved shape and fails loudly on a mismatch —
        or, when the saved state has expired or vanished, refuses just as loudly rather than falling
        back to a fresh run.
        """
        try:
            info = await service.create_rest_client().get_weights_info_by_tinker_path(state_path).result_async()
        except (KeyError, httpx.HTTPError) as missing:
            raise ExpiredStateHandle(
                f"resume state {state_path} is expired or missing ({missing}); re-run from base by dropping "
                f"resume_from, or point the continuation at a live checkpoint."
            ) from missing
        fields = ("rank", "train_mlp", "train_attn", "train_unembed")
        restored = (
            info.lora_rank,
            info.train_mlp if info.train_mlp is not None else True,
            info.train_attn if info.train_attn is not None else True,
            info.train_unembed if info.train_unembed is not None else True,
        )
        requested = (spec.lora.rank, spec.lora.train_mlp, spec.lora.train_attn, spec.lora.train_unembed)
        if restored != requested:
            restored_shape = ", ".join(f"{name}={value}" for name, value in zip(fields, restored, strict=True))
            requested_shape = ", ".join(f"{name}={value}" for name, value in zip(fields, requested, strict=True))
            raise LoraShapeDrift(
                f"resume checkpoint {state_path} restores LoRA shape ({restored_shape}), which the SDK derives "
                f"and the spec cannot override, but spec.lora asks for ({requested_shape}); re-run from base or "
                f"match the saved shape."
            )

    def service(self) -> tinker.ServiceClient:
        import tinker

        return tinker.ServiceClient(api_key=self.settings.api_key.get_secret_value())

    async def fit(
        self,
        spec: TrainSpec,
        *,
        sink: RunSink,
        budget: SpendGuard,
        checkpoints: CheckpointPolicy = CheckpointPolicy(),
        eval_rows: Sequence[EvalRow] | None = None,
        resume: Resume | None = None,
        store: RunStateStore | None = None,
        run_tag: str = "",
    ) -> TrainReport:
        """Train ``spec`` on Tinker, journaling each step and saving checkpoints at the cadence.

        The pool is rendered and under-filled runs aborted before anything billable; the whole
        schedule — training, the DPO reference pass, and every checkpoint's eval scoring — is
        projected against ``budget`` in one reservation. The stream then runs, spend and the
        journal reconciling at each drain point.

        Args:
            spec: The fine-tuning request: base, dataset, hyperparams, method, LoRA.
            sink: The run journal; one record lands per step, plus a checkpoint event per save.
            budget: The spend envelope; the whole schedule's projected cost is reserved against it
                before the first billable call and reconciled to actuals as the run drains.
            checkpoints: Which fractions of the run to snapshot as intermediates; the final step
                is always snapshotted and kept forever.
            eval_rows: Pre-tokenized rows scored against every checkpoint's weights, or None.
            resume: The restore input — seeds the training client, slices the plan to the steps a
                same-run crash left unrun, and carries the DPO reference anchor — or None for a run
                from base.
            store: The run-state store the fold persists this run's most progressed
                :class:`RunState` to at each snapshot, or None to train without a resume ledger
                (an eval-only fit through observe/retrain).
            run_tag: This run's identity namespace, folded into every snapshot name so a fresh
                re-run of the same spec never re-emits a prior run's immortal final state name;
                empty for a standalone fit outside the managed :func:`~athome.train.run` lifecycle.

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

        reservation = Reservation(budget)
        settled = False
        try:
            match spec.method:
                case "sft":
                    schedule, meta, dropped, client = await self.prepare_sft(
                        spec,
                        model=model,
                        checkpoints=checkpoints,
                        eval_datums=eval_datums,
                        reservation=reservation,
                        price=price,
                        resume=resume,
                        run_tag=run_tag,
                    )
                case "dpo":
                    schedule, meta, dropped, client = await self.prepare_dpo(
                        spec,
                        model=model,
                        checkpoints=checkpoints,
                        eval_datums=eval_datums,
                        reservation=reservation,
                        price=price,
                        resume=resume,
                        run_tag=run_tag,
                    )
            report = await run_schedule(
                spec,
                client,
                schedule,
                meta,
                dropped,
                reservation=reservation,
                price=price,
                sink=sink,
                eval_datums=eval_datums,
                base_step=resume.from_step if resume is not None else 0,
                store=store,
                reference=resume.reference if resume is not None else None,
            )
            settled = True
            return report
        finally:
            if not settled:
                # Shield so a cancellation delivered mid-release cannot abort it and strand the reservation.
                with CancelScope(shield=True):
                    await reservation.release()

    async def prepare_sft(
        self,
        spec: TrainSpec,
        *,
        model: TinkerModelId,
        checkpoints: CheckpointPolicy,
        eval_datums: tuple[tinker.Datum, ...],
        reservation: Reservation,
        price: TinkerPrice,
        resume: Resume | None = None,
        run_tag: str,
    ) -> tuple[tuple[Op, ...], list[tuple[int, bool]], int, tinker.TrainingClient]:
        examples = await data.normalize(spec.dataset, method="sft")
        pool = [
            datum
            for example in examples
            if fits(datum := data.render_tinker_sft(example, spec.base.mlx), spec.hyperparams.max_seq_len)
        ]
        if len(pool) < spec.hyperparams.batch_size:
            raise InsufficientData(len(pool), spec.hyperparams.batch_size)
        base_step = resume.from_step if resume is not None else 0
        plan = batches(
            pool, size=spec.hyperparams.batch_size, steps=spec.hyperparams.steps, seed=spec.hyperparams.seed
        )[base_step:]
        schedule, meta = assemble(
            spec,
            [(batch, CrossEntropy()) for batch in plan],
            checkpoints=checkpoints,
            eval_datums=eval_datums,
            base_step=base_step,
            run_tag=run_tag,
        )
        await reservation.reserve(projection(schedule, price))
        client = await self.lora_client(
            self.service(), spec, model=model, seed=resume.handle if resume is not None else None
        )
        return schedule, meta, len(examples) - len(pool), client

    async def prepare_dpo(
        self,
        spec: TrainSpec,
        *,
        model: TinkerModelId,
        checkpoints: CheckpointPolicy,
        eval_datums: tuple[tinker.Datum, ...],
        reservation: Reservation,
        price: TinkerPrice,
        resume: Resume | None = None,
        run_tag: str,
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
        base_step = resume.from_step if resume is not None else 0
        plan = batches(
            range(len(pool)), size=spec.hyperparams.batch_size, steps=spec.hyperparams.steps, seed=spec.hyperparams.seed
        )[base_step:]
        reference_datums = pair_datums(pool)
        shape, meta = assemble(
            spec,
            [(tuple(pair_datums([pool[i] for i in batch])), Custom(dpo_loss)) for batch in plan],
            checkpoints=checkpoints,
            eval_datums=eval_datums,
            base_step=base_step,
            run_tag=run_tag,
        )
        await reservation.reserve(projection(shape, price) + price.prefill * token_count(reference_datums) / 1e6)

        service = self.service()
        reference = await reference_pass(
            await self.lora_client(service, spec, model=model, seed=resume.reference if resume is not None else None),
            reference_datums,
            reservation=reservation,
            price=price,
        )
        client = await self.lora_client(service, spec, model=model, seed=resume.handle if resume is not None else None)
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
            base_step=base_step,
            run_tag=run_tag,
        )
        return schedule, meta, len(examples) - len(pool), client

    async def train(
        self,
        spec: TrainSpec,
        *,
        sink: RunSink,
        work_dir: Path,
        resume: Resume | None = None,
        store: RunStateStore | None = None,
    ) -> Checkpoint:
        """Train ``spec`` on Tinker and fuse the resulting final adapter into a servable MLX model.

        The spend cap binds per run, not per attempt: a ``resume`` seeds the SpendGuard with the cost
        already billed under this run's key, so a crash-looping run never bills N times the cap.

        Args:
            spec: The fine-tuning request: base, dataset, hyperparams, method, LoRA, spend cap.
            sink: The run journal; one record lands per step with its loss, tokens, and spend.
            work_dir: This run's own directory; the adapter and the fused model are written under it.
            resume: The restore input for a same-run recovery or a cross-run continuation, or None to
                train from base.
            store: The run-state store the fit persists its progress to, or None for no resume ledger.

        Returns:
            The checkpoint, whose ``mlx_path`` is the fused standalone 4-bit MLX model.

        Raises:
            UnservableBase: The base has no Tinker id, or no mlx-lm counterpart to fuse into.
            UnsupportedLoraShape: The LoRA shape asks for something Tinker cannot train.
            LoraShapeDrift: A resume seed's saved LoRA shape disagrees with ``spec.lora``.
            TorchRequired: The method is ``dpo`` and torch is not installed locally.
            InsufficientData: The surviving pool is smaller than one batch.
            SpendExceeded: The projected run cost crosses the spend cap.
        """
        require_servable(spec.base, kind="Tinker")
        budget = SpendGuard(max_usd=spend_cap(spec, self.settings.spend_cap_usd))
        if resume is not None:
            await budget.record(0.0, resume.cost_usd)
        report = await self.fit(spec, sink=sink, budget=budget, resume=resume, store=store, run_tag=work_dir.name)
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
        peft = await download_adapter(self.service(), saved.sampler_path, work_dir / "peft")
        adapter_dir = await sidecar.convert_peft_to_mlx(peft, work_dir / "adapter", base=spec.base)
        return Adapter(
            step=saved.step,
            adapter_dir=adapter_dir,
            train_cost_usd=cost,
            sampler_path=saved.sampler_path,
            state=saved.state,
        )

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
            sampler_path=adapter.sampler_path,
            state=adapter.state,
        )

    async def score(
        self,
        path: str,
        rows: Sequence[EvalRow],
        *,
        base: BaseModelSpec,
        budget: SpendGuard,
    ) -> tuple[ScoredSequence, ...]:
        """Score weighted token sequences against a saved Tinker sampling checkpoint.

        Args:
            path: The opaque ``tinker://`` sampler checkpoint path.
            rows: Pre-tokenized weighted rows to score, in return order.
            base: The base model identity used to price the sampling requests.
            budget: The spend envelope; the projected prefill and one sampled token per row are
                reserved against it before any client exists, then reconciled once the rows score.

        Returns:
            One weighted sequence score per row, preserving input order.

        Raises:
            UnservableBase: The base has no Tinker identity.
            SpendExceeded: The projected prefill and one sampled token per row cross the envelope.
        """
        import tinker

        model = tinker_model(base)
        projected = self.cost(
            model=model,
            prefill=sum(len(row.tokens) for row in rows),
            sample=len(rows),
        )
        await budget.check(projected)
        settled = False
        try:
            client = await self.service().create_sampling_client_async(model_path=path)
            prompts = [tinker.ModelInput.from_ints(list(row.tokens)) for row in rows]
            outputs = await gather_bounded(
                [partial(client.compute_logprobs_async, prompt) for prompt in prompts],
                concurrency=64,
            )
            await budget.record(projected, projected)
            settled = True
            return tuple(
                score_sequence(
                    tuple(logprob for logprob in output[1:] if logprob is not None),
                    row.weights[1:],
                )
                for output, row in zip(outputs, rows, strict=True)
            )
        finally:
            if not settled:
                # Shield so a cancellation delivered mid-release cannot abort it and strand the reservation.
                with CancelScope(shield=True):
                    await budget.release(projected)

    async def sample(
        self,
        path: str | None,
        prompts: Sequence[Sequence[Message]],
        *,
        base: BaseModelSpec,
        budget: SpendGuard,
        max_tokens: int,
        temperature: float,
        seed: int | None = None,
    ) -> tuple[SampledSequence, ...]:
        """Sample free-form completions from a Tinker checkpoint, or the base model when ``path`` is None.

        Every prompt is rendered with the base's training chat template and a generation prompt, so a
        sample matches what the policy was trained to continue. The whole batch is projected against
        ``budget`` in one conservative reservation — full prefill plus ``max_tokens`` per prompt —
        before any sampling client exists, then reconciled to the real generated token counts, so a
        batch that cannot fit the envelope aborts having spent nothing and a short generation bills less.

        Args:
            path: The opaque ``tinker://`` sampler checkpoint to sample from, or None to sample the
                base model directly (iteration-zero negatives, before any checkpoint exists).
            prompts: The chat prompts to complete, one completion returned per prompt, in input order.
            base: The base model identity: its chat template tokenizes the prompts and its price sheet
                bills the run, and, when ``path`` is None, its Tinker id is the model sampled.
            budget: The spend envelope; the conservative full-prefill-plus-``max_tokens`` projection is
                reserved against it before any client exists, then reconciled to the real generated cost.
            max_tokens: The generation cap per prompt; also the per-prompt sample count the projection reserves.
            temperature: The sampling temperature applied to every prompt.
            seed: The base seed; prompt ``i`` samples with ``seed + i`` for reproducibility, or None to
                leave every prompt unseeded.

        Returns:
            One sampled sequence per prompt, preserving input order, each with its decoded text, token
            counts, and billed cost.

        Raises:
            UnservableBase: The base has no Tinker identity.
            SpendExceeded: The projected prefill and ``max_tokens`` per prompt cross the envelope.
        """
        import tinker

        model = tinker_model(base)
        tok = data.tokenizer(base.mlx)
        prompt_ids = [data.chat_ids(prompt, base.mlx, add_generation_prompt=True) for prompt in prompts]
        projected = self.cost(
            model=model,
            prefill=sum(len(ids) for ids in prompt_ids),
            sample=max_tokens * len(prompts),
        )
        await budget.check(projected)
        settled = False
        try:
            client = await (
                self.service().create_sampling_client_async(base_model=model)
                if path is None
                else self.service().create_sampling_client_async(model_path=path)
            )
            responses = await gather_bounded(
                [
                    partial(
                        client.sample_async,
                        tinker.ModelInput.from_ints(ids),
                        1,
                        tinker.SamplingParams(
                            max_tokens=max_tokens,
                            temperature=temperature,
                            seed=None if seed is None else seed + index,
                        ),
                    )
                    for index, ids in enumerate(prompt_ids)
                ],
                concurrency=64,
            )
            sampled = tuple(
                SampledSequence(
                    text=tok.decode(sequence.tokens),
                    prompt_tokens=len(ids),
                    sampled_tokens=len(sequence.tokens),
                    usd=self.cost(model=model, prefill=len(ids), sample=len(sequence.tokens)),
                )
                for ids, response in zip(prompt_ids, responses, strict=True)
                for sequence in (response.sequences[0],)
            )
            await budget.record(projected, sum(sequence.usd for sequence in sampled))
            settled = True
            return sampled
        finally:
            if not settled:
                # Shield so a cancellation delivered mid-release cannot abort it and strand the reservation.
                with CancelScope(shield=True):
                    await budget.release(projected)
