from __future__ import annotations

import importlib.util
import os
import random
import tarfile
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, ClassVar

import httpx
from anyio import to_thread

from athome.config import load
from athome.errors import AthomeError
from athome.llm.spend import SpendGuard
from athome.train import data, sidecar
from athome.train.backend import TINKER_ENV
from athome.train.spec import (
    Checkpoint,
    LoraSpec,
    TinkerSettings,
    UnservableBase,
    UnsupportedLoraShape,
    lora_keys,
    spend_cap,
    std_lora_keys,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    import tinker
    import torch

    from athome.progress import RunSink
    from athome.train.data import TinkerPreference
    from athome.train.spec import BackendName, BaseModelSpec, Method, TinkerModelId, TrainSpec

BETA = 0.1
DPO_PASSES = 2
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
    if not base.serves_locally:
        raise UnservableBase(f"{base.mlx} has no mlx-lm LoRA counterpart, so a Tinker adapter cannot be fused into it")
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


def token_count(datums: Iterable[tinker.Datum]) -> int:
    return sum(datum.model_input.length for datum in datums)


def weights_of(datum: tinker.Datum) -> Sequence[float]:
    return datum.loss_fn_inputs["weights"].data


def sequence_logprob(output: dict[str, tinker.TensorData], datum: tinker.Datum) -> float:
    return sum(logprob * weight for logprob, weight in zip(output["logprobs"].data, weights_of(datum), strict=True))


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

    Every run is projected against ``spend_cap_usd`` before its first billable call and
    charged step by step, so a run that cannot fit the cap aborts having spent nothing.

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

    async def train(self, spec: TrainSpec, *, sink: RunSink, work_dir: Path) -> Checkpoint:
        """Train ``spec`` on Tinker and fuse the resulting adapter into a servable MLX model.

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
            SpendExceeded: The projected run cost crosses the spend cap.
        """
        import tinker

        model = tinker_model(spec.base)
        tinker_lora(spec.lora)
        guard = SpendGuard(max_usd=spend_cap(spec, self.settings.spend_cap_usd))
        service = tinker.ServiceClient(api_key=self.settings.api_key.get_secret_value())
        match spec.method:
            case "sft":
                client = await self.run_sft(service, spec, model=model, guard=guard, sink=sink)
            case "dpo":
                client = await self.run_dpo(service, spec, model=model, guard=guard, sink=sink)
        saved = await client.save_weights_for_sampler_async(name=f"{spec.name}-{spec.method}-{spec.hyperparams.steps}")
        return await self.converge(
            service, spec, (await saved.result_async()).path, cost=guard.spent, work_dir=work_dir
        )

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

    def cost(self, tokens: int, *, model: TinkerModelId) -> float:
        return tokens / 1e6 * self.settings.price_per_mtok[model]

    async def run_sft(
        self,
        service: tinker.ServiceClient,
        spec: TrainSpec,
        *,
        model: TinkerModelId,
        guard: SpendGuard,
        sink: RunSink,
    ) -> tinker.TrainingClient:
        import tinker

        examples = await data.normalize(spec.dataset, method="sft")
        pool = [
            datum
            for example in examples
            if fits(datum := data.render_tinker_sft(example, spec.base.mlx), spec.hyperparams.max_seq_len)
        ]
        plan = batches(pool, size=spec.hyperparams.batch_size, steps=spec.hyperparams.steps, seed=spec.hyperparams.seed)
        await guard.check(self.cost(sum(token_count(batch) for batch in plan), model=model))

        client = await self.lora_client(service, spec, model=model)
        adam = tinker.AdamParams(learning_rate=spec.hyperparams.learning_rate)
        for step, batch in enumerate(plan, start=1):
            forward = await client.forward_backward_async(list(batch), "cross_entropy")
            output = await forward.result_async()
            await (await client.optim_step_async(adam)).result_async()
            spent = self.cost(token_count(batch), model=model)
            await guard.record(spent, spent)
            await sink.append(
                {
                    "step": step,
                    "method": "sft",
                    "loss": masked_loss(output.loss_fn_outputs, batch),
                    "tokens": token_count(batch),
                    "cost_usd": guard.spent,
                }
            )
        return client

    async def run_dpo(
        self,
        service: tinker.ServiceClient,
        spec: TrainSpec,
        *,
        model: TinkerModelId,
        guard: SpendGuard,
        sink: RunSink,
    ) -> tinker.TrainingClient:
        import tinker

        require_torch()
        examples = await data.normalize(spec.dataset, method="dpo")
        pool = [
            pref
            for example in examples
            if fits((pref := data.render_tinker_dpo(example, spec.base.mlx)).chosen, spec.hyperparams.max_seq_len)
            and fits(pref.rejected, spec.hyperparams.max_seq_len)
        ]
        plan = batches(
            range(len(pool)), size=spec.hyperparams.batch_size, steps=spec.hyperparams.steps, seed=spec.hyperparams.seed
        )
        stepped = sum(token_count(pair_datums([pool[index] for index in batch])) for batch in plan)
        await guard.check(self.cost(token_count(pair_datums(pool)) + DPO_PASSES * stepped, model=model))

        reference = await self.reference_logprobs(service, spec, pool, model=model, guard=guard)
        client = await self.lora_client(service, spec, model=model)
        adam = tinker.AdamParams(learning_rate=spec.hyperparams.learning_rate)
        for step, batch in enumerate(plan, start=1):
            datums = pair_datums([pool[index] for index in batch])
            frozen = [score for index in batch for score in reference[index]]
            loss_fn = partial(dpo_loss, reference=frozen, beta=BETA)
            forward = await client.forward_backward_custom_async(datums, loss_fn, loss_type_input="logprobs")
            output = await forward.result_async()
            await (await client.optim_step_async(adam)).result_async()
            spent = self.cost(DPO_PASSES * token_count(datums), model=model)
            await guard.record(spent, spent)
            await sink.append(
                {"step": step, "method": "dpo", "tokens": token_count(datums), "cost_usd": guard.spent}
                | {name: output.metrics[name] for name in ("loss", "margin", "accuracy")}
            )
        return client

    async def reference_logprobs(
        self,
        service: tinker.ServiceClient,
        spec: TrainSpec,
        pool: Sequence[TinkerPreference],
        *,
        model: TinkerModelId,
        guard: SpendGuard,
    ) -> list[tuple[float, float]]:
        frozen = await self.lora_client(service, spec, model=model)
        datums = pair_datums(pool)
        output = await (await frozen.forward_async(datums, "cross_entropy")).result_async()
        spent = self.cost(token_count(datums), model=model)
        await guard.record(spent, spent)
        scores = [
            sequence_logprob(logprobs, datum) for logprobs, datum in zip(output.loss_fn_outputs, datums, strict=True)
        ]
        return list(zip(scores[0::2], scores[1::2], strict=True))

    async def converge(
        self, service: tinker.ServiceClient, spec: TrainSpec, path: str, *, cost: float, work_dir: Path
    ) -> Checkpoint:
        peft = await download_adapter(service, path, work_dir / "peft")
        adapter = await sidecar.convert_peft_to_mlx(peft, work_dir / "adapter", base=spec.base)
        return Checkpoint(
            base=spec.base,
            backend="tinker",
            method=spec.method,
            step=spec.hyperparams.steps,
            mlx_path=await sidecar.fuse(adapter, work_dir / "mlx", base=spec.base),
            adapter_dir=adapter,
            train_cost_usd=cost,
        )
