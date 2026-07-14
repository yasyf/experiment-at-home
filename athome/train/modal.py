from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from athome.config import load
from athome.hf import HfSettings, ensure_write_auth, snapshot
from athome.llm.spend import SpendExceeded, SpendGuard
from athome.modal import ParityMismatch, ServiceSpec, parity_mismatches
from athome.train.data import normalize, render_trl
from athome.train.sidecar import convert_peft_to_mlx, fuse
from athome.train.spec import (
    BackendName,
    BaseModelSpec,
    Checkpoint,
    Hyperparams,
    LoraSpec,
    Method,
    ModalTrainSettings,
    TrainSpec,
    UnservableBase,
    lora_keys,
    spend_cap,
)

if TYPE_CHECKING:
    from datasets import Dataset
    from peft import LoraConfig

    from athome.progress import RunSink
    from athome.wire import Wire

PYTHON = "3.13"
HF_HUB_CACHE = "/models/hf"
ADAPTER_DIR = "/tmp/adapter"
PACKAGES: tuple[str, ...] = ("trl", "peft", "torch", "transformers", "datasets")
STARTUP_SECONDS = 300.0
OVERSHOOT_SECONDS = 60.0
LORA_TOKENS_PER_SECOND = 4000.0
MODAL_MAX_TIMEOUT = 86400


def pinned_versions(settings: ModalTrainSettings) -> dict[str, str]:
    return dict(
        zip(
            PACKAGES,
            (
                settings.trl_version,
                settings.peft_version,
                settings.torch_version,
                settings.transformers_version,
                settings.datasets_version,
            ),
            strict=True,
        )
    )


def lora_params(lora: LoraSpec) -> dict[str, Wire]:
    return {
        "lora_rank": lora.rank,
        "lora_alpha": lora.alpha,
        "lora_dropout": lora.dropout,
        "lora_target_modules": sorted(lora_keys(lora)),
        "lora_train_mlp": lora.train_mlp,
        "lora_train_attn": lora.train_attn,
        "lora_train_unembed": lora.train_unembed,
    }


def base_params(base: BaseModelSpec) -> dict[str, Wire]:
    return {
        "base_hf": base.hf,
        "base_hf_revision": base.hf_revision,
        "base_mlx": base.mlx,
        "base_mlx_revision": base.mlx_revision,
    }


def service_spec(settings: ModalTrainSettings, lora: LoraSpec, base: BaseModelSpec) -> ServiceSpec:
    """The parity fingerprint's local half: the image's pins folded with the LoRA shape and the base.

    The pins are declared in ``[train.modal]``, not installed here — trl, peft, torch,
    transformers and datasets live only inside the Modal image — so they travel as params rather
    than as :attr:`~athome.modal.ServiceSpec.version_packages`, which reads local metadata.

    The base's pinned commits ride along because a fingerprint that names only the repo cannot
    tell two runs over different weights apart.
    """
    return ServiceSpec(
        name=settings.app_name,
        version_packages=(),
        params=pinned_versions(settings) | lora_params(lora) | base_params(base),
    )


@dataclass(frozen=True, slots=True)
class RemoteConfig:
    """Everything the GPU container needs to train one adapter and push it."""

    base: BaseModelSpec
    repo: str
    method: Method
    lora: LoraSpec
    hyperparams: Hyperparams


@dataclass(frozen=True, slots=True)
class RemoteResult:
    """What the GPU container reports back: where the adapter landed, and what it billed."""

    repo: str
    revision: str
    step: int
    seconds: float


def lora_config(lora: LoraSpec) -> LoraConfig:
    from peft import LoraConfig

    return LoraConfig(
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora_keys(lora)),
        task_type="CAUSAL_LM",
    )


def resolved_lora(config: LoraConfig, lora: LoraSpec) -> LoraSpec:
    """The LoRA shape peft itself resolved, carrying back the toggles peft has no field for."""
    return LoraSpec(
        rank=config.r,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        target_modules=tuple(sorted(config.target_modules)),
        train_mlp=lora.train_mlp,
        train_attn=lora.train_attn,
        train_unembed=lora.train_unembed,
    )


def download_base(repo: str, revision: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo, revision=revision)


def baked_commit(base: BaseModelSpec) -> str:
    """The commit of the base snapshot baked into this image, read off the cache it was baked into.

    ``local_files_only`` never reaches the hub, so this reports what the image actually carries
    rather than echoing what it was asked for: an image built over a different revision has no
    snapshot to resolve and fails here, before the GPU function is ever called.
    """
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(base.hf, revision=base.hf_revision, local_files_only=True)).name


def fingerprint_remote(config: RemoteConfig) -> dict[str, Wire]:
    params = (
        lora_params(resolved_lora(lora_config(config.lora), config.lora))
        | base_params(config.base)
        | {"base_hf_revision": baked_commit(config.base)}
    )
    return {f"param:{package}": importlib.metadata.version(package) for package in PACKAGES} | {
        f"param:{key}": value for key, value in params.items()
    }


def train_remote(config: RemoteConfig, dataset: Dataset) -> RemoteResult:
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    started = time.monotonic()
    hyper = config.hyperparams
    base, revision = config.base.hf, config.base.hf_revision
    args = {
        "output_dir": ADAPTER_DIR,
        "max_steps": hyper.steps,
        "per_device_train_batch_size": hyper.batch_size,
        "learning_rate": hyper.learning_rate,
        "max_length": hyper.max_seq_len,
        "seed": hyper.seed,
        "report_to": [],
    }
    common = {
        "model": AutoModelForCausalLM.from_pretrained(
            base, revision=revision, torch_dtype="bfloat16", device_map="cuda"
        ),
        "processing_class": AutoTokenizer.from_pretrained(base, revision=revision),
        "train_dataset": dataset,
        "peft_config": lora_config(config.lora),
    }
    match config.method:
        case "sft":
            trainer = SFTTrainer(args=SFTConfig(**args), **common)
        case "dpo":
            trainer = DPOTrainer(args=DPOConfig(**args), **common)
    trainer.train()
    trainer.save_model(ADAPTER_DIR)
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(config.repo, private=True, exist_ok=True)
    commit = api.upload_folder(repo_id=config.repo, folder_path=ADAPTER_DIR)
    return RemoteResult(repo=config.repo, revision=commit.oid, step=hyper.steps, seconds=time.monotonic() - started)


def train_image(settings: ModalTrainSettings, base: BaseModelSpec) -> object:
    """Build the training image: athome's own env, the pinned TRL stack, and the base weights baked in."""
    import modal

    return (
        modal.Image.debian_slim(python_version=PYTHON)
        .uv_sync()
        .uv_pip_install(
            *(f"{package}=={version}" for package, version in pinned_versions(settings).items()), "huggingface_hub"
        )
        .env({"HF_HUB_CACHE": HF_HUB_CACHE})
        .add_local_python_source("athome", copy=True)
        .run_function(download_base, args=(base.hf, base.hf_revision))
    )


def billed_usd(seconds: float, settings: ModalTrainSettings) -> float:
    """What Modal bills for a GPU function that ran ``seconds``: the container's whole life.

    The container is paid for from boot, but the function's own timer only starts once it is
    running, so :data:`STARTUP_SECONDS` is added to every figure the cap is measured against —
    projection and settlement alike, which is what makes the two comparable.
    """
    return (STARTUP_SECONDS + seconds) / 3600 * settings.gpu_usd_per_hour


def budget_seconds(max_usd: float, settings: ModalTrainSettings) -> int:
    """The GPU seconds ``max_usd`` buys, as the function's timeout: the run dies before it outspends the cap.

    The cap has to pay for the container's startup and for Modal's own slack — a function is
    killed *shortly after* its timeout, not at it — so the timeout is what is left of the cap
    once :data:`STARTUP_SECONDS` and :data:`OVERSHOOT_SECONDS` are taken out of it. A run that
    burns every second of it, plus the overshoot, still settles under the cap.

    Raises:
        SpendExceeded: The cap cannot even pay for the container's startup and overshoot, so
            there is no GPU time to grant.
    """
    seconds = max_usd / settings.gpu_usd_per_hour * 3600 - STARTUP_SECONDS - OVERSHOOT_SECONDS
    if seconds < 1:
        raise SpendExceeded(
            f"a ${max_usd:.4f} cap buys no GPU time: container startup and timeout overshoot alone cost "
            f"${billed_usd(OVERSHOOT_SECONDS, settings):.4f}"
        )
    return min(int(seconds), MODAL_MAX_TIMEOUT)


def projected_usd(spec: TrainSpec, settings: ModalTrainSettings) -> float:
    """Project the run's GPU bill: container startup, plus the LoRA passes over the corpus.

    A DPO step scores a chosen and a rejected continuation, so it moves twice the tokens an
    SFT step does.
    """
    hyper = spec.hyperparams
    match spec.method:
        case "sft":
            passes = 1
        case "dpo":
            passes = 2
    tokens = passes * hyper.steps * hyper.batch_size * hyper.max_seq_len
    return billed_usd(tokens / LORA_TOKENS_PER_SECOND, settings)


async def train_dataset(spec: TrainSpec) -> Dataset:
    match spec.method:
        case "sft":
            return render_trl(await normalize(spec.dataset, method="sft"), method="sft")
        case "dpo":
            return render_trl(await normalize(spec.dataset, method="dpo"), method="dpo")


@dataclass(frozen=True, slots=True)
class ModalTrainBackend:
    """Trains a LoRA adapter with TRL on a Modal GPU, converging on a fused MLX artifact.

    The image pins the TRL stack; before the GPU is touched the container reports those versions
    back, along with the LoRA shape peft itself resolved and the commit of the base weights baked
    into the image, and any drift from the local pins raises
    :class:`~athome.modal.ParityMismatch`. There is no local fallback — a skewed remote is not
    comparable with the other backends, so the run dies.

    The spend cap binds twice: the projection is reserved before launch, and what is left of the
    cap after startup and overshoot becomes the GPU function's timeout, so a run is killed rather
    than allowed to outspend it.

    Example:
        >>> checkpoint = await ModalTrainBackend.from_settings().train(spec, sink=sink, work_dir=work_dir)
    """

    settings: ModalTrainSettings
    name: ClassVar[BackendName] = "modal"

    @staticmethod
    def available() -> bool:
        """Whether modal is installed and credentialed: a token in the env, or ``~/.modal.toml``."""
        return importlib.util.find_spec("modal") is not None and (
            bool(os.environ.get("MODAL_TOKEN_ID")) or (Path.home() / ".modal.toml").exists()
        )

    @staticmethod
    def supports(method: Method) -> bool:
        """TRL trains both: ``SFTTrainer`` for ``sft``, ``DPOTrainer`` for ``dpo``."""
        return method in {"sft", "dpo"}

    @classmethod
    def from_settings(cls) -> ModalTrainBackend:
        """Bind the ``[train.modal]`` section."""
        return cls(load(ModalTrainSettings))

    async def train(self, spec: TrainSpec, *, sink: RunSink, work_dir: Path) -> Checkpoint:
        """Train ``spec`` on a Modal GPU and return the fused MLX model it converged on.

        The adapter TRL trains is pushed to ``{hf_repo_prefix}/{spec.name}`` from the container,
        snapshotted back at the commit it landed on, converted from PEFT to an mlx-lm adapter,
        and fused into the base — the one artifact ``rapid-mlx serve`` can serve.

        Everything that can refuse the run refuses it before the first billable operation: a base
        that cannot be fused locally, a LoRA shape mlx-lm cannot express, and a cap the projection
        already crosses.

        Args:
            spec: The fine-tuning request: base, dataset, method, LoRA shape, and spend cap.
            sink: The run journal; the launch, the trained adapter, and the fused artifact land here.
            work_dir: This run's own directory; the adapter and the fused model are written under it.

        Returns:
            The :class:`~athome.train.spec.Checkpoint` naming the fused MLX directory, the mlx-lm
            adapter it was fused from, and what the GPU actually billed.

        Raises:
            UnservableBase: The base has no mlx-lm counterpart to fuse into; nothing is launched.
            UnsupportedLoraShape: The LoRA shape cannot survive the fuse; nothing is launched.
            SpendExceeded: The projected cost crosses the cap, or the run billed past it.
            ParityMismatch: The container's TRL stack, resolved LoRA shape, or baked base weights
                drifted from the pins.
            HfAuthError: ``HF_TOKEN`` cannot push the adapter; nothing is launched.
        """
        import modal

        if not spec.base.serves_locally:
            raise UnservableBase(
                f"{spec.base.mlx} has no mlx-lm LoRA counterpart, so a Modal adapter cannot be fused into it"
            )
        parity = service_spec(self.settings, spec.lora, spec.base)
        await ensure_write_auth()
        guard = SpendGuard(max_usd=spend_cap(spec, self.settings.spend_cap_usd))
        await guard.check(projected := projected_usd(spec, self.settings))
        timeout = budget_seconds(guard.max_usd, self.settings)
        config = RemoteConfig(
            base=spec.base,
            repo=f"{self.settings.hf_repo_prefix}/{spec.name}",
            method=spec.method,
            lora=spec.lora,
            hyperparams=spec.hyperparams,
        )
        dataset = await train_dataset(spec)
        app = modal.App(self.settings.app_name, image=train_image(self.settings, spec.base))
        fingerprint = app.function(serialized=True)(fingerprint_remote)
        trainer = app.function(
            gpu=self.settings.gpu,
            timeout=timeout,
            serialized=True,
            secrets=[modal.Secret.from_dict({"HF_TOKEN": load(HfSettings).token.get_secret_value()})],
        )(train_remote)
        await sink.append({"stage": "launch", "gpu": self.settings.gpu, "projected_usd": projected, "timeout": timeout})
        async with app.run():
            if mismatches := parity_mismatches(parity, await fingerprint.remote.aio(config)):
                raise ParityMismatch(f"{self.settings.app_name}: " + "; ".join(mismatches))
            result: RemoteResult = await trainer.remote.aio(config, dataset)
        await guard.record(projected, actual := billed_usd(result.seconds, self.settings))
        if guard.spent > guard.max_usd:
            raise SpendExceeded(f"modal run billed ${guard.spent:.4f}, over the ${guard.max_usd:.4f} cap")
        await sink.append(
            {"stage": "trained", "repo": result.repo, "revision": result.revision, "usd": actual, "step": result.step}
        )
        adapter_dir = await convert_peft_to_mlx(
            await snapshot(result.repo, revision=result.revision), work_dir / "adapter", base=spec.base
        )
        mlx_path = await fuse(adapter_dir, work_dir / "mlx", base=spec.base)
        await sink.append({"stage": "converged", "mlx_path": str(mlx_path)})
        return Checkpoint(
            base=spec.base,
            backend=self.name,
            method=spec.method,
            step=result.step,
            mlx_path=mlx_path,
            adapter_dir=adapter_dir,
            train_cost_usd=actual,
        )
