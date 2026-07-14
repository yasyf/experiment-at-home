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
    HfRepoId,
    Hyperparams,
    LoraSpec,
    Method,
    ModalTrainSettings,
    TrainSettings,
    TrainSpec,
)

if TYPE_CHECKING:
    from datasets import Dataset
    from peft import LoraConfig

    from athome.progress import RunSink
    from athome.wire import Wire

PYTHON = "3.13"
HF_HUB_CACHE = "/models/hf"
ADAPTER_DIR = "/tmp/adapter"
PACKAGES: tuple[str, ...] = ("trl", "peft", "torch", "transformers")
STARTUP_SECONDS = 300.0
LORA_TOKENS_PER_SECOND = 4000.0
MODAL_MAX_TIMEOUT = 86400


def pinned_versions(settings: ModalTrainSettings) -> dict[str, str]:
    return dict(
        zip(
            PACKAGES,
            (settings.trl_version, settings.peft_version, settings.torch_version, settings.transformers_version),
            strict=True,
        )
    )


def lora_params(lora: LoraSpec) -> dict[str, Wire]:
    return {
        "lora_rank": lora.rank,
        "lora_alpha": lora.alpha,
        "lora_dropout": lora.dropout,
        "lora_target_modules": sorted(lora.target_modules),
    }


def service_spec(settings: ModalTrainSettings, lora: LoraSpec) -> ServiceSpec:
    """The parity fingerprint's local half: the image's version pins folded with the LoRA shape.

    The pins are declared in ``[train.modal]``, not installed here — trl, peft, torch and
    transformers live only inside the Modal image — so they travel as params rather than as
    :attr:`~athome.modal.ServiceSpec.version_packages`, which reads local metadata.
    """
    return ServiceSpec(
        name=settings.app_name, version_packages=(), params=pinned_versions(settings) | lora_params(lora)
    )


@dataclass(frozen=True, slots=True)
class RemoteConfig:
    """Everything the GPU container needs to train one adapter and push it."""

    base_hf: HfRepoId
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
        target_modules=list(lora.target_modules),
        task_type="CAUSAL_LM",
    )


def resolved_lora(config: LoraConfig) -> LoraSpec:
    return LoraSpec(
        rank=config.r,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        target_modules=tuple(config.target_modules),
    )


def download_base(repo: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo)


def fingerprint_remote(config: RemoteConfig) -> dict[str, Wire]:
    return {f"param:{package}": importlib.metadata.version(package) for package in PACKAGES} | {
        f"param:{key}": value for key, value in lora_params(resolved_lora(lora_config(config.lora))).items()
    }


def train_remote(config: RemoteConfig, dataset: Dataset) -> RemoteResult:
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    started = time.monotonic()
    hyper = config.hyperparams
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
        "model": AutoModelForCausalLM.from_pretrained(config.base_hf, torch_dtype="bfloat16", device_map="cuda"),
        "processing_class": AutoTokenizer.from_pretrained(config.base_hf),
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
        .pip_install(
            *(f"{package}=={version}" for package, version in pinned_versions(settings).items()),
            "datasets",
            "huggingface_hub",
        )
        .env({"HF_HUB_CACHE": HF_HUB_CACHE})
        .add_local_python_source("athome", copy=True)
        .run_function(download_base, args=(base.hf,))
    )


def budget_seconds(max_usd: float, settings: ModalTrainSettings) -> int:
    """The wall-clock the cap buys, as the GPU function's timeout: a run is killed before it outspends it."""
    return min(int(max_usd / settings.gpu_usd_per_hour * 3600), MODAL_MAX_TIMEOUT)


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
    return (STARTUP_SECONDS + tokens / LORA_TOKENS_PER_SECOND) / 3600 * settings.gpu_usd_per_hour


async def train_dataset(spec: TrainSpec) -> Dataset:
    match spec.method:
        case "sft":
            return render_trl(await normalize(spec.dataset, method="sft"), method="sft")
        case "dpo":
            return render_trl(await normalize(spec.dataset, method="dpo"), method="dpo")


@dataclass(frozen=True, slots=True)
class ModalTrainBackend:
    """Trains a LoRA adapter with TRL on a Modal GPU, converging on a fused MLX artifact.

    The image pins trl, peft, torch and transformers; before the GPU is touched the container
    reports those versions back along with the LoRA hyperparams peft itself resolved, and any
    drift from the local pins raises :class:`~athome.modal.ParityMismatch`. There is no local
    fallback — a skewed remote is not comparable with the other backends, so the run dies.

    The spend cap binds twice: the projection is reserved before launch, and the same budget
    caps the GPU function's timeout, so the run is killed rather than allowed to outspend it.

    Example:
        >>> checkpoint = await ModalTrainBackend.from_settings().train(spec, sink=sink)
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

    async def train(self, spec: TrainSpec, *, sink: RunSink) -> Checkpoint:
        """Train ``spec`` on a Modal GPU and return the fused MLX model it converged on.

        The adapter TRL trains is pushed to ``{hf_repo_prefix}/{spec.name}`` from the container,
        snapshotted back at the commit it landed on, converted from PEFT to an mlx-lm adapter,
        and fused into the base — the one artifact ``rapid-mlx serve`` can serve.

        Args:
            spec: The fine-tuning request: base, dataset, method, LoRA shape, and spend cap.
            sink: The run journal; the launch, the trained adapter, and the fused artifact land here.

        Returns:
            The :class:`~athome.train.spec.Checkpoint` naming the fused MLX directory, the mlx-lm
            adapter it was fused from, and what the GPU actually billed.

        Raises:
            SpendExceeded: The projected cost crosses the cap, or the run billed past it.
            ParityMismatch: The container's TRL stack or resolved LoRA shape drifted from the pins.
            HfAuthError: ``HF_TOKEN`` cannot push the adapter; nothing is launched.
        """
        import modal

        await ensure_write_auth()
        guard = SpendGuard(max_usd=spec.max_usd or self.settings.spend_cap_usd)
        await guard.check(projected := projected_usd(spec, self.settings))
        config = RemoteConfig(
            base_hf=spec.base.hf,
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
            timeout=budget_seconds(guard.max_usd, self.settings),
            serialized=True,
            secrets=[modal.Secret.from_dict({"HF_TOKEN": load(HfSettings).token.get_secret_value()})],
        )(train_remote)
        await sink.append({"stage": "launch", "gpu": self.settings.gpu, "projected_usd": projected})
        async with app.run():
            if mismatches := parity_mismatches(
                service_spec(self.settings, spec.lora), await fingerprint.remote.aio(config)
            ):
                raise ParityMismatch(f"{self.settings.app_name}: " + "; ".join(mismatches))
            result: RemoteResult = await trainer.remote.aio(config, dataset)
        await guard.record(projected, actual := result.seconds / 3600 * self.settings.gpu_usd_per_hour)
        if guard.spent > guard.max_usd:
            raise SpendExceeded(f"modal run billed ${guard.spent:.4f}, over the ${guard.max_usd:.4f} cap")
        await sink.append(
            {"stage": "trained", "repo": result.repo, "revision": result.revision, "usd": actual, "step": result.step}
        )
        run_dir = load(TrainSettings).work_root / spec.name / self.name
        adapter_dir = await convert_peft_to_mlx(
            await snapshot(result.repo, revision=result.revision),
            run_dir / "adapter",
            base=spec.base,
            lora=spec.lora,
        )
        mlx_path = await fuse(adapter_dir, run_dir / "mlx", base=spec.base)
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
