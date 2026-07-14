from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, NewType

from pydantic import Field, SecretStr

from athome.config import SectionSettings

if TYPE_CHECKING:
    from athome.bakeoff import Leaderboard
    from athome.registry import VersionInfo

MlxModelId = NewType("MlxModelId", str)
TinkerModelId = NewType("TinkerModelId", str)
HfRepoId = NewType("HfRepoId", str)

Method = Literal["sft", "dpo"]
BackendName = Literal["tinker", "local", "modal"]

STD_MODULES: tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
SEED = 1729


@dataclass(frozen=True, slots=True)
class BaseModelSpec:
    """One base model addressed across every backend.

    Attributes:
        mlx: The 4-bit MLX id used for local serving and PEFT-to-MLX conversion.
        hf: The HuggingFace repo for modal (GPU) training and snapshotting.
        tinker: The Tinker base id, or None when Tinker cannot train it.
        num_layers: Layer count, needed to write an mlx-lm ``adapter_config.json``.
        serves_locally: False when the LoRA cannot load into mlx-lm — a split
            ``linear_attn`` maps to a fused ``in_proj_qkv``, so the adapter has no
            mlx-lm counterpart.
    """

    mlx: MlxModelId
    hf: HfRepoId
    tinker: TinkerModelId | None
    num_layers: int
    serves_locally: bool = True


BASE_MODELS: dict[str, BaseModelSpec] = {
    "qwen3-8b": BaseModelSpec(
        mlx=MlxModelId("mlx-community/Qwen3-8B-4bit"),
        hf=HfRepoId("Qwen/Qwen3-8B"),
        tinker=TinkerModelId("Qwen/Qwen3-8B"),
        num_layers=36,
        serves_locally=True,
    ),
    "qwen3.5-4b": BaseModelSpec(
        mlx=MlxModelId("mlx-community/Qwen3.5-4B-4bit"),
        hf=HfRepoId("Qwen/Qwen3.5-4B"),
        tinker=TinkerModelId("Qwen/Qwen3.5-4B"),
        num_layers=32,
        serves_locally=False,
    ),
}


@dataclass(frozen=True, slots=True)
class HfDatasetRef:
    """A cc-steer HuggingFace export: one config of one dataset repo.

    Attributes:
        repo: The ``owner/name`` dataset repo id.
        config: The export config name (a cc-steer export ships ``sft`` and ``dpo``).
        split: The split to train on.
    """

    repo: HfRepoId
    config: str
    split: str = "train"


@dataclass(frozen=True, slots=True)
class LocalJsonlRef:
    """A local jsonl corpus: one ``.jsonl`` file, or a directory of them.

    Attributes:
        path: The jsonl file, or a directory whose ``*.jsonl`` files are read in
            sorted order. SFT rows are mlx-lm chat rows (``{"messages": [...]}``);
            DPO rows carry ``prompt``/``chosen``/``rejected``.
    """

    path: Path


type DatasetSource = HfDatasetRef | LocalJsonlRef


@dataclass(frozen=True, slots=True)
class LoraSpec:
    """The LoRA adapter's shape: its rank, scale, and which modules it wraps."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: tuple[str, ...] = STD_MODULES
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = False


@dataclass(frozen=True, slots=True)
class Hyperparams:
    """The optimization knobs shared by every backend."""

    steps: int
    batch_size: int = 4
    learning_rate: float = 1e-4
    max_seq_len: int = 4096
    seed: int = SEED


@dataclass(frozen=True, slots=True)
class TrainSpec:
    """One fine-tuning request, backend-agnostic.

    Attributes:
        name: The registry family the trained artifact is registered under.
        base: The base model, addressed across backends.
        dataset: Where the training data comes from.
        hyperparams: Steps, batch, learning rate, sequence length, seed.
        method: SFT or DPO. ``local`` trains SFT only.
        lora: LoRA rank, alpha, and target modules.
        backend: An explicit backend override; None selects by availability.
        max_usd: A per-run spend cap for the metered backends (tinker, modal);
            None falls back to that backend's configured ``spend_cap_usd``.
    """

    name: str
    base: BaseModelSpec
    dataset: DatasetSource
    hyperparams: Hyperparams
    method: Method = "sft"
    lora: LoraSpec = field(default_factory=LoraSpec)
    backend: BackendName | None = None
    max_usd: float | None = None


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """The servable artifact one backend produced: a standalone 4-bit MLX model directory.

    ``mlx_path`` is the only serve path — ``rapid-mlx serve`` takes a single model
    and has no adapter flag, so every backend fuses its LoRA into the base.
    ``adapter_dir`` is the intermediate adapter kept for provenance (tinker and
    modal train a PEFT adapter first); it is never served.

    Attributes:
        base: The base model the LoRA was trained over.
        backend: The backend that produced it.
        method: The method it was trained with.
        step: The step count the weights were saved at.
        mlx_path: The fused standalone 4-bit MLX model directory.
        adapter_dir: The mlx-lm adapter directory, when one was materialized.
        train_cost_usd: What the run spent; 0.0 on the unmetered local backend.
    """

    base: BaseModelSpec
    backend: BackendName
    method: Method
    step: int
    mlx_path: Path
    adapter_dir: Path | None
    train_cost_usd: float


@dataclass(frozen=True, slots=True)
class TrainResult:
    """The outcome of one ``run``: the artifact, its eval scalar, and its registry entry.

    Attributes:
        checkpoint: The servable artifact.
        metric: The trained arm's primary metric on the evaluation bake-off.
        leaderboard: The full bake-off result the metric came from.
        version: The registry entry the artifact was registered as.
        promoted: True when this artifact became the family's ``current``.
    """

    checkpoint: Checkpoint
    metric: float
    leaderboard: Leaderboard
    version: VersionInfo
    promoted: bool


class TrainSettings(SectionSettings):
    """Backend selection, the mlx-lm sidecar pin, and the train roots, bound to ``[train]``."""

    section: ClassVar[tuple[str, ...]] = ("train",)
    backend: BackendName | None = None
    mlx_lm_version: str = "0.31.3"
    registry_root: Path = Path("~/.athome/train/registry")
    work_root: Path = Path("~/.athome/train/runs")


class TinkerSettings(SectionSettings):
    """Tinker credentials, spend cap, and per-model training prices, bound to ``[train.tinker]``."""

    section: ClassVar[tuple[str, ...]] = ("train", "tinker")
    api_key: SecretStr = Field(validation_alias="TINKER_API_KEY")
    spend_cap_usd: float = 60.0
    price_per_mtok: dict[str, float] = Field(default_factory=lambda: {"Qwen/Qwen3-8B": 0.40, "Qwen/Qwen3.5-4B": 0.67})


class LocalTrainSettings(SectionSettings):
    """The local mlx-lm trainer's knobs, bound to ``[train.local]``."""

    section: ClassVar[tuple[str, ...]] = ("train", "local")
    val_fraction: float = 0.1
    grad_checkpoint: bool = True


class ModalTrainSettings(SectionSettings):
    """The Modal GPU trainer's image pins, GPU class, and spend cap, bound to ``[train.modal]``."""

    section: ClassVar[tuple[str, ...]] = ("train", "modal")
    app_name: str = "athome-train"
    gpu: str = "H100"
    gpu_usd_per_hour: float = 3.95
    spend_cap_usd: float = 60.0
    hf_repo_prefix: str = "athome-train"
    trl_version: str = "0.21.0"
    peft_version: str = "0.14.0"
    torch_version: str = "2.6.0"
    transformers_version: str = "4.48.0"
