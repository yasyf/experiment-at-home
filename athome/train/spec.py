from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, NewType

from pydantic import Field, SecretStr

from athome.config import SectionSettings
from athome.errors import AthomeError

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
ATTN_PREFIX = "self_attn."
MLP_PREFIX = "mlp."
SEED = 1729


class UnservableBase(AthomeError):
    """Raised when the spec's base model cannot produce a servable fused MLX artifact."""


class UnsupportedLoraShape(AthomeError):
    """Raised when a :class:`LoraSpec` asks for an adapter the MLX fuse path cannot express."""


@dataclass(frozen=True, slots=True)
class BaseModelSpec:
    """One base model addressed across every backend, pinned to exact weights.

    Every repo id carries its commit: an unpinned id floats with the hub's head, so two
    runs could train against different weights — or an adapter trained on one revision
    could be fused into an independently-updated MLX snapshot — while their fingerprints
    claimed to be the same run.

    Attributes:
        mlx: The 4-bit MLX id used for local serving and PEFT-to-MLX conversion.
        hf: The HuggingFace repo for modal (GPU) training and snapshotting.
        hf_revision: The commit the ``hf`` weights are pinned to.
        mlx_revision: The commit the ``mlx`` weights are pinned to.
        tinker: The Tinker base id, or None when Tinker cannot train it.
        num_layers: Layer count, needed to write an mlx-lm ``adapter_config.json``.
        serves_locally: False when the LoRA cannot load into mlx-lm — a split
            ``linear_attn`` maps to a fused ``in_proj_qkv``, so the adapter has no
            mlx-lm counterpart.
    """

    mlx: MlxModelId
    hf: HfRepoId
    hf_revision: str
    mlx_revision: str
    tinker: TinkerModelId | None
    num_layers: int
    serves_locally: bool = True


BASE_MODELS: dict[str, BaseModelSpec] = {
    "qwen3-8b": BaseModelSpec(
        mlx=MlxModelId("mlx-community/Qwen3-8B-4bit"),
        hf=HfRepoId("Qwen/Qwen3-8B"),
        hf_revision="b968826d9c46dd6066d109eabc6255188de91218",
        mlx_revision="545dc4251c05440727734bcd94334791f6ab0192",
        tinker=TinkerModelId("Qwen/Qwen3-8B"),
        num_layers=36,
        serves_locally=True,
    ),
    "qwen3.5-4b": BaseModelSpec(
        mlx=MlxModelId("mlx-community/Qwen3.5-4B-4bit"),
        hf=HfRepoId("Qwen/Qwen3.5-4B"),
        hf_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        mlx_revision="0e7ffd5c629ef7719d4cbc04069232580bfa9d9c",
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


def lora_keys(lora: LoraSpec) -> tuple[str, ...]:
    """The modules an adapter actually wraps: ``target_modules`` filtered by the LoRA toggles.

    Every backend converges on a fused mlx-lm artifact, so every backend is bound by what
    mlx-lm can express — which makes this the one shape rule, not a per-backend one.

    Args:
        lora: The requested LoRA shape.

    Returns:
        The subset of ``target_modules`` the toggles select, in the spec's order.

    Raises:
        UnsupportedLoraShape: ``train_unembed`` is set, or the toggles leave nothing to
            train. mlx-lm reaches the unembedding through a base-dependent module path —
            ``lm_head``, which a base that ties its embeddings does not have at all — and
            :class:`BaseModelSpec` does not carry it, so no backend can fuse an
            unembedding LoRA: training one only burns money on weights the PEFT-to-MLX
            conversion would drop.
    """
    if lora.train_unembed:
        raise UnsupportedLoraShape(
            "no backend can fuse an unembedding LoRA: mlx-lm reaches the unembedding through a "
            "base-dependent module path (`lm_head`, absent on a base that ties its embeddings) and "
            "BaseModelSpec does not carry it, so the conversion would drop the tensor it trained."
        )
    prefixes = tuple(
        prefix for prefix, trains in ((ATTN_PREFIX, lora.train_attn), (MLP_PREFIX, lora.train_mlp)) if trains
    )
    if not (keys := tuple(key for key in lora.target_modules if key.startswith(prefixes))):
        raise UnsupportedLoraShape(
            f"nothing to train: train_attn={lora.train_attn}, train_mlp={lora.train_mlp} "
            f"leave no trainable module in {lora.target_modules}"
        )
    return keys


def std_lora_keys(lora: LoraSpec) -> tuple[str, ...]:
    """The standard modules ``lora``'s toggles select, ignoring its ``target_modules``.

    What a backend that takes only toggles — Tinker — trains for this spec.
    """
    return lora_keys(replace(lora, target_modules=STD_MODULES))


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


def spend_cap(spec: TrainSpec, default: float) -> float:
    """The dollar cap binding this run: the spec's ``max_usd`` when it set one, else ``default``.

    Only ``None`` means "unset". ``max_usd=0.0`` is an explicit "spend nothing" and binds as
    such — a falsy-zero fallback here would silently authorize the whole configured cap.
    """
    return default if spec.max_usd is None else spec.max_usd


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


@dataclass(frozen=True, slots=True)
class TinkerPrice:
    """One Tinker base model's per-million-token price, split by the token class it bills.

    Tinker meters three classes at three rates, so a run costs the sum over the classes it
    actually billed — a flat rate would overcharge a forward-only pass at the training rate.

    Attributes:
        prefill: Tokens through a forward-only pass, such as a frozen reference policy's logprobs.
        sample: Tokens generated by a sampling client.
        train: Tokens through a ``forward_backward`` and its ``optim_step``.

    Example:
        >>> TinkerPrice(prefill=0.13, sample=0.40, train=0.40)
    """

    prefill: float
    sample: float
    train: float


class TinkerSettings(SectionSettings):
    """Tinker credentials, spend cap, and per-model token prices, bound to ``[train.tinker]``."""

    section: ClassVar[tuple[str, ...]] = ("train", "tinker")
    api_key: SecretStr = Field(validation_alias="TINKER_API_KEY")
    spend_cap_usd: float = 60.0
    price_per_mtok: dict[TinkerModelId, TinkerPrice] = Field(
        default_factory=lambda: {
            TinkerModelId("Qwen/Qwen3-8B"): TinkerPrice(prefill=0.13, sample=0.40, train=0.40),
            TinkerModelId("Qwen/Qwen3.5-4B"): TinkerPrice(prefill=0.22, sample=0.67, train=0.67),
        }
    )


class LocalTrainSettings(SectionSettings):
    """The local mlx-lm trainer's knobs, bound to ``[train.local]``."""

    section: ClassVar[tuple[str, ...]] = ("train", "local")
    val_fraction: float = 0.1
    grad_checkpoint: bool = True


class ModalTrainSettings(SectionSettings):
    """The Modal GPU trainer's image pins, GPU class, and spend cap, bound to ``[train.modal]``.

    The pins are one co-installable set — TRL 0.21 requires ``transformers>=4.55``, so a run
    that pins them apart cannot even build its image — and ``datasets`` is pinned rather than
    floating because the training corpus crosses the wire as a pickled
    :class:`datasets.Dataset`, which an unpinned image would unpickle under whatever version
    the build happened to resolve.
    """

    section: ClassVar[tuple[str, ...]] = ("train", "modal")
    app_name: str = "athome-train"
    gpu: str = "H100"
    gpu_usd_per_hour: float = 3.95
    spend_cap_usd: float = 60.0
    hf_repo_prefix: str = "athome-train"
    trl_version: str = "0.21.0"
    peft_version: str = "0.17.0"
    torch_version: str = "2.6.0"
    transformers_version: str = "4.55.4"
    datasets_version: str = "5.0.0"
