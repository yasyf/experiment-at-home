from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, NewType

from pydantic import Field, SecretStr

from athome.config import SectionSettings
from athome.errors import AthomeError

if TYPE_CHECKING:
    from athome.bakeoff import Leaderboard
    from athome.registry import VersionInfo
    from athome.train.data import SftExample

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
INTERMEDIATE_TTL_S = 604_800


class UnservableBase(AthomeError):
    """Raised when the spec's base model cannot produce a servable fused MLX artifact."""


def require_servable(base: BaseModelSpec, *, kind: str) -> None:
    """Refuse a base with no mlx-lm counterpart, in the one place every fuse path shares.

    Every backend converges on a fused mlx-lm model, so a base that cannot load into mlx-lm
    is unservable everywhere; the Tinker and Modal fuse paths funnel their identical precheck
    through here instead of each copy-pasting it.

    Args:
        base: The base model the run would fuse an adapter into.
        kind: The backend named in the refusal — ``"Tinker"`` or ``"Modal"``.

    Raises:
        UnservableBase: ``base.serves_locally`` is False, so the LoRA has no mlx-lm
            counterpart to fuse into.
    """
    if not base.serves_locally:
        raise UnservableBase(f"{base.mlx} has no mlx-lm LoRA counterpart, so a {kind} adapter cannot be fused into it")


class UnsupportedLoraShape(AthomeError):
    """Raised when a :class:`LoraSpec` asks for an adapter the MLX fuse path cannot express."""


class InsufficientData(AthomeError):
    """Raised when the surviving training pool holds fewer examples than one batch.

    Carries the surviving ``count`` so the caller can report how far short the pool fell.
    It fires before the spend guard reserves anything and before any training client is
    constructed, so an under-filled run aborts having spent nothing.
    """

    def __init__(self, count: int, batch_size: int) -> None:
        self.count = count
        super().__init__(f"{count} surviving examples cannot fill a batch of {batch_size}")


class OverlongEvalRows(AthomeError):
    """Raised when any eval row exceeds ``max_seq_len``; the whole frozen set fails up-front.

    Carries how many rows overran, checked before any billable call so an ill-sized eval
    set never rides a snapshot's prefill.
    """

    def __init__(self, count: int, max_seq_len: int) -> None:
        self.count = count
        super().__init__(f"{count} eval rows exceed max_seq_len={max_seq_len}")


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
    "qwen3.5-9b": BaseModelSpec(
        mlx=MlxModelId("mlx-community/Qwen3.5-9B-4bit"),
        hf=HfRepoId("Qwen/Qwen3.5-9B"),
        hf_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        mlx_revision="8b2b98c00a6b4d291155e4890773ca8f769aee53",
        tinker=TinkerModelId("Qwen/Qwen3.5-9B"),
        num_layers=32,
        serves_locally=False,
    ),
    "qwen3.6-35b-a3b": BaseModelSpec(
        mlx=MlxModelId("mlx-community/Qwen3.6-35B-A3B-4bit"),
        hf=HfRepoId("Qwen/Qwen3.6-35B-A3B"),
        hf_revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        mlx_revision="38740b847e4cb78f352aba30aa41c76e08e6eb46",
        tinker=TinkerModelId("Qwen/Qwen3.6-35B-A3B"),
        num_layers=40,
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


@dataclass(frozen=True, slots=True)
class Rows:
    """An in-memory SFT corpus: examples a caller already holds, with no jsonl detour.

    Attributes:
        examples: The SFT examples to train on directly.
    """

    examples: tuple[SftExample, ...]


type DatasetSource = HfDatasetRef | LocalJsonlRef | Rows


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
class EvalRow:
    """A pre-tokenized weighted scoring row, mapping one-to-one onto a Tinker ``Datum``.

    The row's ``tokens`` are the full sequence; the weight-carrying positions are the ones a
    checkpoint's forward pass scores. Unlike a message-level example, this can weight a single
    sentinel-token position, which is what a next-token discriminator scores.

    Attributes:
        tokens: The full token sequence, prompt and completion.
        weights: One weight per token; a nonzero weight marks a scored position.
    """

    tokens: tuple[int, ...]
    weights: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CheckpointPolicy:
    """Which fractions of a run to snapshot as intermediate checkpoints, and for how long.

    The final step is always snapshotted separately and kept forever; these fractions place the
    intermediate saves before it. ``at`` holds fractions in ``(0, 1]``, strictly increasing.

    Example:
        >>> CheckpointPolicy(at=(0.25, 0.5, 0.75)).steps_for(12)
        (3, 6, 9)
    """

    at: tuple[float, ...] = ()
    ttl_seconds: int = INTERMEDIATE_TTL_S

    def __post_init__(self) -> None:
        if any(not (0.0 < fraction <= 1.0) for fraction in self.at):
            raise ValueError(f"checkpoint fractions must lie in (0, 1]: {self.at}")
        if list(self.at) != sorted(set(self.at)):
            raise ValueError(f"checkpoint fractions must be strictly increasing and unique: {self.at}")

    def steps_for(self, total: int) -> tuple[int, ...]:
        """The intermediate step numbers for a run of ``total`` steps, ascending and unique.

        Each fraction rounds up to at least step 1; the final step is excluded because it is
        always snapshotted on its own. Fractions that collapse onto the same step or onto the
        final step drop out.
        """
        return tuple(sorted({max(1, ceil(fraction * total)) for fraction in self.at} - {total}))


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
class Adapter:
    """A materialized non-fused MLX adapter that can be served or fused later.

    Attributes:
        step: The training step whose weights this adapter carries.
        adapter_dir: The mlx-lm adapter directory.
        train_cost_usd: What the training run spent to produce the saved weights.
        sampler_path: The opaque ``tinker://`` sampler checkpoint the adapter was materialized from.
    """

    step: int
    adapter_dir: Path
    train_cost_usd: float
    sampler_path: str


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """The servable artifact one backend produced: a standalone 4-bit MLX model directory.

    ``mlx_path`` is the only serve path — ``rapid-mlx serve`` takes a single model
    and has no adapter flag, so every backend fuses its LoRA into the base.
    ``adapter_dir`` is the intermediate adapter kept for provenance (tinker and
    modal train a PEFT adapter first); consumers with adapter support can also
    serve it directly.

    Attributes:
        base: The base model the LoRA was trained over.
        backend: The backend that produced it.
        method: The method it was trained with.
        step: The step count the weights were saved at.
        mlx_path: The fused standalone 4-bit MLX model directory.
        adapter_dir: The mlx-lm adapter directory, when one was materialized.
        train_cost_usd: What the run spent; 0.0 on the unmetered local backend.
        sampler_path: The opaque ``tinker://`` sampler checkpoint the fused adapter came from, or
            None for a backend that produces no hosted sampler checkpoint (local, modal).
    """

    base: BaseModelSpec
    backend: BackendName
    method: Method
    step: int
    mlx_path: Path
    adapter_dir: Path | None
    train_cost_usd: float
    sampler_path: str | None


@dataclass(frozen=True, slots=True)
class ScoredSequence:
    """One eval row's score against a checkpoint's weights.

    Attributes:
        logprob: The weighted sum of per-token logprobs over the scored positions.
        weight: The exact weight mass, so a caller can normalize ``logprob`` per scored weight.
    """

    logprob: float
    weight: float


@dataclass(frozen=True, slots=True)
class SampledSequence:
    """One free-form completion sampled from a checkpoint, with its token counts and billed cost.

    Attributes:
        text: The decoded completion text.
        prompt_tokens: The prompt's prefilled token count.
        sampled_tokens: The number of tokens generated.
        usd: What this sequence's prefill and generated tokens billed.
    """

    text: str
    prompt_tokens: int
    sampled_tokens: int
    usd: float


@dataclass(frozen=True, slots=True)
class SavedCheckpoint:
    """A Tinker sampler checkpoint saved mid-run, with its eval scores against those weights.

    Attributes:
        step: The training step whose weights this snapshot captured.
        path: The opaque ``tinker://`` address the weights were saved to.
        final: True for the run's last checkpoint, which is kept forever.
        scores: The eval rows scored against these weights, order-preserving, or None when
            the run carried no eval rows.
    """

    step: int
    path: str
    final: bool
    scores: tuple[ScoredSequence, ...] | None


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One training step's telemetry.

    Attributes:
        step: The one-based step number.
        tokens: The step's trained token count.
        usd: What this step alone cost, not the running total.
        metrics: The loss and any method-specific metrics (SFT ``loss``; DPO
            ``loss``/``margin``/``accuracy``).
    """

    step: int
    tokens: int
    usd: float
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class TrainReport:
    """Everything one ``fit`` produced: its per-step records and its saved checkpoints.

    Attributes:
        method: The method trained.
        steps: The per-step records, ascending.
        checkpoints: The saved checkpoints, ascending by step; the last is the final.
        dropped: How many examples were dropped for exceeding ``max_seq_len``.
        wall_s: Wall-clock seconds spent executing the schedule.
        train_cost_usd: The run's total metered spend.
    """

    method: Method
    steps: tuple[StepRecord, ...]
    checkpoints: tuple[SavedCheckpoint, ...]
    dropped: int
    wall_s: float
    train_cost_usd: float

    @property
    def final(self) -> SavedCheckpoint:
        """The run's final checkpoint."""
        return self.checkpoints[-1]


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
    baseline_root: Path = Path("~/.athome/train/baselines.db")
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
            TinkerModelId("Qwen/Qwen3.5-9B"): TinkerPrice(prefill=0.44, sample=1.33, train=1.33),
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
