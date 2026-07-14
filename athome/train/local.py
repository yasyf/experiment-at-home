from __future__ import annotations

import json
import platform
import shutil
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from athome.config import load
from athome.errors import AthomeError
from athome.train import sidecar
from athome.train.data import normalize, render_mlx_jsonl
from athome.train.spec import Checkpoint, LocalTrainSettings, TrainSettings

if TYPE_CHECKING:
    from pathlib import Path

    from athome.progress import RunSink
    from athome.train.spec import BackendName, LoraSpec, Method, TrainSpec

DATA_DIR = "data"
ADAPTER_DIR = "adapter"
FUSED_DIR = "fused"
LORA_CONFIG = "lora.yaml"
OPTIMIZER = "adamw"
ATTN_PREFIX = "self_attn."
MLP_PREFIX = "mlp."


class UnsupportedLoraShape(AthomeError):
    """Raised when a ``LoraSpec`` asks for an adapter mlx-lm cannot express."""


def lora_keys(lora: LoraSpec) -> tuple[str, ...]:
    """The modules mlx-lm wraps: ``target_modules`` filtered by the spec's LoRA toggles.

    Raises:
        UnsupportedLoraShape: ``train_unembed`` is set, or both ``train_attn`` and
            ``train_mlp`` are off. mlx-lm reaches the unembedding through a
            base-dependent module path — ``lm_head``, which a base that ties its
            embeddings does not have at all — and
            :class:`~athome.train.spec.BaseModelSpec` does not carry it, so a guessed
            key would match nothing and train the unembedding silently not at all.
    """
    if lora.train_unembed:
        raise UnsupportedLoraShape(
            "mlx-lm cannot LoRA the unembedding: its module path is base-dependent "
            "(`lm_head`, absent on a base that ties its embeddings) and BaseModelSpec does not carry it. "
            "Train a train_unembed=True spec on tinker or modal."
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


def lora_config(lora: LoraSpec, path: Path) -> Path:
    """Write the ``--config`` file carrying the LoRA shape and return it.

    ``rank``, ``scale``, ``dropout``, and ``keys`` have no mlx-lm CLI flags — the
    ``lora_parameters`` block is only reachable through this file. ``scale`` is
    ``alpha / rank``, matching the adapter config
    :func:`~athome.train.sidecar.run_convert` writes for the other backends.
    """
    # mlx-lm reads --config with yaml.SafeLoader, and JSON is a subset of YAML.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "lora_parameters": {
                    "rank": lora.rank,
                    "scale": lora.alpha / lora.rank,
                    "dropout": lora.dropout,
                    "keys": list(lora_keys(lora)),
                }
            },
            indent=2,
        )
        + "\n"
    )
    return path


def lora_command(
    spec: TrainSpec, *, data_dir: Path, adapter_dir: Path, config: Path, grad_checkpoint: bool
) -> tuple[str, ...]:
    """The ``mlx_lm.lora`` argv that trains ``spec``'s adapter in the pinned sidecar."""
    return sidecar.mlx_lm_command(
        "lora",
        "--train",
        "--model",
        spec.base.mlx,
        "--data",
        str(data_dir),
        "--adapter-path",
        str(adapter_dir),
        "--config",
        str(config),
        "--fine-tune-type",
        "lora",
        "--optimizer",
        OPTIMIZER,
        "--mask-prompt",
        "--num-layers",
        str(spec.base.num_layers),
        "--iters",
        str(spec.hyperparams.steps),
        "--batch-size",
        str(spec.hyperparams.batch_size),
        "--learning-rate",
        str(spec.hyperparams.learning_rate),
        "--max-seq-length",
        str(spec.hyperparams.max_seq_len),
        "--seed",
        str(spec.hyperparams.seed),
        *(("--grad-checkpoint",) if grad_checkpoint else ()),
    )


@dataclass(frozen=True, slots=True)
class LocalBackend:
    """Trains a LoRA on this Mac with mlx-lm, out of process in the pinned uvx sidecar.

    mlx is native and GIL-bound, so nothing here imports it: training shells out to
    ``mlx_lm.lora`` and the fuse to ``mlx_lm.fuse``, both in the sidecar environment
    :mod:`athome.train.sidecar` pins. Training is free, so ``train_cost_usd`` is 0.0
    and no spend cap applies.

    Example:
        >>> checkpoint = await LocalBackend.from_settings().train(spec, sink=sink)
    """

    settings: LocalTrainSettings
    name: ClassVar[BackendName] = "local"

    @staticmethod
    def available() -> bool:
        """Whether this is an arm64 Mac with ``uvx`` on PATH to launch the mlx-lm sidecar."""
        return sys.platform == "darwin" and platform.machine() == "arm64" and shutil.which("uvx") is not None

    @staticmethod
    def supports(method: Method) -> bool:
        """Whether mlx-lm trains ``method``: ``sft`` only, as it has no preference trainer."""
        return method == "sft"

    @classmethod
    def from_settings(cls) -> LocalBackend:
        """Construct the backend from the ``[train.local]`` config section."""
        return cls(load(LocalTrainSettings))

    async def train(self, spec: TrainSpec, *, sink: RunSink) -> Checkpoint:
        """Train ``spec``'s LoRA with mlx-lm and fuse it into a standalone MLX model.

        The run's work directory holds everything the sidecar reads and writes: the
        ``{train,valid}.jsonl`` split, the ``lora.yaml`` config, the trained adapter,
        and the fused model that becomes the checkpoint's serve path.

        Args:
            spec: The fine-tuning request; its method must be ``sft``.
            sink: Journals the data split, the sidecar argv, and the fused artifact.

        Returns:
            A :class:`~athome.train.spec.Checkpoint` whose ``mlx_path`` is the fused
            standalone MLX model directory and whose ``train_cost_usd`` is 0.0.

        Raises:
            UnsupportedLoraShape: The spec's :class:`~athome.train.spec.LoraSpec` asks
                for an adapter mlx-lm cannot express.
        """
        run_dir = load(TrainSettings).work_root / spec.name
        adapter_dir = run_dir / ADAPTER_DIR
        config = lora_config(spec.lora, run_dir / LORA_CONFIG)
        examples = await normalize(spec.dataset, method="sft")
        data_dir = render_mlx_jsonl(
            examples, run_dir / DATA_DIR, val_fraction=self.settings.val_fraction, seed=spec.hyperparams.seed
        )
        await sink.append({"stage": "data", "examples": len(examples), "data_dir": str(data_dir)})
        command = lora_command(
            spec,
            data_dir=data_dir,
            adapter_dir=adapter_dir,
            config=config,
            grad_checkpoint=self.settings.grad_checkpoint,
        )
        await sink.append({"stage": "train", "command": list(command)})
        await sidecar.run_process(command)
        fused = await sidecar.fuse(adapter_dir, run_dir / FUSED_DIR, base=spec.base)
        await sink.append({"stage": "fused", "mlx_path": str(fused)})
        return Checkpoint(
            base=spec.base,
            backend=self.name,
            method="sft",
            step=spec.hyperparams.steps,
            mlx_path=fused,
            adapter_dir=adapter_dir,
            train_cost_usd=0.0,
        )
