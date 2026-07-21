from __future__ import annotations

import json
import platform
import shutil
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from athome.config import load
from athome.train import sidecar
from athome.train.data import normalize, render_mlx_jsonl
from athome.train.spec import Checkpoint, LocalTrainSettings, lora_keys
from athome.train.state import BackendMismatch, LocalState

if TYPE_CHECKING:
    from pathlib import Path

    from athome.progress import RunSink
    from athome.train.runstate import RunStateStore
    from athome.train.spec import BackendName, LoraSpec, Method, TrainSpec
    from athome.train.state import Resume, StateFidelity

DATA_DIR = "data"
ADAPTER_DIR = "adapter"
FUSED_DIR = "fused"
LORA_CONFIG = "lora.yaml"
OPTIMIZER = "adamw"
ADAPTER_WEIGHTS = "adapters.safetensors"


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


def resume_adapter_file(resume: Resume | None) -> Path | None:
    """The mlx-lm ``--resume-adapter-file`` a continuation loads, or None for a fresh run.

    Local restores weights only, from a prior run's durable adapter directory. A foreign handle
    (a Tinker or Modal state) cannot seed a local run and is refused loudly.
    """
    if resume is None:
        return None
    match resume.handle:
        case LocalState(adapter_dir=adapter_dir):
            return adapter_dir / ADAPTER_WEIGHTS
        case other:
            raise BackendMismatch(f"local cannot resume from a {type(other).__name__}: {other}")


def lora_command(
    spec: TrainSpec,
    *,
    data_dir: Path,
    adapter_dir: Path,
    config: Path,
    grad_checkpoint: bool,
    resume_adapter: Path | None = None,
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
        *(("--resume-adapter-file", str(resume_adapter)) if resume_adapter is not None else ()),
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
    state_fidelity: ClassVar[StateFidelity] = "weights"

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

    async def train(
        self,
        spec: TrainSpec,
        *,
        sink: RunSink,
        work_dir: Path,
        resume: Resume | None = None,
        store: RunStateStore | None = None,
    ) -> Checkpoint:
        """Train ``spec``'s LoRA with mlx-lm and fuse it into a standalone MLX model.

        ``work_dir`` holds everything the sidecar reads and writes: the
        ``{train,valid}.jsonl`` split, the ``lora.yaml`` config, the trained adapter,
        and the fused model that becomes the checkpoint's serve path.

        Local training is atomic — one sidecar run, no intermediate snapshots — so it keeps no
        resume ledger (``store`` is unused). A ``resume`` seeds a warm start from a prior run's
        durable adapter at ``weights`` fidelity: mlx-lm reloads the adapter and continues with a
        fresh optimizer, the honest loss of momentum a weights-only restore carries.

        Args:
            spec: The fine-tuning request; its method must be ``sft``.
            sink: Journals the data split, the sidecar argv, and the fused artifact.
            work_dir: The run's private working directory, minted by :func:`athome.train.run`.
            resume: A prior :class:`~athome.train.state.LocalState` to continue the adapter from, or
                None to train from base; a foreign handle is refused.
            store: Unused — local training persists no run state.

        Returns:
            A :class:`~athome.train.spec.Checkpoint` whose ``mlx_path`` is the fused
            standalone MLX model directory, whose ``train_cost_usd`` is 0.0, and whose ``state`` is
            the durable :class:`~athome.train.state.LocalState` adapter directory.

        Raises:
            UnsupportedLoraShape: The spec's :class:`~athome.train.spec.LoraSpec` asks
                for an adapter mlx-lm cannot express.
            BackendMismatch: ``resume`` carries a handle from another backend.
        """
        resume_adapter = resume_adapter_file(resume)
        adapter_dir = work_dir / ADAPTER_DIR
        config = lora_config(spec.lora, work_dir / LORA_CONFIG)
        examples = await normalize(spec.dataset, method="sft")
        data_dir = render_mlx_jsonl(
            examples, work_dir / DATA_DIR, val_fraction=self.settings.val_fraction, seed=spec.hyperparams.seed
        )
        await sink.append({"stage": "data", "examples": len(examples), "data_dir": str(data_dir)})
        command = lora_command(
            spec,
            data_dir=data_dir,
            adapter_dir=adapter_dir,
            config=config,
            grad_checkpoint=self.settings.grad_checkpoint,
            resume_adapter=resume_adapter,
        )
        await sink.append({"stage": "train", "command": list(command)})
        await sidecar.run_process(command)
        fused = await sidecar.fuse(adapter_dir, work_dir / FUSED_DIR, base=spec.base)
        await sink.append({"stage": "fused", "mlx_path": str(fused)})
        return Checkpoint(
            base=spec.base,
            backend=self.name,
            method="sft",
            step=spec.hyperparams.steps,
            mlx_path=fused,
            adapter_dir=adapter_dir,
            train_cost_usd=0.0,
            sampler_path=None,
            state=LocalState(adapter_dir=adapter_dir),
        )
