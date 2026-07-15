"""LoRA fine-tuning over tinker, local mlx-lm, or modal, converging on one servable MLX artifact."""

from __future__ import annotations

import json
import socket
from dataclasses import asdict, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import anyio

from athome import registry
from athome.config import load
from athome.registry import VersionInfo
from athome.train.backend import NoBackendAvailable, TrainBackend, backends, select
from athome.train.data import (
    DpoExample,
    Message,
    SftExample,
    TinkerPreference,
    TrainExample,
    normalize,
    render_mlx_jsonl,
    render_tinker_dpo,
    render_tinker_sft,
    render_trl,
)
from athome.train.spec import (
    BASE_MODELS,
    STD_MODULES,
    BackendName,
    BaseModelSpec,
    Checkpoint,
    DatasetSource,
    HfDatasetRef,
    Hyperparams,
    LocalJsonlRef,
    LocalTrainSettings,
    LoraSpec,
    Method,
    ModalTrainSettings,
    TinkerPrice,
    TinkerSettings,
    TrainResult,
    TrainSettings,
    TrainSpec,
    UnservableBase,
    UnsupportedLoraShape,
    lora_keys,
    spend_cap,
    std_lora_keys,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from athome.bakeoff import BakeoffSpec, Leaderboard

METRIC_FILE = ".athome-metric.json"
METRIC_KEY = "metric"
CHECKPOINT_FILE = "checkpoint.json"
MODEL_DIR = "model"
JOURNAL_FILE = "progress.jsonl"
TRAINED_ARM = "trained"
EVAL_RECIPE = "rapid-mlx"
RUN_ID_CHARS = 8


def free_port() -> int:
    """An unbound loopback port, so two concurrent runs never evaluate on one server identity."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def new_work_dir(spec: TrainSpec, settings: TrainSettings) -> Path:
    """Mint a fresh working directory for one run of ``spec`` under ``[train].work_root``.

    Every call returns a new directory: a run owns its data splits, adapters, fused weights,
    and journal outright, so a second run of the same family cannot overwrite the artifacts a
    first run trained — including the ones the registry has already registered or promoted.
    """
    return settings.work_root / spec.name / f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid4().hex[:RUN_ID_CHARS]}"


def model_path(version: VersionInfo) -> Path:
    """The registry's own copy of ``version``'s fused MLX model — the artifact to serve."""
    return version.path / MODEL_DIR


async def evaluate(checkpoint: Checkpoint, evaluation: BakeoffSpec, *, port: int) -> Leaderboard:
    """Serve ``checkpoint``'s fused artifact and bake it off against ``evaluation``'s arms.

    The fused MLX directory is served with rapid-mlx on ``port`` — an override that leaves the
    configured server alone — and appended to the bake-off as the ``trained`` arm, whose client
    is bound to that endpoint. The server is torn down before returning.

    Args:
        checkpoint: The artifact under test; its ``mlx_path`` is what gets served.
        evaluation: The bake-off supplying the task, corpus, and baseline arms.
        port: The port to serve on. It, with the model, names the managed server, so pass a
            :func:`free_port` per run: a port shared by two concurrent runs is one server
            identity two models contend for.

    Returns:
        The bake-off leaderboard, with the trained arm ranked among the baselines.
    """
    from athome import bakeoff, serve

    server = serve.ManagedServer(EVAL_RECIPE, model=str(checkpoint.mlx_path), port=port)
    handle = await server.ensure()
    arm = bakeoff.Arm(
        name=TRAINED_ARM,
        base_url=handle.base_url,
        model=str(checkpoint.mlx_path),
        client_factory=server.client,
    )
    try:
        return await bakeoff.run(replace(evaluation, arms=(*evaluation.arms, arm)))
    finally:
        with anyio.CancelScope(shield=True):
            await server.stop()


async def register(name: str, mlx_path: Path, metadata: Mapping[str, object], *, root: Path) -> VersionInfo:
    """Register the fused MLX directory ``mlx_path`` under the ``name`` family in ``root``.

    The registry takes ownership of the weights: ``mlx_path`` is copied into the version
    directory as :data:`MODEL_DIR` and frozen read-only, so the registered artifact is the
    registry's own and a later run of the family cannot mutate it. ``model_path(version)``
    resolves it; ``metadata["source_mlx_path"]`` records the run directory it was fused in.
    Registration never promotes.

    Args:
        name: The artifact family.
        mlx_path: The fused standalone MLX model directory to copy in.
        metadata: The version's provenance — backend, method, metric, cost.
        root: The registry root; ``athome.train`` uses ``[train].registry_root``.

    Returns:
        The registered version.
    """
    pointer = {"source_mlx_path": str(mlx_path), "model_dir": MODEL_DIR} | dict(metadata)
    payload = json.dumps(pointer, indent=2, sort_keys=True, default=str).encode()
    return await registry.register(name, {CHECKPOINT_FILE: payload, MODEL_DIR: mlx_path}, pointer, root=root)


async def write_metric(metric: float) -> None:
    """Write the run's scalar to ``.athome-metric.json`` in the working directory."""
    await anyio.Path(METRIC_FILE).write_text(json.dumps({METRIC_KEY: metric}) + "\n")


async def run(spec: TrainSpec, *, evaluation: BakeoffSpec) -> TrainResult:
    """Train ``spec``, score the artifact against ``evaluation``, and register what came out.

    The run gets a fresh :func:`new_work_dir` and a :func:`free_port` of its own, so concurrent
    runs — of one family or several — share neither artifacts nor a server. The selected backend
    trains the LoRA there and converges on a fused standalone MLX directory. That directory is
    served locally and bakes off against ``evaluation``'s arms as the ``trained`` arm; the run is
    registered under ``spec.name`` — weights and all, copied into the registry — and promoted to
    the family's ``current`` only when the trained arm wins the bake-off *and* clears its
    statistical gate.

    Progress journals to ``<work_dir>/progress.jsonl``, and the metric lands in
    ``.athome-metric.json`` in the working directory — the structured channel
    :mod:`athome.research` reads — so ``athome train run`` drops straight into a research loop
    as a metric command.

    Args:
        spec: The fine-tuning request; it also picks the backend.
        evaluation: The bake-off the trained artifact is scored against. Its ``arms[0]`` is the
            baseline the gate measures lift over, and its ``primary_metric`` is the scalar
            reported as :attr:`~athome.train.TrainResult.metric`.

    Returns:
        The checkpoint, its primary metric, the leaderboard it came from, the registered
        version, and whether that version was promoted.

    Raises:
        NoBackendAvailable: No backend can train the spec's method.
    """
    from athome.progress import RunSink

    settings = load(TrainSettings)
    backend = select(spec, settings)
    work_dir = new_work_dir(spec, settings)
    sink = RunSink.open(work_dir / JOURNAL_FILE)
    await sink.append({"event": "selected", "backend": backend.name, "method": spec.method, "work_dir": str(work_dir)})

    checkpoint = await backend.train(spec, sink=sink, work_dir=work_dir)
    await sink.append({"event": "trained", "mlx_path": str(checkpoint.mlx_path), "usd": checkpoint.train_cost_usd})

    leaderboard = await evaluate(checkpoint, evaluation, port=free_port())
    metric = next(result for result in leaderboard.results if result.arm == TRAINED_ARM).metrics[
        evaluation.primary_metric
    ]
    promoted = leaderboard.winner == TRAINED_ARM and leaderboard.passed_gate
    await write_metric(metric)

    version = await register(
        spec.name,
        checkpoint.mlx_path,
        {
            "backend": checkpoint.backend,
            "method": checkpoint.method,
            "base": checkpoint.base.mlx,
            "step": checkpoint.step,
            "adapter_dir": checkpoint.adapter_dir,
            "train_cost_usd": checkpoint.train_cost_usd,
            METRIC_KEY: metric,
            "primary_metric": evaluation.primary_metric,
            "leaderboard": asdict(leaderboard),
        },
        root=settings.registry_root,
    )
    if promoted:
        await registry.promote(spec.name, version.version, root=settings.registry_root)
    await sink.append({"event": "registered", "version": version.version, METRIC_KEY: metric, "promoted": promoted})
    return TrainResult(
        checkpoint=checkpoint,
        metric=metric,
        leaderboard=leaderboard,
        version=version,
        promoted=promoted,
    )
