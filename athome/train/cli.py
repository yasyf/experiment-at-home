from __future__ import annotations

import importlib
import sys
from pathlib import Path

import click

from athome import detach, registry, train
from athome.bakeoff import load_spec
from athome.cli import coro, emit, json_option
from athome.config import load
from athome.errors import AthomeError
from athome.train.spec import TrainSettings, TrainSpec


class TrainSpecError(AthomeError):
    """A ``module:attr`` target does not resolve to a :class:`~athome.train.TrainSpec`."""


def load_train_spec(target: str) -> TrainSpec:
    module_name, _, attr = target.partition(":")
    obj = getattr(importlib.import_module(module_name), attr or "spec")
    spec = obj() if callable(obj) else obj
    if not isinstance(spec, TrainSpec):
        raise TrainSpecError(f"{target} resolved to {type(spec).__name__}, expected a TrainSpec")
    return spec


def run_name(name: str) -> str:
    return f"train-{name}"


def result_record(result: train.TrainResult) -> dict[str, object]:
    return {
        "version": result.version.version,
        "metric": result.metric,
        "promoted": result.promoted,
        "winner": result.leaderboard.winner,
        "passed_gate": result.leaderboard.passed_gate,
        "mlx_path": str(result.checkpoint.mlx_path),
        "backend": result.checkpoint.backend,
        "train_cost_usd": result.checkpoint.train_cost_usd,
    }


@click.group("train")
def cli() -> None:
    """Fine-tune a LoRA, bake it off against its baselines, and register the artifact."""


@cli.command("run")
@click.argument("spec")
@click.option("--evaluation", required=True, help="A module or module:attr path exporting the eval BakeoffSpec.")
@click.option("--detach", "detached", is_flag=True, help="Train in a detached run instead of in the foreground.")
@json_option
@coro
async def run_command(spec: str, evaluation: str, detached: bool, as_json: bool) -> None:
    """Train the TrainSpec exported by SPEC (a ``module`` or ``module:attr`` path)."""
    train_spec = load_train_spec(spec)
    if detached:
        launched = await detach.launch(
            [sys.executable, "-m", "athome", "train", "run", spec, "--evaluation", evaluation],
            name=run_name(train_spec.name),
        )
        emit({"run": launched.name, "pid": launched.pid, "log": str(launched.log_path)}, as_json=as_json)
        return
    emit(result_record(await train.run(train_spec, evaluation=load_spec(evaluation))), as_json=as_json)


@cli.command("status")
@click.argument("name")
@json_option
@coro
async def status_command(name: str, as_json: bool) -> None:
    """Report family NAME: its live detached run, its promoted version, and everything registered."""
    root = load(TrainSettings).registry_root
    promoted = await registry.current(name, root=root)
    emit(
        {
            "name": name,
            "running": detach.running(run_name(name)),
            "current": promoted.version if promoted is not None else None,
            "versions": [info.version for info in await registry.versions(name, root=root)],
        },
        as_json=as_json,
    )


@cli.command("register")
@click.argument("name")
@click.argument("mlx_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--promote", "promoted", is_flag=True, help="Flip the family's current symlink to the new version.")
@json_option
@coro
async def register_command(name: str, mlx_path: Path, promoted: bool, as_json: bool) -> None:
    """Register the fused MLX directory MLX_PATH under family NAME."""
    root = load(TrainSettings).registry_root
    info = await train.register(name, mlx_path.resolve(), {"source": "manual"}, root=root)
    if promoted:
        await registry.promote(name, info.version, root=root)
    emit({"version": info.version, "path": str(info.path), "promoted": promoted}, as_json=as_json)
