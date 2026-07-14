from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import pytest
from click.testing import CliRunner

from athome import registry, train
from athome.cli import main
from athome.config import load
from athome.detach import DetachedRun
from athome.train import cli as train_cli
from athome.train.spec import BASE_MODELS, Hyperparams, LocalJsonlRef, TrainSpec
from tests.test_train_run import WEIGHTS, checkpoint, evaluation, fuse, leaderboard

if TYPE_CHECKING:
    from collections.abc import Iterator

    from athome.bakeoff import BakeoffSpec

TARGET = "tests.test_train_cli:train_spec"

train_spec = TrainSpec(
    name="watcher",
    base=BASE_MODELS["qwen3-8b"],
    dataset=LocalJsonlRef(path=Path("corpus.jsonl")),
    hyperparams=Hyperparams(steps=10),
)


@pytest.fixture(autouse=True)
def train_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    root = tmp_path / "registry"
    monkeypatch.setenv("ATHOME_TRAIN_REGISTRY_ROOT", str(root))
    load.cache_clear()
    yield root
    load.cache_clear()


def invoke(*args: str) -> dict[str, object]:
    result = CliRunner().invoke(main, ["train", *args, "--json"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def test_the_train_group_is_registered_on_the_root_cli() -> None:
    result = CliRunner().invoke(main, ["train", "--help"])
    assert result.exit_code == 0
    for command in ("run", "status", "register"):
        assert command in result.output


def test_load_train_spec_rejects_a_target_that_is_not_a_train_spec() -> None:
    with pytest.raises(train_cli.TrainSpecError, match="expected a TrainSpec"):
        train_cli.load_train_spec("tests.test_train_cli:TARGET")


def test_run_trains_in_the_foreground_and_reports_the_result(
    monkeypatch: pytest.MonkeyPatch, train_root: Path, tmp_path: Path
) -> None:
    fused = fuse(tmp_path / "scratch", b"weights")
    version = anyio.run(lambda: train.register("watcher", fused, {}, root=train_root))
    trained: list[TrainSpec] = []

    async def fake_run(spec: TrainSpec, *, evaluation: BakeoffSpec) -> train.TrainResult:
        trained.append(spec)
        return train.TrainResult(
            checkpoint=checkpoint(),
            metric=0.9,
            leaderboard=leaderboard(winner=train.TRAINED_ARM, passed_gate=True),
            version=version,
            promoted=True,
        )

    monkeypatch.setattr(train, "run", fake_run)
    monkeypatch.setattr(train_cli, "load_spec", lambda target: evaluation())

    record = invoke("run", TARGET, "--evaluation", "tests.test_train_run:evaluation")

    assert trained == [train_spec]
    assert record["version"] == version.version
    assert record["metric"] == 0.9
    assert record["promoted"] is True
    assert record["mlx_path"] == "/runs/watcher/fused"
    assert record["backend"] == "tinker"


def test_run_detached_launches_the_same_invocation_under_a_run_name(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: dict[str, object] = {}

    async def fake_launch(command: list[str], *, name: str) -> DetachedRun:
        launched["command"], launched["name"] = list(command), name
        return DetachedRun(name=name, pid=4321, log_path=Path("/tmp/train.log"))

    async def fail_run(spec: TrainSpec, *, evaluation: BakeoffSpec) -> train.TrainResult:
        raise AssertionError("a detached run must not train in the foreground")

    monkeypatch.setattr(train_cli.detach, "launch", fake_launch)
    monkeypatch.setattr(train, "run", fail_run)

    record = invoke("run", TARGET, "--evaluation", "pkg:eval", "--detach")

    assert record == {"run": "train-watcher", "pid": 4321, "log": "/tmp/train.log"}
    assert launched["command"] == [
        sys.executable,
        "-m",
        "athome",
        "train",
        "run",
        TARGET,
        "--evaluation",
        "pkg:eval",
    ]
    assert launched["name"] == "train-watcher"


def test_status_reports_the_registry_and_the_live_run(
    monkeypatch: pytest.MonkeyPatch, train_root: Path, tmp_path: Path
) -> None:
    async def seed() -> tuple[str, str]:
        first = await train.register("watcher", fuse(tmp_path / "a", b"a"), {}, root=train_root)
        second = await train.register("watcher", fuse(tmp_path / "b", b"b"), {}, root=train_root)
        await registry.promote("watcher", second.version, root=train_root)
        return first.version, second.version

    first, second = anyio.run(seed)
    monkeypatch.setattr(train_cli.detach, "running", lambda name: 99 if name == "train-watcher" else None)

    record = invoke("status", "watcher")

    assert record["running"] == 99
    assert record["current"] == second
    assert record["versions"] == [first, second]


def test_status_of_an_unregistered_family_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_cli.detach, "running", lambda name: None)
    assert invoke("status", "nobody") == {"name": "nobody", "running": None, "current": None, "versions": []}


def test_register_copies_the_model_in_and_promotes_on_demand(tmp_path: Path, train_root: Path) -> None:
    fused = fuse(tmp_path / "scratch", b"weights-manual")

    record = invoke("register", "watcher", str(fused), "--promote")

    promoted = anyio.run(lambda: registry.current("watcher", root=train_root))
    assert record["promoted"] is True
    assert promoted is not None and promoted.version == record["version"]
    assert promoted.metadata["source_mlx_path"] == str(fused.resolve())
    assert promoted.metadata["source"] == "manual"
    assert (train.model_path(promoted) / WEIGHTS).read_bytes() == b"weights-manual"


def test_register_without_promote_leaves_current_unset(tmp_path: Path, train_root: Path) -> None:
    record = invoke("register", "watcher", str(fuse(tmp_path / "scratch", b"weights-manual")))

    assert record["promoted"] is False
    assert anyio.run(lambda: registry.current("watcher", root=train_root)) is None
