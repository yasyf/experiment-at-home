from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pytest

from athome import registry, serve, train
from athome.bakeoff import Arm, ArmResult, BakeoffSpec, Leaderboard
from athome.config import load
from athome.progress import load_journal
from athome.train.spec import BASE_MODELS, Checkpoint, Hyperparams, LocalJsonlRef, TrainSpec

if TYPE_CHECKING:
    from collections.abc import Iterator

    from openai import AsyncOpenAI

    from athome.progress import RunSink
    from athome.train.spec import BackendName, Method

FUSED = Path("/runs/watcher/fused")
BASELINE = Arm(name="base", base_url="http://127.0.0.1:8400/v1", model="mlx-community/Qwen3-8B-4bit")


async def task(client: AsyncOpenAI, item: object) -> dict[str, object]:
    raise AssertionError("the stubbed bake-off never runs the task")


def evaluation() -> BakeoffSpec:
    return BakeoffSpec(task=task, corpus=("a", "b"), arms=(BASELINE,), primary_metric="exact")


def spec(**overrides: object) -> TrainSpec:
    return TrainSpec(
        name="watcher",
        base=BASE_MODELS["qwen3-8b"],
        dataset=LocalJsonlRef(path=Path("corpus.jsonl")),
        hyperparams=Hyperparams(steps=10),
        **overrides,
    )


def checkpoint() -> Checkpoint:
    return Checkpoint(
        base=BASE_MODELS["qwen3-8b"],
        backend="tinker",
        method="sft",
        step=10,
        mlx_path=FUSED,
        adapter_dir=Path("/runs/watcher/adapter"),
        train_cost_usd=1.25,
    )


def leaderboard(*, winner: str, passed_gate: bool, metric: float = 0.9) -> Leaderboard:
    return Leaderboard(
        results=(
            ArmResult(arm=train.TRAINED_ARM, metrics={"exact": metric}, per_field_disagreement={}),
            ArmResult(arm="base", metrics={"exact": 0.5}, per_field_disagreement={}),
        ),
        winner=winner,
        passed_gate=passed_gate,
    )


@dataclass(slots=True)
class FakeBackend:
    name: ClassVar[BackendName] = "tinker"
    trained: list[TrainSpec]
    sinks: list[RunSink]

    @staticmethod
    def available() -> bool:
        return True

    @staticmethod
    def supports(method: Method) -> bool:
        return True

    @classmethod
    def from_settings(cls) -> FakeBackend:
        return cls([], [])

    async def train(self, spec: TrainSpec, *, sink: RunSink) -> Checkpoint:
        self.trained.append(spec)
        self.sinks.append(sink)
        await sink.append({"event": "step", "step": 10})
        return checkpoint()


class FakeServer:
    """A ManagedServer stand-in recording what got served on which port."""

    ensured: ClassVar[list[tuple[str, int]]] = []
    stopped: ClassVar[list[tuple[str, int]]] = []

    def __init__(self, recipe: str, *, model: str, port: int) -> None:
        self.recipe, self.model, self.port = recipe, model, port

    async def ensure(self) -> serve.ServerHandle:
        FakeServer.ensured.append((self.model, self.port))
        return serve.ServerHandle(
            recipe="rapid-mlx", port=self.port, pid=1, base_url=f"http://127.0.0.1:{self.port}/v1"
        )

    async def stop(self) -> None:
        FakeServer.stopped.append((self.model, self.port))

    def client(self) -> AsyncOpenAI:
        raise AssertionError("the stubbed bake-off never builds a client")


@pytest.fixture(autouse=True)
def train_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("ATHOME_TRAIN_REGISTRY_ROOT", str(tmp_path / "registry"))
    monkeypatch.setenv("ATHOME_TRAIN_WORK_ROOT", str(tmp_path / "runs"))
    monkeypatch.chdir(tmp_path)
    load.cache_clear()
    FakeServer.ensured.clear()
    FakeServer.stopped.clear()
    yield
    load.cache_clear()


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    chosen = FakeBackend.from_settings()
    monkeypatch.setattr(train, "select", lambda spec, settings: chosen)
    monkeypatch.setattr(serve, "ManagedServer", FakeServer)
    return chosen


def stub_bakeoff(monkeypatch: pytest.MonkeyPatch, board: Leaderboard) -> list[BakeoffSpec]:
    from athome import bakeoff

    ran: list[BakeoffSpec] = []

    async def fake_run(spec: BakeoffSpec) -> Leaderboard:
        ran.append(spec)
        return board

    monkeypatch.setattr(bakeoff, "run", fake_run)
    return ran


async def test_run_trains_evaluates_registers_and_promotes_the_winner(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend, tmp_path: Path
) -> None:
    ran = stub_bakeoff(monkeypatch, leaderboard(winner=train.TRAINED_ARM, passed_gate=True))

    result = await train.run(spec(), evaluation=evaluation())

    assert backend.trained == [spec()]
    assert result.checkpoint == checkpoint()
    assert result.metric == 0.9
    assert result.promoted is True

    root = tmp_path / "registry"
    promoted = await registry.current("watcher", root=root)
    assert promoted is not None and promoted.version == result.version.version
    assert promoted.metadata["mlx_path"] == str(FUSED)
    assert promoted.metadata["backend"] == "tinker"
    assert promoted.metadata["metric"] == 0.9
    assert json.loads((result.version.path / train.CHECKPOINT_FILE).read_text())["mlx_path"] == str(FUSED)
    assert len(ran) == 1


async def test_run_appends_the_trained_arm_to_the_evaluation_bakeoff(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend
) -> None:
    ran = stub_bakeoff(monkeypatch, leaderboard(winner=train.TRAINED_ARM, passed_gate=True))

    await train.run(spec(), evaluation=evaluation())

    arms = ran[0].arms
    assert [arm.name for arm in arms] == ["base", train.TRAINED_ARM]
    assert arms[0] is BASELINE  # the caller's baseline arm is untouched
    assert arms[-1].base_url == f"http://127.0.0.1:{train.EVAL_PORT}/v1"
    assert arms[-1].model == str(FUSED)
    assert arms[-1].client_factory is not None
    assert (ran[0].task, ran[0].corpus, ran[0].primary_metric) == (task, ("a", "b"), "exact")


async def test_run_serves_the_fused_artifact_on_the_eval_port_and_tears_it_down(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend
) -> None:
    stub_bakeoff(monkeypatch, leaderboard(winner="base", passed_gate=False))

    await train.run(spec(), evaluation=evaluation())

    assert FakeServer.ensured == [(str(FUSED), train.EVAL_PORT)]
    assert FakeServer.stopped == [(str(FUSED), train.EVAL_PORT)]


async def test_run_stops_the_server_when_the_bakeoff_raises(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend
) -> None:
    from athome import bakeoff

    async def boom(spec: BakeoffSpec) -> Leaderboard:
        raise RuntimeError("bake-off exploded")

    monkeypatch.setattr(bakeoff, "run", boom)

    with pytest.raises(RuntimeError, match="exploded"):
        await train.run(spec(), evaluation=evaluation())
    assert FakeServer.stopped == [(str(FUSED), train.EVAL_PORT)]


async def test_run_registers_but_does_not_promote_a_loser(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend, tmp_path: Path
) -> None:
    stub_bakeoff(monkeypatch, leaderboard(winner="base", passed_gate=False, metric=0.4))

    result = await train.run(spec(), evaluation=evaluation())

    assert result.promoted is False
    assert result.metric == 0.4
    root = tmp_path / "registry"
    assert [info.version for info in await registry.versions("watcher", root=root)] == [result.version.version]
    assert await registry.current("watcher", root=root) is None


async def test_run_does_not_promote_a_winner_that_misses_the_gate(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend, tmp_path: Path
) -> None:
    stub_bakeoff(monkeypatch, leaderboard(winner=train.TRAINED_ARM, passed_gate=False))

    result = await train.run(spec(), evaluation=evaluation())

    assert result.promoted is False
    assert await registry.current("watcher", root=tmp_path / "registry") is None


async def test_run_writes_the_research_metric_channel(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend, tmp_path: Path
) -> None:
    stub_bakeoff(monkeypatch, leaderboard(winner=train.TRAINED_ARM, passed_gate=True, metric=0.77))

    await train.run(spec(), evaluation=evaluation())

    assert json.loads((tmp_path / train.METRIC_FILE).read_text()) == {train.METRIC_KEY: 0.77}


async def test_run_journals_progress_to_the_work_root(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend, tmp_path: Path
) -> None:
    stub_bakeoff(monkeypatch, leaderboard(winner=train.TRAINED_ARM, passed_gate=True))

    result = await train.run(spec(), evaluation=evaluation())

    journal = tmp_path / "runs" / "watcher" / train.JOURNAL_FILE
    records = load_journal(journal)
    assert [record["event"] for record in records] == ["selected", "step", "trained", "registered"]
    assert records[0]["backend"] == "tinker"
    assert records[-1]["version"] == result.version.version
    assert records[-1]["promoted"] is True
    assert backend.sinks[0].path == journal


async def test_register_writes_a_pointer_entry_and_never_promotes(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    info = await train.register("watcher", FUSED, {"source": "manual"}, root=root)

    assert info.metadata["mlx_path"] == str(FUSED)
    assert info.metadata["source"] == "manual"
    assert json.loads((info.path / train.CHECKPOINT_FILE).read_text()) == {
        "mlx_path": str(FUSED),
        "source": "manual",
    }
    assert await registry.current("watcher", root=root) is None
