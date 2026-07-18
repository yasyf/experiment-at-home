from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pytest

from athome import registry, serve, train
from athome.bakeoff import Arm, ArmResult, BakeoffSpec, Leaderboard
from athome.config import load
from athome.progress import load_journal
from athome.research.baseline import BaselineStore
from athome.research.spec import Comparability
from athome.train.spec import BASE_MODELS, Checkpoint, Hyperparams, LocalJsonlRef, TrainSettings, TrainSpec

if TYPE_CHECKING:
    from collections.abc import Iterator

    from openai import AsyncOpenAI

    from athome.progress import RunSink
    from athome.registry import VersionInfo
    from athome.train.spec import BackendName, Method

FUSED = Path("/runs/watcher/fused")
WEIGHTS = "model.safetensors"
BASELINE = Arm(name="base", base_url="http://127.0.0.1:8400/v1", model="mlx-community/Qwen3-8B-4bit")


async def task(client: AsyncOpenAI, item: object) -> dict[str, object]:
    raise AssertionError("the stubbed bake-off never runs the task")


async def other_task(client: AsyncOpenAI, item: object) -> dict[str, object]:
    raise AssertionError("the stubbed bake-off never runs the task")


def evaluation() -> BakeoffSpec:
    return BakeoffSpec(task=task, corpus=("a", "b"), arms=(BASELINE,), primary_metric="exact")


def comparability(evaluation: BakeoffSpec) -> Comparability:
    return train._comparability(evaluation)


def test_comparability_hash_is_stable_for_equivalent_evaluations() -> None:
    assert comparability(evaluation()).config_hash == comparability(evaluation()).config_hash


@pytest.mark.parametrize(
    "changed",
    (
        pytest.param(
            BakeoffSpec(task=other_task, corpus=("a", "b"), arms=(BASELINE,), primary_metric="exact"),
            id="task",
        ),
        pytest.param(
            BakeoffSpec(
                task=task,
                corpus=("a", "b"),
                arms=(Arm(name="other", base_url="http://localhost/v1", model="other"),),
                primary_metric="exact",
            ),
            id="arms",
        ),
        pytest.param(
            BakeoffSpec(task=task, corpus=("a", "b"), arms=(BASELINE,), primary_metric="accuracy"),
            id="primary-metric",
        ),
        pytest.param(
            BakeoffSpec(task=task, corpus=("a", "b"), arms=(BASELINE,), primary_metric="exact", tiebreak="latency"),
            id="tiebreak",
        ),
    ),
)
def test_comparability_hash_changes_with_the_evaluation_configuration(changed: BakeoffSpec) -> None:
    assert comparability(changed).config_hash != comparability(evaluation()).config_hash


def spec(**overrides: object) -> TrainSpec:
    return TrainSpec(
        name="watcher",
        base=BASE_MODELS["qwen3-8b"],
        dataset=LocalJsonlRef(path=Path("corpus.jsonl")),
        hyperparams=Hyperparams(steps=10),
        **overrides,
    )


def checkpoint(*, mlx_path: Path = FUSED, adapter_dir: Path = Path("/runs/watcher/adapter")) -> Checkpoint:
    return Checkpoint(
        base=BASE_MODELS["qwen3-8b"],
        backend="tinker",
        method="sft",
        step=10,
        mlx_path=mlx_path,
        adapter_dir=adapter_dir,
        train_cost_usd=1.25,
        sampler_path="tinker://run/watcher-sampler",
    )


def fuse(work_dir: Path, weights: bytes) -> Path:
    """Write a fused MLX directory under ``work_dir``, the way a real backend would."""
    fused = work_dir / "fused"
    fused.mkdir(parents=True)
    (fused / WEIGHTS).write_bytes(weights)
    return fused


def weights_of(version: VersionInfo) -> bytes:
    return (train.model_path(version) / WEIGHTS).read_bytes()


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
    """Fuses distinct weights per run into whatever ``work_dir`` it is handed."""

    name: ClassVar[BackendName] = "tinker"
    trained: list[TrainSpec] = field(default_factory=list)
    sinks: list[RunSink] = field(default_factory=list)
    work_dirs: list[Path] = field(default_factory=list)
    checkpoints: list[Checkpoint] = field(default_factory=list)

    @staticmethod
    def available() -> bool:
        return True

    @staticmethod
    def supports(method: Method) -> bool:
        return True

    @classmethod
    def from_settings(cls) -> FakeBackend:
        return cls()

    async def train(self, spec: TrainSpec, *, sink: RunSink, work_dir: Path) -> Checkpoint:
        self.trained.append(spec)
        self.sinks.append(sink)
        self.work_dirs.append(work_dir)
        await sink.append({"event": "step", "step": 10})
        fused = fuse(work_dir, f"weights-{len(self.trained)}".encode())
        self.checkpoints.append(trained := checkpoint(mlx_path=fused, adapter_dir=work_dir / "adapter"))
        return trained


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
    monkeypatch.setenv("ATHOME_TRAIN_BASELINE_ROOT", str(tmp_path / "baselines.db"))
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

    async def passes_preflight(
        spec: TrainSpec, *, evaluation: BakeoffSpec, settings: TrainSettings
    ) -> train.PreflightReport:
        return train.PreflightReport(checks=())

    monkeypatch.setattr(train, "preflight", passes_preflight)
    monkeypatch.setattr(train, "select", lambda spec, settings: chosen)
    monkeypatch.setattr(serve, "ManagedServer", FakeServer)
    return chosen


def stub_bakeoff(monkeypatch: pytest.MonkeyPatch, *boards: Leaderboard) -> list[BakeoffSpec]:
    """Stub the bake-off with one board per run; the final board answers every run after it."""
    from athome import bakeoff

    ran: list[BakeoffSpec] = []

    async def fake_run(spec: BakeoffSpec) -> Leaderboard:
        ran.append(spec)
        return boards[min(len(ran), len(boards)) - 1]

    monkeypatch.setattr(bakeoff, "run", fake_run)
    return ran


async def test_run_trains_evaluates_registers_and_promotes_the_winner(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend, tmp_path: Path
) -> None:
    ran = stub_bakeoff(monkeypatch, leaderboard(winner=train.TRAINED_ARM, passed_gate=True))

    result = await train.run(spec(), evaluation=evaluation())

    assert backend.trained == [spec()]
    assert result.checkpoint == backend.checkpoints[0]
    assert result.metric == 0.9
    assert result.promoted is True

    promoted = await registry.current("watcher", root=tmp_path / "registry")
    assert promoted is not None and promoted.version == result.version.version
    assert promoted.metadata["source_mlx_path"] == str(backend.checkpoints[0].mlx_path)
    assert promoted.metadata["backend"] == "tinker"
    assert promoted.metadata["sampler_path"] == "tinker://run/watcher-sampler"
    assert promoted.metadata["metric"] == 0.9
    assert json.loads((result.version.path / train.CHECKPOINT_FILE).read_text())["source_mlx_path"] == str(
        backend.checkpoints[0].mlx_path
    )
    assert len(ran) == 1


async def test_run_registers_frozen_baseline_metadata(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend, tmp_path: Path
) -> None:
    bakeoff = evaluation()
    async with BaselineStore.open(tmp_path / "baselines.db") as baselines:
        await baselines.put(comparability(bakeoff), spec().name, 0.75)
    stub_bakeoff(monkeypatch, leaderboard(winner=train.TRAINED_ARM, passed_gate=True, metric=0.9))

    result = await train.run(spec(), evaluation=bakeoff)

    assert result.version.metadata["baseline_metric"] == 0.75
    assert result.version.metadata["uplift"] == pytest.approx(0.2)


async def test_run_omits_frozen_baseline_metadata_when_unseeded(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend
) -> None:
    stub_bakeoff(monkeypatch, leaderboard(winner=train.TRAINED_ARM, passed_gate=True, metric=0.9))

    result = await train.run(spec(), evaluation=evaluation())

    assert "baseline_metric" not in result.version.metadata
    assert "uplift" not in result.version.metadata


async def test_run_preflights_with_loaded_settings_before_selecting_and_training(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend
) -> None:
    events: list[str] = []
    request = spec()
    bakeoff = evaluation()
    expected_settings = load(TrainSettings)
    original_train = FakeBackend.train

    async def preflight(spec: TrainSpec, *, evaluation: BakeoffSpec, settings: TrainSettings) -> train.PreflightReport:
        assert spec is request
        assert evaluation is bakeoff
        assert settings is expected_settings
        events.append("preflight")
        return train.PreflightReport(checks=("passed",))

    def select(spec: TrainSpec, settings: TrainSettings) -> FakeBackend:
        assert settings is expected_settings
        events.append("select")
        return backend

    async def train_backend(self: FakeBackend, spec: TrainSpec, *, sink: RunSink, work_dir: Path) -> Checkpoint:
        events.append("train")
        return await original_train(self, spec, sink=sink, work_dir=work_dir)

    monkeypatch.setattr(train, "preflight", preflight)
    monkeypatch.setattr(train, "select", select)
    monkeypatch.setattr(FakeBackend, "train", train_backend)
    stub_bakeoff(monkeypatch, leaderboard(winner=train.TRAINED_ARM, passed_gate=True))

    await train.run(request, evaluation=bakeoff)

    assert events == ["preflight", "select", "train"]


async def test_the_registry_owns_a_frozen_copy_of_the_weights_it_registers(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend
) -> None:
    stub_bakeoff(monkeypatch, leaderboard(winner=train.TRAINED_ARM, passed_gate=True))

    result = await train.run(spec(), evaluation=evaluation())

    assert train.model_path(result.version) == result.version.path / train.MODEL_DIR
    assert weights_of(result.version) == b"weights-1"
    assert (train.model_path(result.version) / WEIGHTS).stat().st_mode & 0o222 == 0


async def test_a_second_run_cannot_mutate_a_registered_or_promoted_version(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend, tmp_path: Path
) -> None:
    # MUTABLE CHECKPOINTS: a registered version used to be a pointer into the run's scratch dir,
    # so a rerun of the family overwrote v001's weights — even when the challenger lost.
    stub_bakeoff(
        monkeypatch,
        leaderboard(winner=train.TRAINED_ARM, passed_gate=True),
        leaderboard(winner="base", passed_gate=False, metric=0.2),
    )

    winner = await train.run(spec(), evaluation=evaluation())
    challenger = await train.run(spec(), evaluation=evaluation())

    assert (winner.promoted, challenger.promoted) == (True, False)
    assert backend.work_dirs[0] != backend.work_dirs[1]
    assert weights_of(winner.version) == b"weights-1"
    assert weights_of(challenger.version) == b"weights-2"
    promoted = await registry.current("watcher", root=tmp_path / "registry")
    assert promoted is not None and promoted.version == winner.version.version
    assert weights_of(promoted) == b"weights-1"


async def test_every_run_gets_its_own_work_dir(monkeypatch: pytest.MonkeyPatch, backend: FakeBackend) -> None:
    stub_bakeoff(monkeypatch, leaderboard(winner=train.TRAINED_ARM, passed_gate=True))

    await train.run(spec(), evaluation=evaluation())
    await train.run(spec(), evaluation=evaluation())

    first, second = backend.work_dirs
    assert first != second
    assert first.parent == second.parent == load(TrainSettings).work_root / "watcher"
    assert (first / "fused" / WEIGHTS).read_bytes() == b"weights-1"
    assert (second / "fused" / WEIGHTS).read_bytes() == b"weights-2"


async def test_every_run_evaluates_on_its_own_port(monkeypatch: pytest.MonkeyPatch, backend: FakeBackend) -> None:
    # PORT COLLISION: every evaluation used to serve on the one hardcoded port 8410, so two
    # concurrent runs contended for a single server identity and one saw the other's model.
    stub_bakeoff(monkeypatch, leaderboard(winner=train.TRAINED_ARM, passed_gate=True))

    await train.run(spec(), evaluation=evaluation())
    await train.run(spec(), evaluation=evaluation())

    ports = [port for _, port in FakeServer.ensured]
    assert len(set(ports)) == 2
    assert all(port > 0 for port in ports)
    assert FakeServer.stopped == FakeServer.ensured


async def test_run_appends_the_trained_arm_to_the_evaluation_bakeoff(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend
) -> None:
    ran = stub_bakeoff(monkeypatch, leaderboard(winner=train.TRAINED_ARM, passed_gate=True))

    await train.run(spec(), evaluation=evaluation())

    arms = ran[0].arms
    assert [arm.name for arm in arms] == ["base", train.TRAINED_ARM]
    assert arms[0] is BASELINE  # the caller's baseline arm is untouched
    assert arms[-1].base_url == f"http://127.0.0.1:{FakeServer.ensured[0][1]}/v1"
    assert arms[-1].model == str(backend.checkpoints[0].mlx_path)
    assert arms[-1].client_factory is not None
    assert (ran[0].task, ran[0].corpus, ran[0].primary_metric) == (task, ("a", "b"), "exact")


async def test_run_serves_the_fused_artifact_and_tears_it_down(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend
) -> None:
    stub_bakeoff(monkeypatch, leaderboard(winner="base", passed_gate=False))

    await train.run(spec(), evaluation=evaluation())

    assert [model for model, _ in FakeServer.ensured] == [str(backend.checkpoints[0].mlx_path)]
    assert FakeServer.stopped == FakeServer.ensured


async def test_run_stops_the_server_when_the_bakeoff_raises(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend
) -> None:
    from athome import bakeoff

    async def boom(spec: BakeoffSpec) -> Leaderboard:
        raise RuntimeError("bake-off exploded")

    monkeypatch.setattr(bakeoff, "run", boom)

    with pytest.raises(RuntimeError, match="exploded"):
        await train.run(spec(), evaluation=evaluation())
    assert FakeServer.stopped == FakeServer.ensured
    assert [model for model, _ in FakeServer.stopped] == [str(backend.checkpoints[0].mlx_path)]


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


async def test_run_journals_progress_to_its_own_work_dir(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend, tmp_path: Path
) -> None:
    stub_bakeoff(monkeypatch, leaderboard(winner=train.TRAINED_ARM, passed_gate=True))

    result = await train.run(spec(), evaluation=evaluation())

    journal = backend.work_dirs[0] / train.JOURNAL_FILE
    records = load_journal(journal)
    assert journal.parent.parent == tmp_path / "runs" / "watcher"
    assert [record["event"] for record in records] == ["selected", "step", "trained", "registered"]
    assert records[0]["backend"] == "tinker"
    assert records[0]["work_dir"] == str(backend.work_dirs[0])
    assert records[-1]["version"] == result.version.version
    assert records[-1]["promoted"] is True
    assert backend.sinks[0].path == journal


async def test_register_copies_the_model_into_the_version_and_never_promotes(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    fused = fuse(tmp_path / "scratch", b"weights-manual")

    info = await train.register("watcher", fused, {"source": "manual"}, root=root)

    assert info.metadata["source_mlx_path"] == str(fused)
    assert info.metadata["model_dir"] == train.MODEL_DIR
    assert info.metadata["source"] == "manual"
    assert weights_of(info) == b"weights-manual"
    assert json.loads((info.path / train.CHECKPOINT_FILE).read_text()) == {
        "source_mlx_path": str(fused),
        "model_dir": train.MODEL_DIR,
        "source": "manual",
    }
    assert await registry.current("watcher", root=root) is None
