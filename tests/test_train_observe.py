from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

import pytest

from athome.config import load
from athome.llm.spend import SpendExceeded, SpendGuard
from athome.progress import RunSink
from athome.train import data, tinker
from athome.train.observe import ObserveOutcome, observe
from athome.train.spec import (
    BASE_MODELS,
    CheckpointPolicy,
    EvalRow,
    Hyperparams,
    LocalJsonlRef,
    TrainReport,
    TrainSpec,
)
from athome.train.tinker import TinkerBackend, tinker_model
from tests.tinker_fakes import LOGPROB, FakeModelInput, FakeService, FakeTokenizer, install_sdk

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path
    from types import ModuleType

    from athome.train.spec import SavedCheckpoint


@dataclasses.dataclass(slots=True)
class Sampling:
    paths: list[str | None] = dataclasses.field(default_factory=list)
    calls: list[tuple[int, ...]] = dataclasses.field(default_factory=list)

    async def compute_logprobs_async(self, prompt: FakeModelInput) -> list[float | None]:
        self.calls.append(tuple(prompt.ids))
        return [None, *([LOGPROB] * (len(prompt.ids) - 1))]


@dataclasses.dataclass(slots=True)
class SamplingService(FakeService):
    sampling: Sampling = dataclasses.field(default_factory=Sampling)

    async def create_sampling_client_async(
        self, model_path: str | None = None, base_model: str | None = None
    ) -> Sampling:
        self.sampling.paths.append(model_path)
        return self.sampling


@dataclasses.dataclass(slots=True)
class EmptyReportBackend:
    """A backend whose ``fit`` yields no checkpoints, so ``observe``'s ``max`` has nothing to rank."""

    scored: list[str] = dataclasses.field(default_factory=list)

    async def fit(
        self,
        spec: TrainSpec,
        *,
        sink: RunSink,
        budget: SpendGuard,
        checkpoints: CheckpointPolicy,
        eval_rows: Sequence[EvalRow] | None,
        run_tag: str = "",
    ) -> TrainReport:
        return TrainReport(method="sft", steps=(), checkpoints=(), dropped=0, wall_s=0.0, train_cost_usd=0.0)

    async def score(self, path: str, rows: Sequence[EvalRow], *, base: object, budget: SpendGuard) -> object:
        self.scored.append(path)
        raise AssertionError("score must not run after max() has already crashed")


@pytest.fixture(autouse=True)
def sdk(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return install_sdk(monkeypatch)


@pytest.fixture(autouse=True)
def fake_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "tokenizer", lambda mlx_id: FakeTokenizer())


@pytest.fixture(autouse=True)
def tinker_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("TINKER_API_KEY", "sk-tinker-test")
    monkeypatch.setenv("ATHOME_TRAIN_WORK_ROOT", str(tmp_path / "runs"))
    monkeypatch.setattr(tinker, "TINKER_ENV", tmp_path / "tinker.env")
    load.cache_clear()
    yield
    load.cache_clear()


@pytest.fixture
def sampling_service(sdk: ModuleType) -> SamplingService:
    instance = SamplingService("sk-tinker-test")
    sdk.ServiceClient = lambda api_key: instance
    return instance


def corpus(tmp_path: Path, *, rows: int = 4) -> LocalJsonlRef:
    lines = [
        {"messages": [{"role": "user", "content": f"q{index}"}, {"role": "assistant", "content": "yes"}]}
        for index in range(rows)
    ]
    path = tmp_path / "sft.jsonl"
    path.write_text("".join(f"{json.dumps(line)}\n" for line in lines))
    return LocalJsonlRef(path=path)


def spec(dataset: LocalJsonlRef, **overrides: object) -> TrainSpec:
    return dataclasses.replace(
        TrainSpec(
            name="watcher",
            base=BASE_MODELS["qwen3-8b"],
            dataset=dataset,
            hyperparams=Hyperparams(steps=3, batch_size=2, learning_rate=2e-4, max_seq_len=64),
        ),
        **overrides,
    )


def sink(tmp_path: Path) -> RunSink:
    return RunSink.open(tmp_path / "run.jsonl")


def budget(max_usd: float | None = 60.0) -> SpendGuard:
    return SpendGuard(max_usd=max_usd)


def by_step(checkpoint: SavedCheckpoint) -> float:
    return checkpoint.step


async def test_one_envelope_spans_fit_and_score_so_fit_actuals_reduce_score_headroom(
    sampling_service: SamplingService, tmp_path: Path
) -> None:
    backend = TinkerBackend.from_settings()
    request = spec(corpus(tmp_path))
    rows = (EvalRow(tokens=(1, 2, 3), weights=(0.0, 0.0, 1.0)),)

    fit_probe = SpendGuard(max_usd=None)
    await backend.fit(request, sink=sink(tmp_path), budget=fit_probe)
    fit_spent = fit_probe.spent
    score_projection = backend.cost(model=tinker_model(request.base), prefill=3, sample=1)
    assert fit_spent > 0.0

    envelope = SpendGuard(max_usd=fit_spent + score_projection / 2)
    with pytest.raises(SpendExceeded):
        await observe(backend, request, rows, budget=envelope, select=by_step, sink=sink(tmp_path))

    assert envelope.spent == pytest.approx(fit_spent)
    assert sampling_service.sampling.calls == []


async def test_best_is_the_argmax_of_select_over_the_report_checkpoints(
    sampling_service: SamplingService, tmp_path: Path
) -> None:
    backend = TinkerBackend.from_settings()
    request = spec(corpus(tmp_path))
    rows = (EvalRow(tokens=(1, 2, 3), weights=(0.0, 0.0, 1.0)),)

    outcome = await observe(
        backend,
        request,
        rows,
        budget=budget(),
        select=lambda checkpoint: -checkpoint.step,
        sink=sink(tmp_path),
        checkpoints=CheckpointPolicy(at=(0.5,)),
    )

    assert isinstance(outcome, ObserveOutcome)
    assert len(outcome.report.checkpoints) == 2
    assert outcome.best is min(outcome.report.checkpoints, key=by_step)
    assert outcome.best is not outcome.report.final
    assert outcome.best.step < outcome.report.final.step
    assert sampling_service.sampling.paths == [outcome.best.sampler_path]
    assert len(outcome.scored) == 1


async def test_a_hosted_only_base_is_accepted_end_to_end(sampling_service: SamplingService, tmp_path: Path) -> None:
    hosted_only = BASE_MODELS["qwen3.5-4b"]
    assert hosted_only.serves_locally is False
    backend = TinkerBackend.from_settings()
    request = spec(corpus(tmp_path), base=hosted_only)
    rows = (EvalRow(tokens=(5, 6, 7), weights=(0.0, 0.0, 1.0)),)

    outcome = await observe(backend, request, rows, budget=budget(), select=by_step, sink=sink(tmp_path))

    assert outcome.best is outcome.report.final
    assert outcome.best.sampler_path.startswith("tinker://")
    assert sampling_service.sampling.paths == [outcome.best.sampler_path]
    assert len(outcome.scored) == 1


async def test_a_spend_exceeded_from_fit_propagates_before_any_score_spend(
    sampling_service: SamplingService, tmp_path: Path
) -> None:
    backend = TinkerBackend.from_settings()
    request = spec(corpus(tmp_path))
    rows = (EvalRow(tokens=(1, 2, 3), weights=(0.0, 0.0, 1.0)),)

    with pytest.raises(SpendExceeded):
        await observe(backend, request, rows, budget=SpendGuard(max_usd=1e-6), select=by_step, sink=sink(tmp_path))

    assert sampling_service.sampling.calls == []
    assert sampling_service.sampling.paths == []


async def test_an_empty_report_checkpoints_crashes_on_max(tmp_path: Path) -> None:
    backend = EmptyReportBackend()
    request = spec(corpus(tmp_path))
    rows = (EvalRow(tokens=(1, 2, 3), weights=(0.0, 0.0, 1.0)),)

    with pytest.raises(ValueError, match="empty"):
        await observe(backend, request, rows, budget=budget(), select=by_step, sink=sink(tmp_path))

    assert backend.scored == []


async def test_two_observes_of_one_spec_save_disjoint_state_names(
    sampling_service: SamplingService, tmp_path: Path
) -> None:
    backend = TinkerBackend.from_settings()
    request = spec(corpus(tmp_path))
    rows = (EvalRow(tokens=(1, 2, 3), weights=(0.0, 0.0, 1.0)),)

    first = await observe(backend, request, rows, budget=budget(), select=by_step, sink=sink(tmp_path / "a"))
    second = await observe(backend, request, rows, budget=budget(), select=by_step, sink=sink(tmp_path / "b"))

    assert first.best.state != second.best.state
