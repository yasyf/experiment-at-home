from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from athome.llm.spend import SpendGuard
from athome.train.gate import GateVerdict
from athome.train.retrain import RetrainOutcome, retrain
from athome.train.spec import (
    BASE_MODELS,
    CheckpointPolicy,
    EvalRow,
    Hyperparams,
    LocalJsonlRef,
    SavedCheckpoint,
    TrainReport,
    TrainSpec,
    UnservableBase,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

TRAIN_COST_USD = 1.25
BUDGET_CAP_USD = 100.0
FIT_PROJECTED_USD = 3.0
CHECKPOINTS: tuple[SavedCheckpoint, ...] = (
    SavedCheckpoint(step=3, path="tinker://run/3", final=False, scores=None),
    SavedCheckpoint(step=6, path="tinker://run/6", final=False, scores=None),
    SavedCheckpoint(step=9, path="tinker://run/9", final=False, scores=None),
    SavedCheckpoint(step=12, path="tinker://run/12", final=True, scores=None),
)
# The argmax is a middle checkpoint, never the final — proving retrain materializes what select
# picks, not the last-saved snapshot.
SELECT_SCORES: dict[int, float] = {3: 0.10, 6: 0.90, 9: 0.40, 12: 0.20}
BEST = CHECKPOINTS[1]
CHECKPOINT_POLICY = CheckpointPolicy(at=(0.25, 0.5, 0.75))
EVAL_ROWS: tuple[EvalRow, ...] = (EvalRow(tokens=(1, 2, 3), weights=(0.0, 0.0, 1.0)),)
SERVED: dict[str, float] = {"auc": 0.82, "drift": 0.03}
VERDICT = GateVerdict(promote=True, reason="clears the bar", stats={"auc": 0.82})


@dataclass(frozen=True, slots=True)
class FakeAdapter:
    step: int
    adapter_dir: Path
    train_cost_usd: float


@dataclass(frozen=True, slots=True)
class FitCall:
    spec: TrainSpec
    sink: object
    budget: SpendGuard
    checkpoints: CheckpointPolicy
    eval_rows: object


@dataclass(frozen=True, slots=True)
class MaterializeCall:
    saved: SavedCheckpoint
    spec: TrainSpec
    work_dir: Path
    cost: float


@dataclass(slots=True)
class ScriptedBackend:
    """A duck-typed backend recording every call, returning a canned report and adapter.

    ``fit`` and ``materialize`` are the only calls ``retrain`` may make; ``score``/``fuse``/``train``
    are tripwires that fail loudly if the orchestrator ever leaks a side effect through them.
    """

    report: TrainReport
    adapter: FakeAdapter
    calls: list[str] = field(default_factory=list)
    fit_calls: list[FitCall] = field(default_factory=list)
    materialize_calls: list[MaterializeCall] = field(default_factory=list)

    async def fit(
        self,
        spec: TrainSpec,
        *,
        sink: object,
        budget: SpendGuard,
        checkpoints: CheckpointPolicy,
        eval_rows: Sequence[EvalRow] | None,
    ) -> TrainReport:
        self.calls.append("fit")
        self.fit_calls.append(FitCall(spec, sink, budget, checkpoints, eval_rows))
        await budget.check(FIT_PROJECTED_USD)
        await budget.record(FIT_PROJECTED_USD, self.report.train_cost_usd)
        return self.report

    async def materialize(self, saved: SavedCheckpoint, spec: TrainSpec, *, work_dir: Path, cost: float) -> FakeAdapter:
        self.calls.append("materialize")
        self.materialize_calls.append(MaterializeCall(saved, spec, work_dir, cost))
        return self.adapter

    async def score(self, *args: object, **kwargs: object) -> object:
        self.calls.append("score")
        raise AssertionError("retrain must not call backend.score")

    async def fuse(self, *args: object, **kwargs: object) -> object:
        self.calls.append("fuse")
        raise AssertionError("retrain must not call backend.fuse")

    async def train(self, *args: object, **kwargs: object) -> object:
        self.calls.append("train")
        raise AssertionError("retrain must not call backend.train")


def report(*, checkpoints: tuple[SavedCheckpoint, ...] = CHECKPOINTS) -> TrainReport:
    return TrainReport(
        method="sft",
        steps=(),
        checkpoints=checkpoints,
        dropped=0,
        wall_s=4.0,
        train_cost_usd=TRAIN_COST_USD,
    )


def spec() -> TrainSpec:
    return TrainSpec(
        name="watcher",
        base=BASE_MODELS["qwen3-8b"],
        dataset=LocalJsonlRef(path=Path("corpus.jsonl")),
        hyperparams=Hyperparams(steps=12, batch_size=16),
    )


def adapter() -> FakeAdapter:
    return FakeAdapter(step=BEST.step, adapter_dir=Path("/runs/watcher/adapter"), train_cost_usd=TRAIN_COST_USD)


def backend() -> ScriptedBackend:
    return ScriptedBackend(report=report(), adapter=adapter())


def guard() -> SpendGuard:
    return SpendGuard(max_usd=BUDGET_CAP_USD)


def by_step_select(scores: dict[int, float] = SELECT_SCORES) -> Callable[[SavedCheckpoint], float]:
    return lambda saved: scores[saved.step]


async def run(
    *,
    be: ScriptedBackend,
    budget: SpendGuard | None = None,
    select: Callable[[SavedCheckpoint], float] | None = None,
    artifact_scorer: Callable[[FakeAdapter], dict[str, float]] | None = None,
    gate: Callable[[dict[str, float]], GateVerdict] | None = None,
    sink: object,
    work_dir: Path,
) -> RetrainOutcome:
    return await retrain(
        be,  # type: ignore[arg-type]
        spec(),
        checkpoints=CHECKPOINT_POLICY,
        eval_rows=EVAL_ROWS,
        budget=budget or guard(),
        select=select or by_step_select(),
        artifact_scorer=artifact_scorer or (lambda _adapter: SERVED),
        gate=gate or (lambda _served: VERDICT),
        work_dir=work_dir,
        sink=sink,
    )


async def test_fit_receives_spec_sink_budget_checkpoints_and_eval_rows_verbatim(tmp_path: Path) -> None:
    be, sink, train_spec, envelope = backend(), object(), spec(), guard()

    await retrain(
        be,  # type: ignore[arg-type]
        train_spec,
        checkpoints=CHECKPOINT_POLICY,
        eval_rows=EVAL_ROWS,
        budget=envelope,
        select=by_step_select(),
        artifact_scorer=lambda _adapter: SERVED,
        gate=lambda _served: VERDICT,
        work_dir=tmp_path,
        sink=sink,
    )

    [call] = be.fit_calls
    assert call.spec is train_spec
    assert call.sink is sink
    assert call.budget is envelope
    assert call.checkpoints is CHECKPOINT_POLICY
    assert call.eval_rows is EVAL_ROWS


async def test_select_runs_on_every_checkpoint_and_its_argmax_is_materialized(tmp_path: Path) -> None:
    be, seen = backend(), []

    def select(saved: SavedCheckpoint) -> float:
        seen.append(saved)
        return SELECT_SCORES[saved.step]

    outcome = await run(be=be, select=select, sink=object(), work_dir=tmp_path)

    assert seen == list(CHECKPOINTS)
    [call] = be.materialize_calls
    assert call.saved is BEST
    assert outcome.best is BEST


async def test_raising_select_propagates_and_materialize_never_runs(tmp_path: Path) -> None:
    be = backend()

    def boom(saved: SavedCheckpoint) -> float:
        raise ValueError(f"no score for step {saved.step}")

    with pytest.raises(ValueError, match="no score for step"):
        await run(be=be, select=boom, sink=object(), work_dir=tmp_path)

    assert be.calls == ["fit"]
    assert be.materialize_calls == []


async def test_materialize_receives_best_spec_work_dir_and_report_cost(tmp_path: Path) -> None:
    be, train_spec = backend(), spec()

    await retrain(
        be,  # type: ignore[arg-type]
        train_spec,
        checkpoints=CHECKPOINT_POLICY,
        eval_rows=EVAL_ROWS,
        budget=guard(),
        select=by_step_select(),
        artifact_scorer=lambda _adapter: SERVED,
        gate=lambda _served: VERDICT,
        work_dir=tmp_path,
        sink=object(),
    )

    [call] = be.materialize_calls
    assert call.saved is BEST
    assert call.spec is train_spec
    assert call.work_dir is tmp_path
    assert call.cost == TRAIN_COST_USD


async def test_artifact_scorer_receives_the_materialized_adapter_and_its_dict_lands_in_served(
    tmp_path: Path,
) -> None:
    be, received = backend(), []

    def scorer(passed: FakeAdapter) -> dict[str, float]:
        received.append(passed)
        return SERVED

    outcome = await run(be=be, artifact_scorer=scorer, sink=object(), work_dir=tmp_path)

    assert received == [be.adapter]
    assert outcome.adapter is be.adapter
    assert outcome.served is SERVED


async def test_gate_receives_the_served_dict_and_its_verdict_lands_in_the_outcome(tmp_path: Path) -> None:
    be, received = backend(), []

    def gate(served: dict[str, float]) -> GateVerdict:
        received.append(served)
        return VERDICT

    outcome = await run(be=be, gate=gate, sink=object(), work_dir=tmp_path)

    assert received == [SERVED]
    assert received[0] is SERVED
    assert outcome.verdict is VERDICT


async def test_retrain_refuses_a_hosted_only_base_before_any_billable_call(tmp_path: Path) -> None:
    be = backend()

    with pytest.raises(UnservableBase, match="mlx-lm LoRA counterpart"):
        await retrain(
            be,  # type: ignore[arg-type]
            replace(spec(), base=BASE_MODELS["qwen3.5-4b"]),
            checkpoints=CHECKPOINT_POLICY,
            eval_rows=EVAL_ROWS,
            budget=guard(),
            select=by_step_select(),
            artifact_scorer=lambda _adapter: SERVED,
            gate=lambda _served: VERDICT,
            work_dir=tmp_path,
            sink=object(),
        )

    assert be.calls == []


async def test_backend_sees_only_fit_and_materialize(tmp_path: Path) -> None:
    be = backend()

    await run(be=be, sink=object(), work_dir=tmp_path)

    assert be.calls == ["fit", "materialize"]


async def test_retrain_outcome_field_integrity(tmp_path: Path) -> None:
    be = backend()

    outcome = await run(be=be, sink=object(), work_dir=tmp_path)

    assert isinstance(outcome, RetrainOutcome)
    assert outcome.report is be.report
    assert outcome.best is BEST
    assert outcome.adapter is be.adapter
    assert outcome.served is SERVED
    assert outcome.verdict is VERDICT


async def test_retrain_bills_fit_against_the_callers_guard(tmp_path: Path) -> None:
    be, envelope = backend(), SpendGuard(max_usd=5.0)

    await run(be=be, budget=envelope, sink=object(), work_dir=tmp_path)

    [call] = be.fit_calls
    assert call.budget is envelope
    assert envelope.reserved == 0.0
    assert envelope.spent == TRAIN_COST_USD
