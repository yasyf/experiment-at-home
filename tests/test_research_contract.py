from __future__ import annotations

from typing import TYPE_CHECKING

from athome.progress import RunSink
from athome.research.contract import Memory, build_contract, render_history
from athome.research.journal import Journal, JournalRow, Verdict
from athome.research.spec import Budget, ExperimentSpec

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def row(
    unit: int, *, metric: float | None, verdict: Verdict = Verdict.KEEP, description: str = "stub edited train.py"
) -> JournalRow:
    return JournalRow(
        unit=unit, commit=f"c{unit}", metric=metric, verdict=verdict, resources={"usd": 0.0}, description=description
    )


def journal_with(tmp_path: Path, rows: Sequence[JournalRow]) -> Journal:
    return Journal(RunSink.open(tmp_path / "j.jsonl"), False, list(rows))


def toy_spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="toy",
        metric_command=("python", "score.py"),
        metric_key="loss",
        direction="min",
        mutable_paths=("train.py",),
        immutable_paths=("score.py",),
        budget=Budget(max_units=3),
    )


def test_memory_from_journal_truncates_to_the_last_ten_oldest_first(tmp_path: Path) -> None:
    journal = journal_with(tmp_path, [row(i, metric=float(i)) for i in range(15)])

    memory = Memory.from_journal(journal, baseline=99.0, incumbent=0.0, direction="min")

    assert [r.unit for r in memory.recent] == list(range(5, 15))  # last 10, oldest first
    assert memory.baseline == 99.0 and memory.incumbent == 0.0
    assert memory.best is not None and memory.best.unit == 0  # best spans the whole journal, not just recent


def test_memory_from_journal_keeps_all_rows_when_fewer_than_ten(tmp_path: Path) -> None:
    journal = journal_with(
        tmp_path, [row(0, metric=1.0, verdict=Verdict.DISCARD, description="lost"), row(1, metric=0.5)]
    )

    memory = Memory.from_journal(journal, baseline=2.0, incumbent=0.5, direction="min")

    assert [r.unit for r in memory.recent] == [0, 1]  # all rows, oldest first
    assert memory.best is not None and memory.best.unit == 1  # only KEEP rows are eligible for best


def test_render_history_renders_head_and_row_lines(tmp_path: Path) -> None:
    rows = [
        row(
            0,
            metric=None,
            verdict=Verdict.DISCARD,
            description="hostile edited score.py | ImmutableViolation: ['score.py']",
        ),
        row(1, metric=0.5, description="stub edited train.py"),
    ]
    memory = Memory.from_journal(journal_with(tmp_path, rows), baseline=1.0, incumbent=0.5, direction="min")

    rendered = render_history(memory)

    assert rendered.splitlines()[0] == "## History"
    assert "- baseline (untouched tree): 1.0" in rendered
    assert "- incumbent: 0.5" in rendered
    assert "- best kept: 0.5" in rendered
    assert "unit 0 [discard] metric=unscored — hostile edited score.py | ImmutableViolation: ['score.py']" in rendered
    assert "unit 1 [keep] metric=0.5 — stub edited train.py" in rendered


def test_render_history_handles_an_empty_memory() -> None:
    rendered = render_history(Memory(baseline=None, incumbent=None, best=None, recent=()))

    assert "## History" in rendered
    assert "- baseline (untouched tree): unscored" in rendered
    assert "- best kept: none" in rendered


def test_build_contract_appends_history_after_simplicity(tmp_path: Path) -> None:
    memory = Memory.from_journal(
        journal_with(tmp_path, [row(0, metric=0.5)]), baseline=1.0, incumbent=0.5, direction="min"
    )

    contract = build_contract(toy_spec(), budget_low=False, memory=memory)

    assert contract.index("## Simplicity") < contract.index("## History")
    assert "unit 0 [keep] metric=0.5 — stub edited train.py" in contract
