from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import anyio
import pytest
from loguru import logger

from athome.progress import (
    FailureBudgetExceeded,
    PhaseMissing,
    Phases,
    ReservedFieldConflict,
    RunSink,
    WorkSet,
    append_line,
    load_journal,
)

if TYPE_CHECKING:
    from pathlib import Path


async def test_done_and_pending_resume_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "work.jsonl"
    work = WorkSet.open(path)
    await work.done("a")
    await work.done("c")
    assert work.is_done("a")
    assert work.pending(["a", "b", "c", "d"]) == ["b", "d"]

    resumed = WorkSet.open(path)
    assert resumed.is_done("a") and resumed.is_done("c")
    assert not resumed.is_done("b")
    assert resumed.pending(["a", "b", "c", "d"]) == ["b", "d"]


async def test_error_unit_stays_pending_on_resume(tmp_path: Path) -> None:
    path = tmp_path / "work.jsonl"
    work = WorkSet.open(path)
    await work.done("a")
    await work.error("b", "boom")
    assert not work.is_done("b")

    resumed = WorkSet.open(path)
    assert resumed.is_done("a")
    assert not resumed.is_done("b")
    assert resumed.pending(["a", "b"]) == ["b"]


async def test_error_then_done_completes_on_resume(tmp_path: Path) -> None:
    path = tmp_path / "work.jsonl"
    work = WorkSet.open(path)
    await work.error("b", "first attempt failed")
    await work.done("b")
    assert WorkSet.open(path).is_done("b")


async def test_pending_preserves_input_order(tmp_path: Path) -> None:
    work = WorkSet.open(tmp_path / "work.jsonl")
    await work.done("m")
    assert work.pending(["z", "m", "a", "q", "b"]) == ["z", "a", "q", "b"]


async def test_done_journals_extra_fields(tmp_path: Path) -> None:
    path = tmp_path / "work.jsonl"
    work = WorkSet.open(path)
    await work.done("a", cost=1.5, attempts=2)
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert '"cost": 1.5' in lines[0]
    assert WorkSet.open(path).is_done("a")


async def test_done_rejects_reserved_status_field(tmp_path: Path) -> None:
    path = tmp_path / "work.jsonl"
    work = WorkSet.open(path)
    with pytest.raises(ReservedFieldConflict):
        await work.done("a", status="error")
    assert not path.exists()
    assert not work.is_done("a")


async def test_truncated_tail_is_skipped_with_a_warning(tmp_path: Path) -> None:
    path = tmp_path / "work.jsonl"
    work = WorkSet.open(path)
    await work.done("a")
    with path.open("a") as file:
        file.write('{"unit": "b", "stat')

    messages: list[str] = []
    handler = logger.add(messages.append, level="WARNING")
    try:
        resumed = WorkSet.open(path)
    finally:
        logger.remove(handler)

    assert resumed.is_done("a")
    assert not resumed.is_done("b")
    assert any("truncated final line" in message for message in messages)


async def test_corrupt_non_final_line_raises(tmp_path: Path) -> None:
    path = tmp_path / "work.jsonl"
    with path.open("w") as file:
        file.write('not json\n{"unit": "a", "status": "done"}\n')
    with pytest.raises(ValueError):
        WorkSet.open(path)


async def test_runsink_append_and_failures_count(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    sink = RunSink.open(path, failure_budget=5)
    await sink.append({"item": "a", "ok": True})
    await sink.fail({"item": "b"})
    assert sink.failures == 1
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert '"failed": true' in lines[1]
    assert "failed" not in lines[0]


@pytest.mark.parametrize("budget", [0, 1, 3])
async def test_failure_budget_raises_at_the_boundary_exactly(tmp_path: Path, budget: int) -> None:
    sink = RunSink.open(tmp_path / "out.jsonl", failure_budget=budget)
    for _ in range(budget):
        await sink.fail({"x": 1})
    assert sink.failures == budget
    with pytest.raises(FailureBudgetExceeded):
        await sink.fail({"x": 1})
    assert sink.failures == budget + 1


async def test_failure_budget_survives_resume(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    sink = RunSink.open(path, failure_budget=2)
    await sink.fail({"x": 1})
    await sink.fail({"x": 2})
    assert sink.failures == 2

    resumed = RunSink.open(path, failure_budget=2)
    assert resumed.failures == 2
    with pytest.raises(FailureBudgetExceeded):
        await resumed.fail({"x": 3})


async def test_phases_mark_require_and_done(tmp_path: Path) -> None:
    path = tmp_path / "phases.jsonl"
    phases = Phases.open(path)
    assert not phases.done("extract")
    with pytest.raises(PhaseMissing):
        phases.require("extract")

    await phases.mark("extract")
    assert phases.done("extract")
    phases.require("extract")

    resumed = Phases.open(path)
    assert resumed.done("extract")
    resumed.require("extract")
    with pytest.raises(PhaseMissing):
        resumed.require("refine")


async def test_phase_missing_carries_the_phase_name(tmp_path: Path) -> None:
    phases = Phases.open(tmp_path / "phases.jsonl")
    with pytest.raises(PhaseMissing, match="refine"):
        phases.require("refine")


async def test_append_is_one_kernel_write_per_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_write = os.write
    writes: list[bytes] = []

    def recording_write(fd: int, data: bytes) -> int:
        writes.append(data)
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", recording_write)
    await append_line(tmp_path / "out.jsonl", {"unit": "a", "status": "done"})

    record_writes = [data for data in writes if b'"unit"' in data]
    assert len(record_writes) == 1
    assert record_writes[0].endswith(b"\n")
    assert json.loads(record_writes[0]) == {"unit": "a", "status": "done"}


async def test_append_loops_past_a_short_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "out.jsonl"
    real_write = os.write
    calls = {"n": 0}

    def short_write(fd: int, data: bytes) -> int:
        calls["n"] += 1
        return real_write(fd, bytes(data)[:5]) if calls["n"] == 1 else real_write(fd, data)

    monkeypatch.setattr(os, "write", short_write)
    record = {"unit": "a", "status": "done", "payload": "x" * 200}
    await append_line(path, record)

    assert calls["n"] >= 2
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == record


async def test_concurrent_appends_never_splice_records(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    records = [{"i": i, "blob": str(i) * 8000} for i in range(30)]
    async with anyio.create_task_group() as tg:
        for record in records:
            tg.start_soon(append_line, path, record)

    parsed = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(parsed) == len(records)
    assert sorted(rec["i"] for rec in parsed) == list(range(30))


async def test_truncated_tail_after_append_skips_only_partial_final_line(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    await append_line(path, {"i": 0})
    await append_line(path, {"i": 1})
    with path.open("a") as file:
        file.write('{"i": 2, "par')

    messages: list[str] = []
    handler = logger.add(messages.append, level="WARNING")
    try:
        loaded = load_journal(path)
    finally:
        logger.remove(handler)
    assert [rec["i"] for rec in loaded] == [0, 1]
    assert any("truncated final line" in message for message in messages)
