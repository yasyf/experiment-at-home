from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import pytest

from athome.research.journal import CC_NOTES_BIN, Journal, JournalRow, Verdict

if TYPE_CHECKING:
    from pathlib import Path


def make_row(unit: int, metric: float | None, verdict: Verdict, *, commit: str = "abc123") -> JournalRow:
    return JournalRow(
        unit=unit,
        commit=commit,
        metric=metric,
        verdict=verdict,
        resources={"wall_s": 1.0, "usd": 0.01},
        description=f"unit {unit}",
    )


async def test_resume_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = Journal.open(path)
    await journal.append(make_row(0, 0.5, Verdict.KEEP))
    await journal.append(make_row(1, 0.4, Verdict.DISCARD))
    assert journal.resume_unit() == 2

    resumed = Journal.open(path)
    assert resumed.resume_unit() == 2
    assert resumed.rows() == [make_row(0, 0.5, Verdict.KEEP), make_row(1, 0.4, Verdict.DISCARD)]


async def test_resume_unit_is_zero_on_empty_journal(tmp_path: Path) -> None:
    assert Journal.open(tmp_path / "journal.jsonl").resume_unit() == 0


async def test_journal_before_result_row_is_durable_before_the_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("cc-notes unavailable")

    monkeypatch.setattr(anyio, "run_process", boom)
    path = tmp_path / "journal.jsonl"
    journal = Journal.open(path, mirror_cc_notes=True)
    with pytest.raises(RuntimeError):
        await journal.append(make_row(0, 0.5, Verdict.KEEP))

    assert Journal.open(path).rows() == [make_row(0, 0.5, Verdict.KEEP)]


async def test_best_picks_the_best_kept_row(tmp_path: Path) -> None:
    journal = Journal.open(tmp_path / "journal.jsonl")
    await journal.append(make_row(0, 0.5, Verdict.KEEP))
    await journal.append(make_row(1, 0.7, Verdict.KEEP))
    await journal.append(make_row(2, 0.9, Verdict.DISCARD))
    await journal.append(make_row(3, None, Verdict.CRASH))
    assert journal.best("max") == make_row(1, 0.7, Verdict.KEEP)
    assert journal.best("min") == make_row(0, 0.5, Verdict.KEEP)


async def test_best_is_none_without_a_kept_row(tmp_path: Path) -> None:
    journal = Journal.open(tmp_path / "journal.jsonl")
    await journal.append(make_row(0, 0.9, Verdict.DISCARD))
    await journal.append(make_row(1, None, Verdict.CRASH))
    assert journal.best("max") is None


async def test_cc_notes_mirror_invokes_the_installed_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    async def fake(command: list[str], *args: object, **kwargs: object) -> None:
        commands.append(command)

    monkeypatch.setattr(anyio, "run_process", fake)
    journal = Journal.open(tmp_path / "journal.jsonl", mirror_cc_notes=True)
    await journal.append(make_row(2, 0.5, Verdict.KEEP))

    assert str(CC_NOTES_BIN) == "/opt/homebrew/bin/cc-notes"
    assert len(commands) == 1
    assert commands[0][0] == "/opt/homebrew/bin/cc-notes"
    assert "uvx" not in commands[0]
    assert commands[0][1:3] == ["note", "add"]


async def test_no_mirror_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    async def fake(command: list[str], *args: object, **kwargs: object) -> None:
        commands.append(command)

    monkeypatch.setattr(anyio, "run_process", fake)
    journal = Journal.open(tmp_path / "journal.jsonl")
    await journal.append(make_row(0, 0.5, Verdict.KEEP))
    assert commands == []
