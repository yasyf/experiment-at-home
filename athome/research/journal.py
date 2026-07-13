from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import anyio

from athome.progress import RunSink, load_journal

if TYPE_CHECKING:
    from collections.abc import Mapping

CC_NOTES_BIN = Path("/opt/homebrew/bin/cc-notes")
CC_NOTES_LABEL = "athome-research"


class Verdict(StrEnum):
    KEEP = "keep"
    DISCARD = "discard"
    CRASH = "crash"


@dataclass(frozen=True, slots=True)
class JournalRow:
    """One journaled unit outcome: the verdict, the metric, and the resources it cost.

    Attributes:
        unit: The work-unit index.
        commit: The candidate commit the row describes.
        metric: The structured-channel metric, or ``None`` on a crash.
        verdict: Whether the candidate was kept, discarded, or crashed.
        resources: What the iteration consumed (``wall_s``, ``peak_rss_mb``, ``usd``).
        description: The driver's one-line description of the proposal.
    """

    unit: int
    commit: str
    metric: float | None
    verdict: Verdict
    resources: dict[str, float]
    description: str

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> JournalRow:
        return cls(
            unit=record["unit"],
            commit=record["commit"],
            metric=record["metric"],
            verdict=Verdict(record["verdict"]),
            resources=record["resources"],
            description=record["description"],
        )

    def to_record(self) -> dict[str, object]:
        return {
            "unit": self.unit,
            "commit": self.commit,
            "metric": self.metric,
            "verdict": self.verdict.value,
            "resources": self.resources,
            "description": self.description,
        }


@dataclass(slots=True)
class Journal:
    """Typed append-only JSONL journal on ``progress.RunSink``; the row is durable before it acts.

    Each :class:`JournalRow` is written to the crash-safe JSONL sink before any
    mirror runs and before the loop commits or resets the candidate, so a crash
    never loses an attempted unit (the lab's journal-before-result rule). With
    ``mirror_cc_notes`` the row is also recorded to the installed ``cc-notes``
    binary (``/opt/homebrew/bin/cc-notes``, never ``uvx``) as a browsable note.

    Example:
        >>> journal = Journal.open(Path("~/.athome/research/exp.jsonl"))
        >>> await journal.append(row)
    """

    sink: RunSink
    mirror_cc_notes: bool
    _rows: list[JournalRow]

    @classmethod
    def open(cls, path: Path, *, mirror_cc_notes: bool = False) -> Journal:
        return cls(RunSink.open(path), mirror_cc_notes, [JournalRow.from_record(rec) for rec in load_journal(path)])

    async def append(self, row: JournalRow) -> None:
        await self.sink.append(row.to_record())
        self._rows.append(row)
        if self.mirror_cc_notes:
            await self._mirror(row)

    def rows(self) -> list[JournalRow]:
        return list(self._rows)

    def resume_unit(self) -> int:
        return max((row.unit for row in self._rows), default=-1) + 1

    def best(self, direction: Literal["min", "max"]) -> JournalRow | None:
        if not (kept := [row for row in self._rows if row.verdict is Verdict.KEEP and row.metric is not None]):
            return None
        return (min if direction == "min" else max)(kept, key=lambda row: row.metric)

    async def _mirror(self, row: JournalRow) -> None:
        await anyio.run_process(
            [
                str(CC_NOTES_BIN),
                "note",
                "add",
                f"athome unit {row.unit} [{row.verdict.value}]",
                "--body",
                f"commit {row.commit}\nmetric {row.metric}\nresources {row.resources}\n{row.description}",
                "--label",
                CC_NOTES_LABEL,
            ]
        )
