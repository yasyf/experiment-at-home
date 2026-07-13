from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from athome.errors import AthomeError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping
    from pathlib import Path

RESERVED_FIELDS = frozenset({"unit", "status"})


class FailureBudgetExceeded(AthomeError):
    """Raised when a ``RunSink``'s recorded failures exceed its budget."""


class PhaseMissing(AthomeError):
    """Raised by ``Phases.require`` when a required phase has not been marked."""


class ReservedFieldConflict(AthomeError):
    """Raised when ``WorkSet.done`` receives an extra field that the record owns."""


def parse_records(path: Path, lines: list[str]) -> Iterator[dict[str, object]]:
    for index, line in enumerate(lines):
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            logger.warning("skipping truncated final line in {}", path)


def load_journal(path: Path) -> list[dict[str, object]]:
    return list(parse_records(path, path.read_text().splitlines())) if path.exists() else []


async def append_line(path: Path, record: Mapping[str, object]) -> None:
    # Loop past short writes. Within the FS atomic-append size the record is one atomic O_APPEND
    # write; a larger record lands intact but not atomically against concurrent writers (contract:
    # single-writer resume).
    payload = (json.dumps(record) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
    finally:
        os.close(fd)


@dataclass(slots=True)
class WorkSet:
    """A resumable set of work units journaled as JSONL; errors are retried on resume.

    Each unit is journaled as it lands: a ``done`` record marks it complete, an
    ``error`` record leaves it pending so it is retried on the next ``open``.

    Example:
        >>> work = WorkSet.open(Path("~/.athome/run.jsonl"))
        >>> for unit in work.pending(["a", "b", "c"]):
        ...     await work.done(unit)
    """

    path: Path
    _done: set[str]

    @classmethod
    def open(cls, path: Path) -> WorkSet:
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(path, {str(rec["unit"]) for rec in load_journal(path) if rec.get("status") == "done"})

    def is_done(self, unit: str) -> bool:
        return unit in self._done

    def pending(self, units: Iterable[str]) -> list[str]:
        return [unit for unit in units if unit not in self._done]

    async def done(self, unit: str, **extra: object) -> None:
        if reserved := RESERVED_FIELDS & extra.keys():
            raise ReservedFieldConflict(f"reserved journal fields: {sorted(reserved)}")
        await append_line(self.path, {"unit": unit, "status": "done"} | extra)
        self._done.add(unit)

    async def error(self, unit: str, message: str) -> None:
        await append_line(self.path, {"unit": unit, "status": "error", "message": message})


@dataclass(slots=True)
class RunSink:
    """Crash-safe incremental JSONL sink with a failure budget.

    Every record is one appended JSON line; ``fail`` tags its record ``"failed": true``
    and raises ``FailureBudgetExceeded`` once the recorded failures exceed the budget.
    The budget survives a crash: ``open`` recounts failed records from the journal.

    Example:
        >>> sink = RunSink.open(Path("~/.athome/out.jsonl"), failure_budget=3)
        >>> await sink.append({"item": "a", "ok": True})
    """

    path: Path
    failure_budget: int
    _failures: int

    @classmethod
    def open(cls, path: Path, *, failure_budget: int = 0) -> RunSink:
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(path, failure_budget, sum(rec.get("failed") is True for rec in load_journal(path)))

    async def append(self, record: Mapping[str, object]) -> None:
        await append_line(self.path, record)

    async def fail(self, record: Mapping[str, object]) -> None:
        await append_line(self.path, record | {"failed": True})
        self._failures += 1
        if self._failures > self.failure_budget:
            raise FailureBudgetExceeded(f"{self._failures} failures exceed budget of {self.failure_budget}")

    @property
    def failures(self) -> int:
        return self._failures


@dataclass(slots=True)
class Phases:
    """Ordered phase markers gating multi-stage runs.

    ``mark`` journals a phase as reached; ``require`` raises ``PhaseMissing`` when a
    prerequisite has not been marked. Markers survive a crash via the journal.

    Example:
        >>> phases = Phases.open(Path("~/.athome/phases.jsonl"))
        >>> phases.require("extract")
    """

    path: Path
    _marked: set[str]

    @classmethod
    def open(cls, path: Path) -> Phases:
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(path, {str(rec["phase"]) for rec in load_journal(path)})

    async def mark(self, name: str) -> None:
        await append_line(self.path, {"phase": name})
        self._marked.add(name)

    def require(self, name: str) -> None:
        if name not in self._marked:
            raise PhaseMissing(name)

    def done(self, name: str) -> bool:
        return name in self._marked
