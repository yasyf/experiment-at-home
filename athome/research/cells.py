"""athome.store-backed sqlite cell index: one row per (experiment, unit, arm)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from athome.research.common import canonical_json
from athome.store import Store

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from pathlib import Path

CELLS_SCHEMA = """
CREATE TABLE IF NOT EXISTS cells (
    experiment TEXT NOT NULL,
    unit       INTEGER NOT NULL,
    arm        TEXT NOT NULL,
    metric     REAL,
    verdict    TEXT NOT NULL,
    payload    TEXT NOT NULL,
    PRIMARY KEY (experiment, unit, arm)
) WITHOUT ROWID;
"""


@dataclass(frozen=True, slots=True)
class Cell:
    """One (experiment, unit, arm) result: its scalar ``metric`` and keep/discard ``verdict``.

    Example:
        >>> Cell(experiment="e1", unit=0, arm="candidate", metric=0.9, verdict="keep")
    """

    experiment: str
    unit: int
    arm: str
    metric: float | None
    verdict: str


@dataclass(slots=True)
class CellIndex:
    """Rebuildable sqlite index over experiment cells with leaderboard queries.

    Inserts upsert on the (experiment, unit, arm) primary key, so re-running a unit
    overwrites its cell rather than duplicating it.

    Example:
        >>> async with CellIndex.open(path) as index:
        ...     await index.insert(cell)
        ...     board = await index.leaderboard("e1", direction="max")
    """

    store: Store

    @classmethod
    @asynccontextmanager
    async def open(cls, path: Path) -> AsyncIterator[CellIndex]:
        async with Store.open(path, schema=CELLS_SCHEMA) as store:
            yield cls(store)

    async def insert(self, cell: Cell, *, payload: Mapping[str, object] | None = None) -> None:
        await self.store.execute(
            "INSERT OR REPLACE INTO cells(experiment, unit, arm, metric, verdict, payload) VALUES (?, ?, ?, ?, ?, ?)",
            (cell.experiment, cell.unit, cell.arm, cell.metric, cell.verdict, canonical_json(payload or {}).decode()),
        )

    async def leaderboard(self, experiment: str, *, direction: Literal["min", "max"], limit: int = 10) -> list[Cell]:
        rows = await self.store.fetch_all(
            "SELECT experiment, unit, arm, metric, verdict FROM cells "
            "WHERE experiment = ? AND metric IS NOT NULL "
            f"ORDER BY metric {'ASC' if direction == 'min' else 'DESC'}, unit ASC LIMIT ?",
            (experiment, limit),
        )
        return [Cell(row["experiment"], row["unit"], row["arm"], row["metric"], row["verdict"]) for row in rows]

    async def best(self, experiment: str, *, direction: Literal["min", "max"]) -> Cell | None:
        board = await self.leaderboard(experiment, direction=direction, limit=1)
        return board[0] if board else None
