from __future__ import annotations

from typing import TYPE_CHECKING

from athome.research.cells import Cell, CellIndex

if TYPE_CHECKING:
    from pathlib import Path


async def test_insert_and_leaderboard_ranks_by_metric_desc(tmp_path: Path) -> None:
    async with CellIndex.open(tmp_path / "cells.db") as index:
        await index.insert(Cell("e1", 0, "cand", 0.4, "keep"))
        await index.insert(Cell("e1", 1, "cand", 0.9, "keep"))
        await index.insert(Cell("e1", 2, "cand", 0.6, "discard"))
        board = await index.leaderboard("e1", direction="max")
    assert [cell.metric for cell in board] == [0.9, 0.6, 0.4]
    assert [cell.unit for cell in board] == [1, 2, 0]


async def test_leaderboard_min_direction_ranks_ascending(tmp_path: Path) -> None:
    async with CellIndex.open(tmp_path / "cells.db") as index:
        await index.insert(Cell("e1", 0, "cand", 0.4, "keep"))
        await index.insert(Cell("e1", 1, "cand", 0.9, "keep"))
        board = await index.leaderboard("e1", direction="min")
    assert [cell.metric for cell in board] == [0.4, 0.9]


async def test_leaderboard_respects_limit(tmp_path: Path) -> None:
    async with CellIndex.open(tmp_path / "cells.db") as index:
        for unit in range(5):
            await index.insert(Cell("e1", unit, "cand", float(unit), "keep"))
        board = await index.leaderboard("e1", direction="max", limit=2)
    assert [cell.metric for cell in board] == [4.0, 3.0]


async def test_leaderboard_excludes_null_metric_cells(tmp_path: Path) -> None:
    async with CellIndex.open(tmp_path / "cells.db") as index:
        await index.insert(Cell("e1", 0, "cand", 0.5, "keep"))
        await index.insert(Cell("e1", 1, "cand", None, "crash"))
        board = await index.leaderboard("e1", direction="max")
    assert [cell.unit for cell in board] == [0]


async def test_insert_is_idempotent_on_the_primary_key(tmp_path: Path) -> None:
    async with CellIndex.open(tmp_path / "cells.db") as index:
        await index.insert(Cell("e1", 0, "cand", 0.4, "discard"))
        await index.insert(Cell("e1", 0, "cand", 0.8, "keep"))
        board = await index.leaderboard("e1", direction="max")
    assert len(board) == 1
    assert board[0].metric == 0.8
    assert board[0].verdict == "keep"


async def test_leaderboard_scopes_to_the_experiment(tmp_path: Path) -> None:
    async with CellIndex.open(tmp_path / "cells.db") as index:
        await index.insert(Cell("e1", 0, "cand", 0.4, "keep"))
        await index.insert(Cell("e2", 0, "cand", 0.9, "keep"))
        board = await index.leaderboard("e1", direction="max")
    assert [cell.experiment for cell in board] == ["e1"]


async def test_best_returns_the_top_cell_or_none(tmp_path: Path) -> None:
    async with CellIndex.open(tmp_path / "cells.db") as index:
        assert await index.best("e1", direction="max") is None
        await index.insert(Cell("e1", 0, "cand", 0.4, "keep"))
        await index.insert(Cell("e1", 1, "cand", 0.9, "keep"))
        best = await index.best("e1", direction="max")
    assert best is not None
    assert best.metric == 0.9


async def test_insert_persists_the_payload_json(tmp_path: Path) -> None:
    async with CellIndex.open(tmp_path / "cells.db") as index:
        await index.insert(Cell("e1", 0, "cand", 0.4, "keep"), payload={"wall_s": 1.5, "usd": 0.02})
        row = await index.store.fetch_one("SELECT payload FROM cells WHERE experiment = 'e1' AND unit = 0")
    assert row is not None
    assert row["payload"] == '{"usd":0.02,"wall_s":1.5}'
