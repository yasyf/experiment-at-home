from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import aiosqlite

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from pathlib import Path


@dataclass(slots=True)
class Store:
    """An open aiosqlite database with WAL, busy_timeout, and a schema applied."""

    db: aiosqlite.Connection

    @classmethod
    @asynccontextmanager
    async def open(cls, path: Path, *, schema: str) -> AsyncIterator[Store]:
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA synchronous=NORMAL")
            await db.execute("PRAGMA busy_timeout=5000")
            await db.execute("PRAGMA foreign_keys=ON")
            await db.executescript(schema)
            db.row_factory = aiosqlite.Row
            await db.commit()
            yield cls(db)

    async def fetch_one(self, sql: str, params: Sequence[object] = ()) -> aiosqlite.Row | None:
        async with self.db.execute(sql, params) as cur:
            return await cur.fetchone()

    async def fetch_all(self, sql: str, params: Sequence[object] = ()) -> list[aiosqlite.Row]:
        async with self.db.execute(sql, params) as cur:
            return list(await cur.fetchall())

    async def execute(self, sql: str, params: Sequence[object] = ()) -> None:
        await self.db.execute(sql, params)
        await self.db.commit()
