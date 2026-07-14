from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import aiosqlite
import anyio

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from pathlib import Path

LOCKED_RETRY_ATTEMPTS = 4
LOCKED_RETRY_BASE_DELAY = 0.1
LOCKED_RETRY_MAX_DELAY = 2.0


@dataclass(slots=True)
class Store:
    """An open aiosqlite database with WAL, busy_timeout, and a schema applied.

    Single-writer: a per-``Store`` ``anyio.Lock`` serializes ``execute`` so a write and
    its commit land as one unit and no other task's commit flushes a partial write.
    Beyond the connection ``busy_timeout``, ``execute`` retries a bounded number of times
    on ``database is locked`` before propagating.
    """

    db: aiosqlite.Connection
    lock: anyio.Lock = field(default_factory=anyio.Lock)

    @classmethod
    @asynccontextmanager
    async def open(cls, path: Path, *, schema: str, busy_timeout_ms: int = 5000) -> AsyncIterator[Store]:
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA synchronous=NORMAL")
            await db.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
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
        async with self.lock:
            for attempt in range(LOCKED_RETRY_ATTEMPTS - 1):
                try:
                    await self.db.execute(sql, params)
                    await self.db.commit()
                    return
                except sqlite3.OperationalError as error:
                    if "locked" not in str(error):
                        raise
                    await anyio.sleep(min(LOCKED_RETRY_MAX_DELAY, LOCKED_RETRY_BASE_DELAY * 2.0**attempt))
            await self.db.execute(sql, params)
            await self.db.commit()
