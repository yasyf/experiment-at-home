from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import aiosqlite
import anyio
import pytest

from athome.store import Store

if TYPE_CHECKING:
    from pathlib import Path

    from athome.store import Synchronous

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
"""


async def test_open_creates_parent_and_applies_schema(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "db.sqlite"
    async with Store.open(path, schema=SCHEMA) as store:
        assert path.exists()
        await store.execute("INSERT INTO items (name) VALUES (?)", ["alpha"])
        row = await store.fetch_one("SELECT name FROM items WHERE id = ?", [1])
        assert row is not None
        assert row["name"] == "alpha"


async def test_pragmas_applied(tmp_path: Path) -> None:
    async with Store.open(tmp_path / "db.sqlite", schema=SCHEMA) as store:
        journal = await store.fetch_one("PRAGMA journal_mode")
        assert journal is not None
        assert journal[0] == "wal"
        foreign_keys = await store.fetch_one("PRAGMA foreign_keys")
        assert foreign_keys is not None
        assert foreign_keys[0] == 1


async def test_fetch_all_and_miss(tmp_path: Path) -> None:
    async with Store.open(tmp_path / "db.sqlite", schema=SCHEMA) as store:
        await store.execute("INSERT INTO items (name) VALUES (?)", ["a"])
        await store.execute("INSERT INTO items (name) VALUES (?)", ["b"])
        rows = await store.fetch_all("SELECT name FROM items ORDER BY name")
        assert [row["name"] for row in rows] == ["a", "b"]
        assert await store.fetch_one("SELECT name FROM items WHERE name = ?", ["missing"]) is None


async def test_schema_idempotent_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite"
    async with Store.open(path, schema=SCHEMA) as store:
        await store.execute("INSERT INTO items (name) VALUES (?)", ["persist"])
    async with Store.open(path, schema=SCHEMA) as store:
        rows = await store.fetch_all("SELECT name FROM items")
        assert [row["name"] for row in rows] == ["persist"]


async def test_foreign_keys_enforced(tmp_path: Path) -> None:
    schema = """
    CREATE TABLE IF NOT EXISTS parent (id INTEGER PRIMARY KEY);
    CREATE TABLE IF NOT EXISTS child (
        id INTEGER PRIMARY KEY,
        parent_id INTEGER NOT NULL REFERENCES parent(id)
    );
    """
    async with Store.open(tmp_path / "fk.sqlite", schema=schema) as store:
        with pytest.raises(aiosqlite.IntegrityError):
            await store.execute("INSERT INTO child (parent_id) VALUES (?)", [999])


async def test_execute_serializes_write_and_commit(tmp_path: Path) -> None:
    async with Store.open(tmp_path / "db.sqlite", schema=SCHEMA) as store:
        async with store.lock:
            with anyio.CancelScope() as scope, anyio.fail_after(1):
                async with anyio.create_task_group() as tg:

                    async def blocked_writer() -> None:
                        await store.execute("INSERT INTO items (name) VALUES (?)", ["leaked"])

                    tg.start_soon(blocked_writer)
                    await anyio.sleep(0.05)
                    scope.cancel()
        assert await store.fetch_one("SELECT name FROM items WHERE name = ?", ["leaked"]) is None
        await store.execute("INSERT INTO items (name) VALUES (?)", ["ok"])
        row = await store.fetch_one("SELECT name FROM items WHERE name = ?", ["ok"])
        assert row is not None and row["name"] == "ok"


async def test_busy_timeout_default(tmp_path: Path) -> None:
    async with Store.open(tmp_path / "db.sqlite", schema=SCHEMA) as store:
        row = await store.fetch_one("PRAGMA busy_timeout")
        assert row is not None
        assert row[0] == 5000


async def test_busy_timeout_override_reflected(tmp_path: Path) -> None:
    async with Store.open(tmp_path / "db.sqlite", schema=SCHEMA, busy_timeout_ms=30000) as store:
        row = await store.fetch_one("PRAGMA busy_timeout")
        assert row is not None
        assert row[0] == 30000


@pytest.mark.parametrize(
    ("synchronous", "level"),
    [("NORMAL", 1), ("FULL", 2)],
    ids=["normal", "full"],
)
async def test_synchronous_pragma_applied(tmp_path: Path, synchronous: Synchronous, level: int) -> None:
    async with Store.open(tmp_path / "db.sqlite", schema=SCHEMA, synchronous=synchronous) as store:
        row = await store.fetch_one("PRAGMA synchronous")
        assert row is not None
        assert row[0] == level


async def test_synchronous_defaults_to_normal(tmp_path: Path) -> None:
    async with Store.open(tmp_path / "db.sqlite", schema=SCHEMA) as store:
        row = await store.fetch_one("PRAGMA synchronous")
        assert row is not None
        assert row[0] == 1


async def test_busy_timeout_armed_before_wal_flip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executed: list[str] = []
    execute = aiosqlite.Connection.execute

    async def record(self: aiosqlite.Connection, sql: str, *args: object, **kwargs: object) -> aiosqlite.Cursor:
        executed.append(sql)
        return await execute(self, sql, *args, **kwargs)

    monkeypatch.setattr(aiosqlite.Connection, "execute", record)
    async with Store.open(tmp_path / "db.sqlite", schema=SCHEMA, busy_timeout_ms=7000):
        pass

    pragmas = [sql for sql in executed if sql.startswith("PRAGMA")]
    assert pragmas.index("PRAGMA busy_timeout=7000") < pragmas.index("PRAGMA journal_mode=WAL")


async def test_execute_retries_until_lock_released(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("athome.store.LOCKED_RETRY_ATTEMPTS", 100)
    monkeypatch.setattr("athome.store.LOCKED_RETRY_BASE_DELAY", 0.01)
    monkeypatch.setattr("athome.store.LOCKED_RETRY_MAX_DELAY", 0.02)
    db_path = tmp_path / "db.sqlite"
    async with Store.open(db_path, schema=SCHEMA, busy_timeout_ms=1) as store:
        async with aiosqlite.connect(db_path) as blocker:
            await blocker.execute("INSERT INTO items (name) VALUES (?)", ["holder"])
            async with anyio.create_task_group() as tg:

                async def release() -> None:
                    await anyio.sleep(0.1)
                    await blocker.rollback()

                tg.start_soon(release)
                await store.execute("INSERT INTO items (name) VALUES (?)", ["delayed"])
        row = await store.fetch_one("SELECT name FROM items WHERE name = ?", ["delayed"])
        assert row is not None and row["name"] == "delayed"


async def test_execute_raises_after_retry_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("athome.store.LOCKED_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr("athome.store.LOCKED_RETRY_BASE_DELAY", 0.01)
    monkeypatch.setattr("athome.store.LOCKED_RETRY_MAX_DELAY", 0.02)
    db_path = tmp_path / "db.sqlite"
    async with Store.open(db_path, schema=SCHEMA, busy_timeout_ms=1) as store:
        async with aiosqlite.connect(db_path) as blocker:
            await blocker.execute("INSERT INTO items (name) VALUES (?)", ["holder"])
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                await store.execute("INSERT INTO items (name) VALUES (?)", ["never"])
        assert await store.fetch_one("SELECT name FROM items WHERE name = ?", ["never"]) is None
