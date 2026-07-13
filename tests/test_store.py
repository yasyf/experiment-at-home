from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from athome.store import Store

if TYPE_CHECKING:
    from pathlib import Path

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
