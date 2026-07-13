from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING

import anyio
import pytest
from click.testing import CliRunner

from athome.cache import Cache, CacheKeyError, CacheStats, cached, cli, stats_all
from athome.config import AthomeSettings, load

if TYPE_CHECKING:
    from pathlib import Path


def version_root(namespace: str, version: int) -> Path:
    return load(AthomeSettings).cache_root / namespace / f"v{version}"


async def test_put_get_bytes_round_trip() -> None:
    cache = Cache.open("blobs", version=1)
    key = cache.key("greeting", 42)
    assert await cache.get(key) is None
    assert await cache.get_bytes(key) is None
    path = await cache.put_bytes(key, b"hello world")
    assert await cache.get(key) == path
    assert await cache.get_bytes(key) == b"hello world"


async def test_entry_layout_shards_by_digest() -> None:
    cache = Cache.open("blobs", version=3)
    key = cache.key("x")
    path = await cache.put_bytes(key, b"data")
    assert path.name == key.digest
    assert path.parent.name == key.digest[:2]
    assert path.parent.parent == cache.root


async def test_directory_entry_via_write() -> None:
    cache = Cache.open("frames", version=1)
    key = cache.key("clip", 7)
    async with cache.write(key) as staging:
        await staging.mkdir()
        await (staging / "000.txt").write_bytes(b"a")
        await (staging / "001.txt").write_bytes(b"bb")
    entry = await cache.get(key)
    assert entry is not None
    assert await entry.is_dir()
    assert await (entry / "000.txt").read_bytes() == b"a"
    assert await (entry / "001.txt").read_bytes() == b"bb"


async def test_write_discards_staging_on_error() -> None:
    cache = Cache.open("blobs", version=1)
    key = cache.key("boom")
    with pytest.raises(RuntimeError, match="kaboom"):
        async with cache.write(key) as staging:
            await staging.write_bytes(b"partial")
            raise RuntimeError("kaboom")
    assert await cache.get(key) is None
    shard = cache.root / key.digest[:2]
    leftovers = [entry.name async for entry in shard.iterdir()] if await shard.exists() else []
    assert leftovers == []


async def test_version_bump_isolates_entries() -> None:
    v1 = Cache.open("model", version=1)
    v2 = Cache.open("model", version=2)
    material = ("prompt", 1, True)
    await v1.put_bytes(v1.key(*material), b"v1-value")
    assert v1.root != v2.root
    assert await v2.get_bytes(v2.key(*material)) is None
    assert await v1.get_bytes(v1.key(*material)) == b"v1-value"


async def test_key_is_type_tagged_and_collision_safe() -> None:
    cache = Cache.open("keys", version=1)
    assert cache.key("1") != cache.key(1)
    assert cache.key("ab", "c") != cache.key("a", "bc")
    assert cache.key("a", "b") != cache.key("a", None, "b")
    assert cache.key(1, 2) == cache.key(1, 2)


async def test_stale_tmp_swept_on_open() -> None:
    Cache.open("sweep", version=1)
    shard = version_root("sweep", 1) / "ab"
    shard.mkdir(parents=True, exist_ok=True)
    stale_file = shard / "deadbeef.tmp-oldfile"
    stale_file.write_bytes(b"orphan")
    stale_dir = shard / "cafef00d.tmp-olddir"
    stale_dir.mkdir()
    (stale_dir / "member").write_bytes(b"leftover")
    fresh = shard / "beadfeed.tmp-live"
    fresh.write_bytes(b"in-flight")
    old = time.time() - 2 * 86400
    os.utime(stale_file, (old, old))
    os.utime(stale_dir, (old, old))

    Cache.open("sweep", version=1)

    assert not stale_file.exists()
    assert not stale_dir.exists()
    assert fresh.exists()


async def test_cached_decorator_hit_and_miss() -> None:
    calls: list[tuple[int, int]] = []

    @cached(ns="memo", version=1)
    async def add(a: int, b: int) -> int:
        calls.append((a, b))
        return a + b

    assert await add(2, 3) == 5
    assert await add(2, 3) == 5
    assert calls == [(2, 3)]
    assert await add(4, 5) == 9
    assert calls == [(2, 3), (4, 5)]


async def test_cached_distinguishes_kwargs_and_qualname() -> None:
    @cached(ns="memo", version=1)
    async def scale(value: int, *, factor: int = 1) -> int:
        return value * factor

    @cached(ns="memo", version=1)
    async def shift(value: int, *, factor: int = 1) -> int:
        return value + factor

    assert await scale(3, factor=2) == 6
    assert await scale(3, factor=4) == 12
    assert await shift(3, factor=2) == 5


async def test_cached_raises_on_unhashable_args() -> None:
    @cached(ns="memo", version=1)
    async def summarize(items: object) -> int:
        return 1

    with pytest.raises(CacheKeyError):
        await summarize([1, 2, 3])
    with pytest.raises(CacheKeyError):
        await summarize({"a": 1})


async def test_stats_counts_entries_and_bytes() -> None:
    cache = Cache.open("statsns", version=1)
    await cache.put_bytes(cache.key("a"), b"12345")
    await cache.put_bytes(cache.key("b"), b"678")
    async with cache.write(cache.key("dir")) as staging:
        await staging.mkdir()
        await (staging / "f").write_bytes(b"xy")
    assert await cache.stats() == CacheStats("statsns", entries=3, bytes=10)


async def test_stats_all_aggregates_versions_per_namespace() -> None:
    v1 = Cache.open("agg", version=1)
    v2 = Cache.open("agg", version=2)
    other = Cache.open("other", version=1)
    await v1.put_bytes(v1.key("x"), b"aaaa")
    await v2.put_bytes(v2.key("y"), b"bb")
    await other.put_bytes(other.key("z"), b"c")
    by_namespace = {stat.namespace: stat for stat in await stats_all()}
    assert by_namespace["agg"] == CacheStats("agg", entries=2, bytes=6)
    assert by_namespace["other"] == CacheStats("other", entries=1, bytes=1)


async def test_stats_all_empty_when_cache_root_absent() -> None:
    assert await stats_all() == []


def test_cli_stats_json() -> None:
    async def seed() -> None:
        cache = Cache.open("clins", version=1)
        await cache.put_bytes(cache.key("only"), b"1234")

    anyio.run(seed)
    result = CliRunner().invoke(cli, ["stats", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == [{"namespace": "clins", "entries": 1, "bytes": 4}]
