from __future__ import annotations

import json
import os
import struct
import time
from typing import TYPE_CHECKING

import anyio
import pytest
from click.testing import CliRunner

from athome.cache import (
    ACTIVE_STAGING,
    HEARTBEAT_SUFFIX,
    STALE_TMP_SECONDS,
    Cache,
    CacheKey,
    CacheKeyError,
    CacheStats,
    atomic_write_bytes,
    atomic_write_text,
    cached,
    cli,
    stats_all,
)
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


async def test_write_cleans_up_eagerly_on_cancellation() -> None:
    cache = Cache.open("cancelled", version=1)
    key = cache.key("interrupted")
    with anyio.move_on_after(0.05) as scope:
        async with cache.write(key) as staging:
            await staging.mkdir()
            await (staging / "part").write_bytes(b"partial")
            await anyio.sleep(30)

    assert scope.cancelled_caught
    assert await cache.get(key) is None
    shard = cache.root / key.digest[:2]
    residue = [entry.name async for entry in shard.iterdir()] if await shard.exists() else []
    assert residue == []
    assert ACTIVE_STAGING == set()


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
    old = time.time() - 2 * STALE_TMP_SECONDS
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


@pytest.mark.parametrize(
    "namespace",
    ["../../x", "..", ".", "a/b"],
    ids=["traversal", "dotdot", "dot", "slash"],
)
def test_open_rejects_bad_namespace(namespace: str) -> None:
    with pytest.raises(CacheKeyError):
        Cache.open(namespace, version=1)


async def test_forged_digest_key_rejected(tmp_path: Path) -> None:
    cache = Cache.open("blobs", version=1)
    escape = tmp_path / "escape-target"
    for forged in (CacheKey(digest=str(escape)), CacheKey(digest="/etc/x"), CacheKey(digest="..")):
        with pytest.raises(CacheKeyError):
            cache.entry_path(forged)
        with pytest.raises(CacheKeyError):
            await cache.get(forged)
        with pytest.raises(CacheKeyError):
            await cache.put_bytes(forged, b"pwn")
    assert not escape.exists()
    real = cache.entry_path(cache.key("legit"))
    assert load(AthomeSettings).cache_root in real.parents


async def test_open_sweep_skips_active_staging_dir() -> None:
    cache = Cache.open("live", version=1)
    key = cache.key("longwrite")
    async with cache.write(key) as staging:
        await staging.mkdir()
        await (staging / "part").write_bytes(b"partial")
        old = time.time() - 2 * STALE_TMP_SECONDS
        os.utime(str(staging), (old, old))
        Cache.open("live", version=1)
        assert await staging.exists()
        assert await (staging / "part").exists()
    entry = await cache.get(key)
    assert entry is not None
    assert await (entry / "part").read_bytes() == b"partial"


async def test_sweep_skips_staging_with_recent_heartbeat_from_other_process() -> None:
    Cache.open("crossproc", version=1)
    shard = version_root("crossproc", 1) / "ab"
    shard.mkdir(parents=True, exist_ok=True)
    staging = shard / "deadbeef.tmp-otherproc"
    staging.mkdir()
    (staging / "part").write_bytes(b"in-flight")
    marker = shard / f"{staging.name}{HEARTBEAT_SUFFIX}"
    marker.write_bytes(b"")
    old = time.time() - 2 * STALE_TMP_SECONDS
    os.utime(staging, (old, old))

    Cache.open("crossproc", version=1)

    assert staging.exists()
    assert (staging / "part").read_bytes() == b"in-flight"
    assert marker.exists()


async def test_key_distinguishes_float_bit_patterns() -> None:
    cache = Cache.open("floats", version=1)
    nan1 = struct.unpack(">d", b"\x7f\xf8\x00\x00\x00\x00\x00\x01")[0]
    nan2 = struct.unpack(">d", b"\x7f\xf8\x00\x00\x00\x00\x00\x02")[0]
    assert cache.key(nan1) != cache.key(nan2)
    assert cache.key(0.0) != cache.key(-0.0)
    assert cache.key(nan1) == cache.key(struct.unpack(">d", b"\x7f\xf8\x00\x00\x00\x00\x00\x01")[0])
    assert cache.key(1.5) == cache.key(1.5)


def test_cli_stats_json() -> None:
    async def seed() -> None:
        cache = Cache.open("clins", version=1)
        await cache.put_bytes(cache.key("only"), b"1234")

    anyio.run(seed)
    result = CliRunner().invoke(cli, ["stats", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == [{"namespace": "clins", "entries": 1, "bytes": 4}]


async def test_atomic_write_bytes_round_trip(tmp_path: Path) -> None:
    target = anyio.Path(tmp_path) / "meta.bin"
    await atomic_write_bytes(target, b"payload")
    assert await target.read_bytes() == b"payload"


async def test_atomic_write_text_round_trip(tmp_path: Path) -> None:
    target = anyio.Path(tmp_path) / "meta.json"
    await atomic_write_text(target, "hello ünïcode")
    assert await target.read_text() == "hello ünïcode"


async def test_atomic_write_creates_parent_dirs(tmp_path: Path) -> None:
    target = anyio.Path(tmp_path) / "a" / "b" / "c" / "file"
    await atomic_write_bytes(target, b"deep")
    assert await target.read_bytes() == b"deep"
    assert await target.parent.is_dir()


async def test_atomic_write_overwrites_existing(tmp_path: Path) -> None:
    target = anyio.Path(tmp_path) / "f"
    await atomic_write_text(target, "first")
    await atomic_write_text(target, "second")
    assert await target.read_text() == "second"
    leftovers = [entry.name async for entry in anyio.Path(tmp_path).iterdir() if ".tmp-" in entry.name]
    assert leftovers == []


async def test_atomic_write_crash_before_rename_keeps_original_and_no_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = anyio.Path(tmp_path) / "meta.json"
    await atomic_write_text(target, "original")

    async def boom(staging: object, final: object) -> None:
        raise RuntimeError("crash before rename")

    monkeypatch.setattr("athome.cache.publish", boom)
    with pytest.raises(RuntimeError, match="crash before rename"):
        await atomic_write_text(target, "replacement")
    assert await target.read_text() == "original"
    leftovers = [entry.name async for entry in anyio.Path(tmp_path).iterdir() if ".tmp-" in entry.name]
    assert leftovers == []


async def test_open_with_explicit_root_writes_under_it(tmp_path: Path) -> None:
    cache = Cache.open("blobs", version=1, root=tmp_path / "cd")
    path = await cache.put_bytes(cache.key("x"), b"data")
    assert str(path).startswith(str(tmp_path / "cd"))
    assert not str(path).startswith(str(load(AthomeSettings).cache_root))
    assert await cache.get_bytes(cache.key("x")) == b"data"


async def test_open_explicit_roots_are_isolated(tmp_path: Path) -> None:
    a = Cache.open("ns", version=1, root=tmp_path / "a")
    b = Cache.open("ns", version=1, root=tmp_path / "b")
    await a.put_bytes(a.key("k"), b"in-a")
    assert a.root != b.root
    assert await b.get_bytes(b.key("k")) is None
    assert await a.get_bytes(a.key("k")) == b"in-a"


async def test_open_default_root_uses_settings(tmp_path: Path) -> None:
    cache = Cache.open("defaultns", version=1)
    path = await cache.put_bytes(cache.key("x"), b"data")
    assert str(path).startswith(str(load(AthomeSettings).cache_root))
