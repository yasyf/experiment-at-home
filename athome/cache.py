from __future__ import annotations

import errno
import functools
import os
import pickle
import shutil
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass
from hashlib import blake2b
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import click
from anyio import Path, to_thread

from athome.cli import coro, emit, json_option
from athome.config import AthomeSettings, load
from athome.errors import AthomeError

if TYPE_CHECKING:
    import pathlib
    from collections.abc import AsyncIterator

STALE_TMP_SECONDS = 86400.0
PERSON = b"athome-cache"
TYPE_TAGS: dict[type, bytes] = {bool: b"?:", int: b"i:", float: b"f:", str: b"s:", bytes: b"b:"}


class CacheKeyError(AthomeError):
    """Raised when a cached call receives an argument that is not a hashable primitive or tuple."""


def frame(chunk: bytes) -> bytes:
    return len(chunk).to_bytes(8, "big") + chunk


def encode_part(part: bytes | str | int | float | bool | None | tuple[object, ...]) -> bytes:
    match part:
        case bool() | int() | float():
            return TYPE_TAGS[type(part)] + repr(part).encode()
        case str():
            return b"s:" + part.encode()
        case bytes():
            return b"b:" + part
        case None:
            return b"n:"
        case tuple():
            return b"t:" + b"".join(frame(encode_part(item)) for item in part)
        case _:
            raise CacheKeyError(f"unhashable cache key argument of type {type(part).__name__!r}")


def canonical(args: tuple[object, ...], kwargs: dict[str, object]) -> bytes:
    return encode_part((args, tuple(sorted(kwargs.items()))))


def remove_path(path: os.PathLike[str] | str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        os.unlink(path)


def sweep_stale_tmps(root: pathlib.Path) -> None:
    cutoff = time.time() - STALE_TMP_SECONDS
    for tmp in root.glob("*/*.tmp-*"):
        with suppress(FileNotFoundError):
            if tmp.stat().st_mtime < cutoff:
                remove_path(tmp)


async def discard(path: Path) -> None:
    await to_thread.run_sync(remove_path, path)


async def publish(staging: Path, final: Path) -> None:
    try:
        os.replace(staging, final)
    except OSError as error:
        if error.errno not in (errno.ENOTEMPTY, errno.EEXIST):
            raise
        await discard(staging)


async def entry_size(path: Path) -> int:
    if await path.is_file():
        return (await path.stat()).st_size
    return sum([await entry_size(child) async for child in path.iterdir()])


async def count_tree(version_root: Path) -> tuple[int, int]:
    entries = [
        entry
        async for shard in version_root.iterdir()
        if await shard.is_dir()
        async for entry in shard.iterdir()
        if ".tmp-" not in entry.name
    ]
    return len(entries), sum([await entry_size(entry) for entry in entries])


@dataclass(frozen=True, slots=True)
class CacheKey:
    """A content-derived cache address (blake2b-128 hex digest)."""

    digest: str


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Entry and byte totals for one cache namespace."""

    namespace: str
    entries: int
    bytes: int


@dataclass(frozen=True, slots=True)
class Cache:
    """A namespaced, versioned, content-keyed blob/directory cache under ``cache_root``.

    Entries live at ``<cache_root>/<namespace>/v<version>/<digest[:2]>/<digest>`` — the version is
    part of the path, so bumping it is the only invalidation. Every write stages a temp sibling and
    publishes it with a single atomic :func:`os.replace`, so a reader never observes a partial entry.

    Example:
        >>> cache = Cache.open("frames", version=1)
        >>> path = await cache.put_bytes(cache.key("clip", 7), jpeg)
    """

    namespace: str
    version: int
    root: Path

    @classmethod
    def open(cls, namespace: str, *, version: int) -> Cache:
        """Open (creating on first use) the cache tree for ``namespace`` at ``version``, sweeping stale temps."""
        base = load(AthomeSettings).cache_root / namespace / f"v{version}"
        base.mkdir(parents=True, exist_ok=True)
        sweep_stale_tmps(base)
        return cls(namespace, version, Path(base))

    def key(self, *parts: bytes | str | int | float | bool | None) -> CacheKey:
        """Derive a content address from a type-tagged, length-framed encoding of ``parts``."""
        material = b"".join(frame(encode_part(part)) for part in parts)
        return CacheKey(blake2b(material, digest_size=16, person=PERSON).hexdigest())

    def entry_path(self, key: CacheKey) -> Path:
        return self.root / key.digest[:2] / key.digest

    async def get(self, key: CacheKey) -> Path | None:
        """Return the published entry path (a file or a directory), or ``None`` on a miss."""
        return path if await (path := self.entry_path(key)).exists() else None

    async def get_bytes(self, key: CacheKey) -> bytes | None:
        """Read a published blob entry, or ``None`` on a miss."""
        return await path.read_bytes() if await (path := self.entry_path(key)).exists() else None

    async def put_bytes(self, key: CacheKey, data: bytes) -> Path:
        """Atomically store ``data`` under ``key`` and return the published entry path."""
        async with self.write(key) as staging:
            await staging.write_bytes(data)
        return self.entry_path(key)

    @asynccontextmanager
    async def write(self, key: CacheKey) -> AsyncIterator[Path]:
        """Yield a staging path; publish it atomically on clean exit, discard it on error.

        Write a single file at the yielded path for a blob entry, or ``mkdir`` it and fill it for a
        directory entry. Incremental producers append to the staging file across the block.
        """
        final = self.entry_path(key)
        await final.parent.mkdir(parents=True, exist_ok=True)
        staging = final.parent / f"{final.name}.tmp-{uuid4().hex}"
        try:
            yield staging
        except BaseException:
            await discard(staging)
            raise
        await publish(staging, final)

    async def stats(self) -> CacheStats:
        """Count the published entries and total bytes in this cache's version tree."""
        return CacheStats(self.namespace, *await count_tree(self.root))


async def namespace_stats(namespace_dir: Path) -> CacheStats:
    counts = [await count_tree(version) async for version in namespace_dir.iterdir() if await version.is_dir()]
    return CacheStats(namespace_dir.name, sum(entries for entries, _ in counts), sum(size for _, size in counts))


async def stats_all() -> list[CacheStats]:
    """Aggregate entry and byte totals for every namespace under ``cache_root`` (across all versions)."""
    root = Path(load(AthomeSettings).cache_root)
    if not await root.exists():
        return []
    return [await namespace_stats(ns) async for ns in root.iterdir() if await ns.is_dir()]


def cached[F: Callable[..., Awaitable[object]]](*, ns: str, version: int) -> Callable[[F], F]:
    """Decorate an async function to memoize its pickled result in ``Cache.open(ns, version=version)``.

    The key is the function's qualified name plus a canonical encoding of the call arguments; arguments
    must be hashable primitives or tuples thereof, or :class:`CacheKeyError` is raised.

    Example:
        >>> @cached(ns="ocr", version=2)
        ... async def read(page: int) -> str: ...
    """

    def decorate(func: F) -> F:
        cache, qualname = Cache.open(ns, version=version), func.__qualname__

        @functools.wraps(func)
        async def wrapper(*args: object, **kwargs: object) -> object:
            key = cache.key(qualname, canonical(args, kwargs))
            if (hit := await cache.get_bytes(key)) is not None:
                return pickle.loads(hit)
            await cache.put_bytes(key, pickle.dumps(result := await func(*args, **kwargs)))
            return result

        return cast("F", wrapper)

    return decorate


@click.group("cache")
def cli() -> None:
    """Inspect the on-disk athome cache."""


@cli.command("stats")
@json_option
@coro
async def cache_stats(*, as_json: bool) -> None:
    """Print entry and byte totals for every cache namespace."""
    emit([asdict(stat) for stat in await stats_all()], as_json=as_json)
