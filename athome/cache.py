from __future__ import annotations

import errno
import functools
import os
import pickle
import re
import shutil
import struct
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass
from hashlib import blake2b
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import click
from anyio import CancelScope, Path, to_thread

from athome.cli import coro, emit, json_option
from athome.config import AthomeSettings, load
from athome.errors import AthomeError

if TYPE_CHECKING:
    import pathlib
    from collections.abc import AsyncIterator

STALE_TMP_SECONDS = 604800.0
HEARTBEAT_SUFFIX = ".heartbeat"
PERSON = b"athome-cache"
DIGEST_BYTES = 16
TYPE_TAGS: dict[type, bytes] = {bool: b"?:", int: b"i:", float: b"f:", str: b"s:", bytes: b"b:"}
NAMESPACE_RE = re.compile(r"[A-Za-z0-9._-]+")
DIGEST_RE = re.compile(rf"[0-9a-f]{{{DIGEST_BYTES * 2}}}")
ACTIVE_STAGING: set[str] = set()


class CacheKeyError(AthomeError):
    """Raised for an invalid cache address: a non-primitive argument, a bad namespace, or a malformed digest."""


def frame(chunk: bytes) -> bytes:
    return len(chunk).to_bytes(8, "big") + chunk


def encode_part(part: bytes | str | int | float | bool | None | tuple[object, ...]) -> bytes:
    match part:
        case float():
            return TYPE_TAGS[float] + struct.pack(">d", part)
        case bool() | int():
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


def write_fsync(path: str, data: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def heartbeat_fresh(staging: pathlib.Path, cutoff: float) -> bool:
    with suppress(FileNotFoundError):
        return staging.with_name(staging.name + HEARTBEAT_SUFFIX).stat().st_mtime >= cutoff
    return False


def reapable(tmp: pathlib.Path, cutoff: float) -> bool:
    return tmp.stat().st_mtime < cutoff and (tmp.name.endswith(HEARTBEAT_SUFFIX) or not heartbeat_fresh(tmp, cutoff))


def sweep_stale_tmps(root: pathlib.Path) -> None:
    # No cross-process locking: the threshold sits well above any real write duration and write()
    # drops a heartbeat marker at start, so another process's sweep never reaps an in-flight write.
    # A write outliving STALE_TMP_SECONDS remains a residual (unrefreshed) race.
    cutoff = time.time() - STALE_TMP_SECONDS
    for tmp in root.glob("*/*.tmp-*"):
        if str(tmp) not in ACTIVE_STAGING:
            with suppress(FileNotFoundError):
                if reapable(tmp, cutoff):
                    remove_path(tmp)


async def discard(path: Path) -> None:
    # Shielded: cleanup runs on the cancellation path, where an unshielded await re-raises immediately
    # and strands the staging path for the stale-temp sweep (or forever, for temps outside a cache tree).
    with CancelScope(shield=True):
        await to_thread.run_sync(remove_path, path)


async def publish(staging: Path, final: Path) -> None:
    try:
        os.replace(staging, final)
    except OSError as error:
        if error.errno not in (errno.ENOTEMPTY, errno.EEXIST):
            raise
        await discard(staging)


async def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically write ``data`` to ``path`` via a temp sibling, ``fsync``, and :func:`os.replace`.

    Parent directories are created as needed. A reader never observes a partial write, and a crash
    before the rename leaves any existing file at ``path`` intact. For standalone files at a
    caller-chosen path (transcripts, labels, ``meta.json``); use :class:`Cache` for content-keyed entries.

    Example:
        >>> await atomic_write_text(root / "meta.json", json.dumps(meta))
    """
    await path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / f"{path.name}.tmp-{uuid4().hex}"
    try:
        await to_thread.run_sync(write_fsync, str(staging), data)
        await publish(staging, path)
    except BaseException:
        await discard(staging)
        raise


async def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write ``text`` as UTF-8 to ``path``; see :func:`atomic_write_bytes`."""
    await atomic_write_bytes(path, text.encode())


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
    def open(cls, namespace: str, *, version: int, root: pathlib.Path | None = None) -> Cache:
        """Open (creating on first use) the cache tree for ``namespace`` at ``version``, sweeping stale temps.

        Pass ``root`` to override the settings ``cache_root`` for this cache only (a CLI ``--cache-dir``
        flag, a throwaway test directory) without mutating the process-wide settings singleton.
        """
        if not NAMESPACE_RE.fullmatch(namespace) or namespace in {".", ".."}:
            raise CacheKeyError(f"invalid cache namespace {namespace!r}")
        base = (root or load(AthomeSettings).cache_root) / namespace / f"v{version}"
        base.mkdir(parents=True, exist_ok=True)
        sweep_stale_tmps(base)
        return cls(namespace, version, Path(base))

    def key(self, *parts: bytes | str | int | float | bool | None) -> CacheKey:
        """Derive a content address from a type-tagged, length-framed encoding of ``parts``."""
        material = b"".join(frame(encode_part(part)) for part in parts)
        return CacheKey(blake2b(material, digest_size=DIGEST_BYTES, person=PERSON).hexdigest())

    def entry_path(self, key: CacheKey) -> Path:
        if not DIGEST_RE.fullmatch(key.digest):
            raise CacheKeyError(f"invalid cache digest {key.digest!r}")
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
        marker = final.parent / f"{staging.name}{HEARTBEAT_SUFFIX}"
        # Registered until published so a same-process sweep never reaps an in-flight write; the
        # marker's mtime carries that liveness to another process's sweep.
        ACTIVE_STAGING.add(str(staging))
        try:
            await marker.write_bytes(b"")
            try:
                yield staging
            except BaseException:
                await discard(staging)
                raise
            await publish(staging, final)
        finally:
            ACTIVE_STAGING.discard(str(staging))
            await discard(marker)

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
        # Keyed on __qualname__: safe for module-level functions; two closures sharing a qualname collide.
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
