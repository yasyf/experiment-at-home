"""Immutable, content-addressed artifact versions with an atomic ``current`` promotion."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import anyio
import anyio.to_thread

from athome.errors import ResearchError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

METADATA_NAME = "metadata.json"
CURRENT_LINK = "current"
STAGING_PREFIX = f".{CURRENT_LINK}.tmp."
VERSION_STAGING_PREFIX = ".version.tmp."
VERSION_PATTERN = re.compile(r"^v(\d{3,})-(\d{8})-([0-9a-f]{12})$")
DIGEST_CHARS = 12
LOCK_SUFFIX = ".lock"
LOCK_POLL_SECONDS = 0.01
READ_ONLY_MODE = 0o444
DIGEST_CHUNK_BYTES = 1 << 20


class RegistryError(ResearchError):
    """The registry cannot satisfy the request: an unknown version, or an empty registration.

    Subclasses :class:`~athome.errors.ResearchError` — the registry grew out of the research
    harness, and an ``except ResearchError`` around a registry call predates its promotion to
    :mod:`athome.registry`, so it must keep catching. Outside research, catch this or
    :class:`~athome.errors.AthomeError`.
    """


@dataclass(frozen=True, slots=True)
class VersionInfo:
    """One registered artifact version.

    Attributes:
        name: The artifact family (the experiment or model name).
        version: The version directory name, ``v<NNN>-<YYYYMMDD>-<digest12>``.
        path: The version directory holding the artifact files and metadata.
        metadata: The parsed ``metadata.json``.
    """

    name: str
    version: str
    path: Path
    metadata: dict[str, object]

    @property
    def number(self) -> int:
        match = VERSION_PATTERN.match(self.version)
        assert match is not None
        return int(match.group(1))


async def versions(name: str, *, root: Path) -> list[VersionInfo]:
    """Every registered version of an artifact family under ``root``, oldest first."""
    base = anyio.Path(root) / name
    if not await base.is_dir():
        return []
    found = [
        await _info(name, child)
        async for child in base.iterdir()
        if VERSION_PATTERN.match(child.name) and await child.is_dir() and not await child.is_symlink()
    ]
    return sorted(found, key=lambda info: info.number)


async def current(name: str, *, root: Path) -> VersionInfo | None:
    """The promoted version the ``current`` symlink names, or ``None`` when nothing is promoted."""
    link = anyio.Path(root) / name / CURRENT_LINK
    if not await link.is_symlink():
        return None
    return await _info(name, link.parent / await link.readlink())


async def register(
    name: str, files: Mapping[str, bytes | Path], metadata: Mapping[str, object], *, root: Path
) -> VersionInfo:
    """Writes a new immutable version directory under ``root``; never flips ``current``.

    The registry owns what it registers: every path source is *copied* into the version
    directory, so the registered artifact is the registry's own and no later writer to the
    source can mutate it. The copies land read-only. The directory name embeds the next
    version number, today's date, and a 12-hex content digest over the stored artifact files,
    and ``metadata.json`` is the caller's metadata stamped with ``name``, ``version``, and
    ``created_at``.

    The version is staged under a temporary name and renamed into place, so a reader never
    sees a half-copied version directory.

    Args:
        files: Artifact entry name to content — raw bytes, a file to copy, or a directory
            to copy as a tree (its files are digested under ``<entry>/<relative path>``).
        metadata: The version's provenance: dataset digest, config, metrics.
        root: The registry root the family lives under.

    Returns:
        The freshly registered :class:`VersionInfo`.
    """
    if not files:
        raise RegistryError(f"refusing to register an empty {name} version")
    family = anyio.Path(root) / name
    async with _family_lock(family):
        number = existing[-1].number + 1 if (existing := await versions(name, root=root)) else 1
        now = datetime.now(UTC)
        staging = family / f"{VERSION_STAGING_PREFIX}{uuid4().hex}"
        await staging.mkdir(parents=True)
        for filename, content in files.items():
            await _materialize(staging / filename, content)
        version = f"v{number:03d}-{now:%Y%m%d}-{await _digest(staging)}"
        stamped = dict(metadata) | {"name": name, "version": version, "created_at": now.isoformat()}
        await (staging / METADATA_NAME).write_text(json.dumps(stamped, indent=2, sort_keys=True, default=str) + "\n")
        path = family / version
        await staging.rename(path)
        await _freeze(path)
    return VersionInfo(name=name, version=version, path=Path(path), metadata=stamped)


async def promote(name: str, version: str, *, root: Path) -> None:
    """Atomically flips ``current`` to the named version (full name or ``v<NNN>`` prefix).

    The flip writes a staging symlink under a unique per-promotion name and renames
    it over ``current``, so a reader never sees a missing or half-written link. The
    family lock serialises it against :func:`register`, :func:`rollback`, :func:`prune`,
    and concurrent promotions, so no prune can delete the version being promoted.
    """
    async with _family_lock(anyio.Path(root) / name):
        await _promote(name, version, root=root)


async def components(root: Path) -> tuple[str, ...]:
    """Every artifact family with at least one registered version under ``root``, sorted.

    A family directory exists from the moment a registration stages its first version, so a
    registration that dies before its commit leaves the directory behind. Such a family has
    registered nothing and is not listed.
    """
    base = anyio.Path(root)
    if not await base.is_dir():
        return ()
    return tuple(
        sorted(
            [
                child.name
                async for child in base.iterdir()
                if await child.is_dir() and await versions(child.name, root=root)
            ]
        )
    )


async def rollback(name: str, *, root: Path) -> VersionInfo:
    """Repoints ``current`` to the version registered just before the current promotion.

    Serialised against :func:`register`, :func:`prune`, and concurrent rollbacks by the family lock,
    the repoint itself reusing :func:`promote`'s atomic symlink swap.

    Args:
        name: The artifact family to roll back.
        root: The registry root the family lives under.

    Returns:
        The now-promoted prior :class:`VersionInfo`.

    Raises:
        RegistryError: Nothing is promoted, or the current version is already the earliest.
    """
    async with _family_lock(anyio.Path(root) / name):
        if (promoted := await current(name, root=root)) is None:
            raise RegistryError(f"cannot roll back {name}: nothing is promoted")
        ordered = await versions(name, root=root)
        if (prior := next((info for info in reversed(ordered) if info.number < promoted.number), None)) is None:
            raise RegistryError(f"cannot roll back {name}: {promoted.version} is the earliest version")
        await _promote(name, prior.version, root=root)
    return prior


async def prune(name: str, *, keep: int = 3, root: Path) -> tuple[VersionInfo, ...]:
    """Deletes all but the newest ``keep`` versions of ``name``, never the one ``current`` points to.

    The promoted version is always retained, even when it falls outside the newest-``keep`` window.
    Doomed versions are renamed out of discovery and then removed under the family lock, so a prune
    never races a :func:`register`, :func:`promote`, or :func:`rollback` on the same family, and a
    reader never sees a half-deleted version.

    Args:
        name: The artifact family to prune.
        keep: How many of the newest versions to retain.
        root: The registry root the family lives under.

    Returns:
        The removed versions, oldest first.

    Raises:
        RegistryError: ``keep`` is negative.
    """
    if keep < 0:
        raise RegistryError(f"refusing to prune {name} with a negative keep of {keep}")
    async with _family_lock(anyio.Path(root) / name):
        ordered = await versions(name, root=root)
        promoted = await current(name, root=root)
        retained = {info.version for info in ordered[max(len(ordered) - keep, 0) :]}
        if promoted is not None:
            retained.add(promoted.version)
        doomed = [info for info in ordered if info.version not in retained]
        for info in doomed:
            await _discard(anyio.Path(info.path))
    return tuple(doomed)


@asynccontextmanager
async def _family_lock(family: anyio.Path) -> AsyncIterator[None]:
    await family.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(Path(family.parent) / f"{family.name}{LOCK_SUFFIX}", os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await anyio.sleep(LOCK_POLL_SECONDS)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


async def _promote(name: str, version: str, *, root: Path) -> None:
    info = await _resolve(name, version, root=root)
    family = anyio.Path(info.path).parent
    staging = family / f"{STAGING_PREFIX}{uuid4().hex}"
    await staging.symlink_to(info.version)
    await staging.replace(family / CURRENT_LINK)


async def _discard(path: anyio.Path) -> None:
    staging = path.parent / f"{VERSION_STAGING_PREFIX}{uuid4().hex}"
    await path.rename(staging)
    await anyio.to_thread.run_sync(shutil.rmtree, staging)


async def _materialize(destination: anyio.Path, content: bytes | Path) -> None:
    await destination.parent.mkdir(parents=True, exist_ok=True)
    match content:
        case bytes():
            await destination.write_bytes(content)
        case Path() if content.is_dir():
            await anyio.to_thread.run_sync(partial(shutil.copytree, content, destination))
        case Path():
            await anyio.to_thread.run_sync(shutil.copy2, content, destination)


async def _digest(staging: anyio.Path) -> str:
    hasher = hashlib.sha256()
    stored = [path async for path in staging.rglob("*") if await path.is_file()]
    for name, path in sorted((str(path.relative_to(staging)), path) for path in stored):
        hasher.update(name.encode())
        hasher.update(b"\0")
        async with await path.open("rb") as handle:
            while chunk := await handle.read(DIGEST_CHUNK_BYTES):
                hasher.update(chunk)
        hasher.update(b"\0")
    return hasher.hexdigest()[:DIGEST_CHARS]


async def _freeze(path: anyio.Path) -> None:
    async for child in path.rglob("*"):
        if await child.is_file():
            await child.chmod(READ_ONLY_MODE)


async def _info(name: str, path: anyio.Path) -> VersionInfo:
    metadata_path = path / METADATA_NAME
    metadata = json.loads(await metadata_path.read_text()) if await metadata_path.exists() else {}
    return VersionInfo(name=name, version=path.name, path=Path(path), metadata=metadata)


async def _resolve(name: str, version: str, *, root: Path) -> VersionInfo:
    known = await versions(name, root=root)
    if not (matches := [info for info in known if version in (info.version, info.version.split("-")[0])]):
        listing = ", ".join(info.version for info in known) or "none registered"
        raise RegistryError(f"no {name} version {version!r} ({listing})")
    return matches[-1]
