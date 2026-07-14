"""Immutable, content-addressed artifact versions with an atomic ``current`` promotion."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import anyio

from athome.errors import AthomeError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

METADATA_NAME = "metadata.json"
CURRENT_LINK = "current"
STAGING_PREFIX = f".{CURRENT_LINK}.tmp."
VERSION_PATTERN = re.compile(r"^v(\d{3,})-(\d{8})-([0-9a-f]{12})$")
DIGEST_CHARS = 12
LOCK_SUFFIX = ".lock"
LOCK_POLL_SECONDS = 0.01


class RegistryError(AthomeError):
    """The registry cannot satisfy the request: an unknown version, or an empty registration."""


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

    The directory name embeds the next version number, today's date, and a
    12-hex content digest over the artifact files. ``metadata.json`` is the
    caller's metadata stamped with ``name``, ``version``, and ``created_at``.

    Args:
        files: Artifact file name to content — raw bytes, or a path to copy.
        metadata: The version's provenance: dataset digest, config, metrics.
        root: The registry root the family lives under.

    Returns:
        The freshly registered :class:`VersionInfo`.
    """
    if not files:
        raise RegistryError(f"refusing to register an empty {name} version")
    snapshot = {
        filename: content if isinstance(content, bytes) else await anyio.Path(content).read_bytes()
        for filename, content in files.items()
    }
    family = anyio.Path(root) / name
    async with _family_lock(family):
        number = existing[-1].number + 1 if (existing := await versions(name, root=root)) else 1
        now = datetime.now(UTC)
        version = f"v{number:03d}-{now:%Y%m%d}-{_digest(snapshot)}"
        path = family / version
        await path.mkdir(parents=True)
        for filename, payload in snapshot.items():
            await (path / filename).write_bytes(payload)
        stamped = dict(metadata) | {"name": name, "version": version, "created_at": now.isoformat()}
        await (path / METADATA_NAME).write_text(json.dumps(stamped, indent=2, sort_keys=True, default=str) + "\n")
    return VersionInfo(name=name, version=version, path=Path(path), metadata=stamped)


async def promote(name: str, version: str, *, root: Path) -> None:
    """Atomically flips ``current`` to the named version (full name or ``v<NNN>`` prefix).

    The flip writes a staging symlink under a unique per-promotion name and renames
    it over ``current``, so a reader never sees a missing or half-written link and
    two concurrent promotions never clobber one another's staging.
    """
    info = await _resolve(name, version, root=root)
    family = anyio.Path(info.path).parent
    staging = family / f"{STAGING_PREFIX}{uuid4().hex}"
    await staging.symlink_to(info.version)
    await staging.replace(family / CURRENT_LINK)


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


def _digest(files: Mapping[str, bytes]) -> str:
    hasher = hashlib.sha256()
    for filename in sorted(files):
        hasher.update(filename.encode())
        hasher.update(b"\0")
        hasher.update(files[filename])
        hasher.update(b"\0")
    return hasher.hexdigest()[:DIGEST_CHARS]
