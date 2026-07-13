from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import anyio

from athome.config import SectionSettings, load
from athome.research.errors import ResearchError

if TYPE_CHECKING:
    from collections.abc import Mapping

METADATA_NAME = "metadata.json"
CURRENT_LINK = "current"
STAGING_LINK = f".{CURRENT_LINK}.tmp"
VERSION_PATTERN = re.compile(r"^v(\d{3,})-(\d{8})-([0-9a-f]{12})$")
DIGEST_CHARS = 12


class RegistryError(ResearchError):
    """The registry cannot satisfy the request: an unknown version, or an empty registration."""


class RegistrySettings(SectionSettings):
    """The research artifact registry, bound to ``[research]`` of the athome config."""

    section: ClassVar[tuple[str, ...]] = ("research",)
    registry_root: Path = Path("~/.athome/research/registry")


@dataclass(frozen=True, slots=True)
class VersionInfo:
    """One registered artifact version.

    Attributes:
        name: The artifact family (the experiment name).
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


def registry_root(root: Path | None = None) -> Path:
    """The registry root: the ``root`` argument, or ``[research].registry_root`` from the config."""
    return root if root is not None else load(RegistrySettings).registry_root


async def versions(name: str, *, root: Path | None = None) -> list[VersionInfo]:
    """Every registered version of an artifact family, oldest first."""
    base = anyio.Path(registry_root(root)) / name
    if not await base.is_dir():
        return []
    found = [
        await _info(name, child)
        async for child in base.iterdir()
        if VERSION_PATTERN.match(child.name) and await child.is_dir() and not await child.is_symlink()
    ]
    return sorted(found, key=lambda info: info.number)


async def current(name: str, *, root: Path | None = None) -> VersionInfo | None:
    """The promoted version the ``current`` symlink names, or ``None`` when nothing is promoted."""
    link = anyio.Path(registry_root(root)) / name / CURRENT_LINK
    if not await link.is_symlink():
        return None
    return await _info(name, link.parent / await link.readlink())


async def register(
    name: str, files: Mapping[str, bytes | Path], metadata: Mapping[str, object], *, root: Path | None = None
) -> VersionInfo:
    """Writes a new immutable version directory; never flips ``current``.

    The directory name embeds the next version number, today's date, and a
    12-hex content digest over the artifact files. ``metadata.json`` is the
    caller's metadata stamped with ``name``, ``version``, and ``created_at``.

    Args:
        files: Artifact file name to content — raw bytes, or a path to copy.
        metadata: The version's provenance: dataset digest, config, metrics.

    Returns:
        The freshly registered :class:`VersionInfo`.
    """
    if not files:
        raise RegistryError(f"refusing to register an empty {name} version")
    existing = await versions(name, root=root)
    number = existing[-1].number + 1 if existing else 1
    now = datetime.now(UTC)
    version = f"v{number:03d}-{now:%Y%m%d}-{await _digest(files)}"
    path = anyio.Path(registry_root(root)) / name / version
    await path.mkdir(parents=True)
    for filename, content in files.items():
        payload = content if isinstance(content, bytes) else await anyio.Path(content).read_bytes()
        await (path / filename).write_bytes(payload)
    stamped = dict(metadata) | {"name": name, "version": version, "created_at": now.isoformat()}
    await (path / METADATA_NAME).write_text(json.dumps(stamped, indent=2, sort_keys=True, default=str) + "\n")
    return VersionInfo(name=name, version=version, path=Path(path), metadata=stamped)


async def promote(name: str, version: str, *, root: Path | None = None) -> None:
    """Atomically flips ``current`` to the named version (full name or ``v<NNN>`` prefix).

    The flip writes a staging symlink and renames it over ``current``, so a
    reader never sees a missing or half-written link.
    """
    info = await _resolve(name, version, root=root)
    family = anyio.Path(info.path).parent
    staging = family / STAGING_LINK
    await staging.unlink(missing_ok=True)
    await staging.symlink_to(info.version)
    await staging.replace(family / CURRENT_LINK)


async def _info(name: str, path: anyio.Path) -> VersionInfo:
    metadata_path = path / METADATA_NAME
    metadata = json.loads(await metadata_path.read_text()) if await metadata_path.exists() else {}
    return VersionInfo(name=name, version=path.name, path=Path(path), metadata=metadata)


async def _resolve(name: str, version: str, *, root: Path | None) -> VersionInfo:
    known = await versions(name, root=root)
    if not (matches := [info for info in known if version in (info.version, info.version.split("-")[0])]):
        listing = ", ".join(info.version for info in known) or "none registered"
        raise RegistryError(f"no {name} version {version!r} ({listing})")
    return matches[-1]


async def _digest(files: Mapping[str, bytes | Path]) -> str:
    hasher = hashlib.sha256()
    for filename in sorted(files):
        content = files[filename]
        hasher.update(filename.encode())
        hasher.update(b"\0")
        hasher.update(content if isinstance(content, bytes) else await anyio.Path(content).read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()[:DIGEST_CHARS]
