"""The research artifact registry: :mod:`athome.registry` bound to the ``[research]`` root."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from athome import registry
from athome.config import SectionSettings, load
from athome.registry import CURRENT_LINK as CURRENT_LINK
from athome.registry import DIGEST_CHARS as DIGEST_CHARS
from athome.registry import LOCK_POLL_SECONDS as LOCK_POLL_SECONDS
from athome.registry import LOCK_SUFFIX as LOCK_SUFFIX
from athome.registry import METADATA_NAME as METADATA_NAME
from athome.registry import STAGING_PREFIX as STAGING_PREFIX
from athome.registry import VERSION_PATTERN as VERSION_PATTERN
from athome.registry import RegistryError as RegistryError
from athome.registry import VersionInfo

if TYPE_CHECKING:
    from collections.abc import Mapping


class RegistrySettings(SectionSettings):
    """The research artifact registry, bound to ``[research]`` of the athome config."""

    section: ClassVar[tuple[str, ...]] = ("research",)
    registry_root: Path = Path("~/.athome/research/registry")


def registry_root(root: Path | None = None) -> Path:
    """The registry root: the ``root`` argument, or ``[research].registry_root`` from the config."""
    return root if root is not None else load(RegistrySettings).registry_root


async def versions(name: str, *, root: Path | None = None) -> list[VersionInfo]:
    """Every registered version of an artifact family, oldest first."""
    return await registry.versions(name, root=registry_root(root))


async def current(name: str, *, root: Path | None = None) -> VersionInfo | None:
    """The promoted version the ``current`` symlink names, or ``None`` when nothing is promoted."""
    return await registry.current(name, root=registry_root(root))


async def register(
    name: str, files: Mapping[str, bytes | Path], metadata: Mapping[str, object], *, root: Path | None = None
) -> VersionInfo:
    """Writes a new immutable version directory; never flips ``current``."""
    return await registry.register(name, files, metadata, root=registry_root(root))


async def promote(name: str, version: str, *, root: Path | None = None) -> None:
    """Atomically flips ``current`` to the named version (full name or ``v<NNN>`` prefix)."""
    await registry.promote(name, version, root=registry_root(root))
