from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from athome import registry, train
from athome.config import load
from athome.errors import AthomeError
from athome.research import registry as research_registry
from athome.research.registry import (
    DIGEST_CHARS,
    METADATA_NAME,
    STAGING_PREFIX,
    VERSION_PATTERN,
    RegistryError,
    RegistrySettings,
    VersionInfo,
    current,
    promote,
    register,
    registry_root,
    versions,
)
from athome.train.spec import TrainSettings

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Path, Path]]:
    research, trained = tmp_path / "research-registry", tmp_path / "train-registry"
    monkeypatch.setenv("ATHOME_RESEARCH_REGISTRY_ROOT", str(research))
    monkeypatch.setenv("ATHOME_TRAIN_REGISTRY_ROOT", str(trained))
    load.cache_clear()
    yield research, trained
    load.cache_clear()


def test_the_shim_re_exports_the_promoted_registry() -> None:
    assert RegistryError is registry.RegistryError
    assert VersionInfo is registry.VersionInfo
    assert (DIGEST_CHARS, STAGING_PREFIX, METADATA_NAME, VERSION_PATTERN) == (
        registry.DIGEST_CHARS,
        registry.STAGING_PREFIX,
        registry.METADATA_NAME,
        registry.VERSION_PATTERN,
    )


def test_the_research_package_still_exports_the_registry_surface() -> None:
    import athome.research as research

    assert (research.RegistryError, research.VersionInfo) == (RegistryError, VersionInfo)
    assert (research.register, research.promote, research.versions, research.current) == (
        register,
        promote,
        versions,
        current,
    )


def test_registry_error_is_an_athome_error() -> None:
    assert issubclass(RegistryError, AthomeError)


def test_the_research_root_default_is_unchanged() -> None:
    assert RegistrySettings().registry_root == Path.home() / ".athome/research/registry"


def test_registry_root_falls_back_to_the_research_section(roots: tuple[Path, Path]) -> None:
    research, _ = roots
    assert registry_root() == research
    assert registry_root(Path("/explicit")) == Path("/explicit")


async def test_research_writes_to_its_configured_root_without_an_explicit_root(roots: tuple[Path, Path]) -> None:
    research, trained = roots
    info = await register("toy", {"m": b"a"}, {"metric": 0.5})
    assert info.path.parent.parent == research
    assert (info.path / METADATA_NAME).exists()
    assert not trained.exists()

    await promote("toy", info.version)
    promoted = await current("toy")
    assert promoted is not None and promoted.version == info.version
    assert [version.version for version in await versions("toy")] == [info.version]


async def test_the_research_and_train_roots_are_isolated(roots: tuple[Path, Path]) -> None:
    research, trained = roots
    assert load(TrainSettings).registry_root == trained

    researched = await register("watcher", {"m": b"research"}, {})
    fused = await train.register("watcher", Path("/runs/watcher/fused"), {}, root=trained)

    assert researched.path.parent.parent == research
    assert fused.path.parent.parent == trained
    assert [version.version for version in await versions("watcher")] == [researched.version]
    assert [version.version for version in await registry.versions("watcher", root=trained)] == [fused.version]

    await promote("watcher", researched.version)
    assert await registry.current("watcher", root=trained) is None
    assert (await current("watcher")).version == researched.version  # type: ignore[union-attr]


async def test_the_shim_delegates_to_the_promoted_registry(roots: tuple[Path, Path], tmp_path: Path) -> None:
    calls: list[Path] = []
    real_register = registry.register

    async def recording_register(name: str, files: object, metadata: object, *, root: Path) -> VersionInfo:
        calls.append(root)
        return await real_register(name, files, metadata, root=root)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(research_registry.registry, "register", recording_register)
        await register("toy", {"m": b"a"}, {})
        await register("toy", {"m": b"b"}, {}, root=tmp_path / "explicit")

    assert calls == [roots[0], tmp_path / "explicit"]


async def test_the_shim_still_raises_registry_error(roots: tuple[Path, Path]) -> None:
    with pytest.raises(RegistryError):
        await register("toy", {}, {})
    with pytest.raises(RegistryError):
        await promote("toy", "v999")
