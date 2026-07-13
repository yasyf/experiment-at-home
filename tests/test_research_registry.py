from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from athome.research.registry import RegistryError, current, promote, register, versions

if TYPE_CHECKING:
    from pathlib import Path


async def test_register_writes_a_content_addressed_version(tmp_path: Path) -> None:
    info = await register("toy", {"model.bin": b"weights"}, {"metric": 0.9}, root=tmp_path)
    assert info.name == "toy"
    assert info.number == 1
    era, date, digest = info.version.split("-")
    assert era == "v001"
    assert len(date) == 8 and date.isdigit()
    assert len(digest) == 12
    assert (info.path / "model.bin").read_bytes() == b"weights"
    assert info.metadata["metric"] == 0.9
    assert info.metadata["name"] == "toy"
    assert info.metadata["version"] == info.version


async def test_content_addressing_reflects_the_bytes(tmp_path: Path) -> None:
    a = await register("toy", {"m": b"same"}, {}, root=tmp_path)
    b = await register("other", {"m": b"same"}, {}, root=tmp_path)
    assert a.version.split("-")[2] == b.version.split("-")[2]


async def test_versions_are_ordered_and_register_never_promotes(tmp_path: Path) -> None:
    await register("toy", {"m": b"a"}, {}, root=tmp_path)
    await register("toy", {"m": b"b"}, {}, root=tmp_path)
    assert [v.number for v in await versions("toy", root=tmp_path)] == [1, 2]
    assert await current("toy", root=tmp_path) is None


async def test_register_refuses_an_empty_version(tmp_path: Path) -> None:
    with pytest.raises(RegistryError):
        await register("toy", {}, {}, root=tmp_path)


async def test_promote_is_an_atomic_symlink_swap(tmp_path: Path) -> None:
    v1 = await register("toy", {"m": b"a"}, {}, root=tmp_path)
    v2 = await register("toy", {"m": b"b"}, {}, root=tmp_path)

    await promote("toy", v1.version, root=tmp_path)
    promoted = await current("toy", root=tmp_path)
    assert promoted is not None and promoted.version == v1.version

    await promote("toy", "v002", root=tmp_path)
    promoted = await current("toy", root=tmp_path)
    assert promoted is not None and promoted.version == v2.version

    family = tmp_path / "toy"
    assert not (family / ".current.tmp").exists()
    assert (family / "current").is_symlink()
    assert sorted(child.name for child in family.iterdir() if not child.is_symlink()) == [v1.version, v2.version]


async def test_promote_resolves_a_bare_version_prefix(tmp_path: Path) -> None:
    v1 = await register("toy", {"m": b"a"}, {}, root=tmp_path)
    await promote("toy", "v001", root=tmp_path)
    promoted = await current("toy", root=tmp_path)
    assert promoted is not None and promoted.version == v1.version


async def test_promote_unknown_version_raises(tmp_path: Path) -> None:
    await register("toy", {"m": b"a"}, {}, root=tmp_path)
    with pytest.raises(RegistryError):
        await promote("toy", "v999", root=tmp_path)


async def test_versions_of_unknown_family_is_empty(tmp_path: Path) -> None:
    assert await versions("nope", root=tmp_path) == []
    assert await current("nope", root=tmp_path) is None
