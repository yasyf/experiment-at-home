from __future__ import annotations

import hashlib
import itertools
from typing import TYPE_CHECKING

import anyio
import pytest

from athome import registry
from athome.registry import (
    DIGEST_CHARS,
    METADATA_NAME,
    STAGING_PREFIX,
    VERSION_STAGING_PREFIX,
    RegistryError,
    VersionInfo,
    components,
    current,
    promote,
    prune,
    register,
    rollback,
    versions,
)

if TYPE_CHECKING:
    from pathlib import Path


def content_digest(files: dict[str, bytes]) -> str:
    hasher = hashlib.sha256()
    for filename in sorted(files):
        hasher.update(filename.encode())
        hasher.update(b"\0")
        hasher.update(files[filename])
        hasher.update(b"\0")
    return hasher.hexdigest()[:DIGEST_CHARS]


class Barrier:
    """A minimal count-gated async barrier: every waiter blocks until ``n`` have arrived."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.count = 0
        self.event = anyio.Event()

    async def wait(self) -> None:
        self.count += 1
        if self.count >= self.n:
            self.event.set()
        await self.event.wait()


async def promote_one(name: str, version: str, root: Path) -> None:
    await promote(name, version, root=root)


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


async def test_path_source_is_read_once_so_digest_matches_stored_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # DIGEST TOCTOU: reading a path source twice (digest, then write) lets it mutate between
    # reads and store bytes that disagree with the version digest.
    src = tmp_path / "weights.bin"
    src.write_bytes(b"placeholder")
    reads = itertools.count()

    async def mutating_read(self: anyio.Path) -> bytes:
        return f"content-{next(reads)}".encode()

    monkeypatch.setattr(anyio.Path, "read_bytes", mutating_read)

    info = await register("toy", {"m": src}, {}, root=tmp_path)

    stored = (info.path / "m").read_bytes()
    assert info.version.split("-")[2] == content_digest({"m": stored})


async def test_register_copies_a_directory_tree_the_registry_then_owns(tmp_path: Path) -> None:
    # MUTABLE CHECKPOINTS: a version that pointed at a caller's scratch directory was mutable —
    # the next writer to that directory silently rewrote already-registered weights.
    source = tmp_path / "scratch" / "fused"
    (source / "nested").mkdir(parents=True)
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "nested" / "config.json").write_bytes(b"{}")

    info = await register("toy", {"model": source, "checkpoint.json": b"{}"}, {}, root=tmp_path)

    assert (info.path / "model" / "model.safetensors").read_bytes() == b"weights"
    assert (info.path / "model" / "nested" / "config.json").read_bytes() == b"{}"

    (source / "model.safetensors").write_bytes(b"clobbered")
    assert (info.path / "model" / "model.safetensors").read_bytes() == b"weights"


async def test_registered_files_are_frozen_read_only(tmp_path: Path) -> None:
    source = tmp_path / "scratch" / "fused"
    source.mkdir(parents=True)
    (source / "model.safetensors").write_bytes(b"weights")

    info = await register("toy", {"model": source}, {}, root=tmp_path)

    assert (info.path / "model" / "model.safetensors").stat().st_mode & 0o222 == 0
    assert (info.path / METADATA_NAME).stat().st_mode & 0o222 == 0
    with pytest.raises(PermissionError):
        (info.path / "model" / "model.safetensors").write_bytes(b"clobbered")


async def test_a_directory_digest_addresses_the_tree_and_leaves_no_staging(tmp_path: Path) -> None:
    def tree(where: Path, weights: bytes) -> Path:
        (where / "nested").mkdir(parents=True)
        (where / "nested" / "model.safetensors").write_bytes(weights)
        return where

    same = await register("toy", {"model": tree(tmp_path / "a", b"weights")}, {}, root=tmp_path)
    twin = await register("other", {"model": tree(tmp_path / "b", b"weights")}, {}, root=tmp_path)
    differs = await register("third", {"model": tree(tmp_path / "c", b"other")}, {}, root=tmp_path)

    assert same.version.split("-")[2] == twin.version.split("-")[2]
    assert same.version.split("-")[2] != differs.version.split("-")[2]
    assert not any(child.name.startswith(VERSION_STAGING_PREFIX) for child in (tmp_path / "toy").iterdir())


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
    assert not any(child.name.startswith(STAGING_PREFIX) for child in family.iterdir())
    assert (family / "current").is_symlink()
    assert sorted(child.name for child in family.iterdir() if not child.is_symlink()) == [v1.version, v2.version]


async def test_concurrent_promotions_do_not_clobber(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # WR4: two promotions of the same family race at the symlink step; a shared staging
    # name makes one clobber the other, so both must use a unique per-promotion name.
    v1 = await register("toy", {"m": b"a"}, {}, root=tmp_path)
    v2 = await register("toy", {"m": b"b"}, {}, root=tmp_path)

    barrier = Barrier(2)
    real_symlink = anyio.Path.symlink_to

    async def synced_symlink(self: anyio.Path, target: object, *args: object, **kwargs: object) -> None:
        await barrier.wait()  # force both promotions to collide at the staging symlink
        return await real_symlink(self, target, *args, **kwargs)

    monkeypatch.setattr(anyio.Path, "symlink_to", synced_symlink)

    async with anyio.create_task_group() as tg:
        tg.start_soon(promote_one, "toy", v1.version, tmp_path)
        tg.start_soon(promote_one, "toy", v2.version, tmp_path)

    promoted = await current("toy", root=tmp_path)
    assert promoted is not None and promoted.version in {v1.version, v2.version}
    family = tmp_path / "toy"
    assert (family / "current").is_symlink()
    assert not any(child.name.startswith(STAGING_PREFIX) for child in family.iterdir())
    assert sorted(child.name for child in family.iterdir() if not child.is_symlink()) == [v1.version, v2.version]


async def test_concurrent_registrations_mint_distinct_numbers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # SINGLE-WRITER: two registrations that read the next number concurrently must not both
    # mint v001; the family-scoped lock serializes the read-number -> write window.
    gate = anyio.Event()
    readers = 0
    real_versions = registry.versions

    async def gated_versions(name: str, *, root: Path) -> list[VersionInfo]:
        nonlocal readers
        result = await real_versions(name, root=root)
        readers += 1
        if readers >= 2:
            gate.set()  # a second reader arrived: release the first (unserialized) reader
        else:
            with anyio.move_on_after(0.25):  # serialized: no second reader comes, so time out
                await gate.wait()
        return result

    monkeypatch.setattr(registry, "versions", gated_versions)

    async def register_one(content: bytes) -> None:
        await register("toy", {"m": content}, {}, root=tmp_path)

    async with anyio.create_task_group() as tg:
        tg.start_soon(register_one, b"a")
        tg.start_soon(register_one, b"b")

    eras = sorted(info.version.split("-")[0] for info in await real_versions("toy", root=tmp_path))
    assert eras == ["v001", "v002"]


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


async def test_components_lists_registered_families_sorted(tmp_path: Path) -> None:
    await register("zeta", {"m": b"a"}, {}, root=tmp_path)
    await register("alpha", {"m": b"b"}, {}, root=tmp_path)
    await register("alpha", {"m": b"c"}, {}, root=tmp_path)
    # The per-family <name>.lock files are regular files, not directories, so they stay out.
    assert await components(tmp_path) == ("alpha", "zeta")


async def test_components_of_missing_root_is_empty(tmp_path: Path) -> None:
    assert await components(tmp_path / "nope") == ()


async def test_rollback_repoints_current_to_prior_version(tmp_path: Path) -> None:
    await register("toy", {"m": b"a"}, {}, root=tmp_path)
    v2 = await register("toy", {"m": b"b"}, {}, root=tmp_path)
    v3 = await register("toy", {"m": b"c"}, {}, root=tmp_path)
    await promote("toy", v3.version, root=tmp_path)

    rolled = await rollback("toy", root=tmp_path)

    assert rolled.version == v2.version
    promoted = await current("toy", root=tmp_path)
    assert promoted is not None and promoted.version == v2.version


async def test_rollback_from_earliest_version_raises(tmp_path: Path) -> None:
    v1 = await register("toy", {"m": b"a"}, {}, root=tmp_path)
    await register("toy", {"m": b"b"}, {}, root=tmp_path)
    await promote("toy", v1.version, root=tmp_path)
    with pytest.raises(RegistryError):
        await rollback("toy", root=tmp_path)


async def test_rollback_without_promotion_raises(tmp_path: Path) -> None:
    await register("toy", {"m": b"a"}, {}, root=tmp_path)
    with pytest.raises(RegistryError):
        await rollback("toy", root=tmp_path)


async def test_prune_keeps_newest_and_removes_the_rest(tmp_path: Path) -> None:
    for content in (b"a", b"b", b"c", b"d", b"e"):
        await register("toy", {"m": content}, {}, root=tmp_path)

    removed = await prune("toy", keep=2, root=tmp_path)

    # Removed versions are frozen read-only on disk, so a passing rmtree proves the unfreeze.
    assert [info.number for info in removed] == [1, 2, 3]
    assert not any(info.path.exists() for info in removed)
    assert [info.number for info in await versions("toy", root=tmp_path)] == [4, 5]


async def test_prune_never_deletes_the_current_version(tmp_path: Path) -> None:
    v1 = await register("toy", {"m": b"a"}, {}, root=tmp_path)
    for content in (b"b", b"c", b"d"):
        await register("toy", {"m": content}, {}, root=tmp_path)
    await promote("toy", v1.version, root=tmp_path)  # current is the oldest version

    removed = await prune("toy", keep=1, root=tmp_path)

    assert v1.version not in {info.version for info in removed}
    surviving = {info.number for info in await versions("toy", root=tmp_path)}
    assert surviving == {1, 4}  # current (outside the keep=1 window) plus the newest
    promoted = await current("toy", root=tmp_path)
    assert promoted is not None and promoted.version == v1.version


async def test_prune_keep_larger_than_count_removes_nothing(tmp_path: Path) -> None:
    await register("toy", {"m": b"a"}, {}, root=tmp_path)
    await register("toy", {"m": b"b"}, {}, root=tmp_path)
    assert await prune("toy", keep=5, root=tmp_path) == ()
    assert [info.number for info in await versions("toy", root=tmp_path)] == [1, 2]
