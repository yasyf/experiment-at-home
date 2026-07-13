from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pytest

import athome.sync
from athome.sync import Location, SyncVerificationError, mirror

if TYPE_CHECKING:
    from pathlib import Path


def make_tree(root: Path, files: dict[str, bytes]) -> Path:
    for name, data in files.items():
        (path := root / name).parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


async def test_local_roundtrip_verifies(tmp_path: Path) -> None:
    src = make_tree(tmp_path / "src", {"a.txt": b"hello", "sub/b.txt": b"world!!"})
    dst = tmp_path / "dst"
    report = await mirror(src, str(dst))
    assert report.verified is True
    assert report.files == 2
    assert report.bytes == len(b"hello") + len(b"world!!")
    assert report.swept_appledoubles == 0
    assert (dst / "a.txt").read_bytes() == b"hello"
    assert (dst / "sub" / "b.txt").read_bytes() == b"world!!"
    assert (src / "a.txt").exists()


async def test_appledoubles_swept_and_source_ones_excluded(tmp_path: Path) -> None:
    src = make_tree(tmp_path / "src", {"a.txt": b"keep", "._src": b"apple-junk"})
    dst = make_tree(tmp_path / "dst", {"._seed": b"stale", "._also": b"stale"})
    report = await mirror(src, str(dst))
    assert report.verified is True
    assert report.swept_appledoubles == 2
    assert not (dst / "._seed").exists()
    assert not (dst / "._also").exists()
    assert not (dst / "._src").exists()
    assert (dst / "a.txt").read_bytes() == b"keep"


async def test_corruption_blocks_move_and_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = make_tree(tmp_path / "src", {"a.txt": b"precious"})
    dst = tmp_path / "dst"
    real_dst_manifest = athome.sync.dst_manifest

    async def corrupt(location: Location) -> dict[str, str]:
        manifest = await real_dst_manifest(location)
        return manifest | {next(iter(manifest)): "0" * 64}

    monkeypatch.setattr(athome.sync, "dst_manifest", corrupt)
    with pytest.raises(SyncVerificationError) as excinfo:
        await mirror(src, str(dst), delete_source=True)
    assert excinfo.value.mismatches == ("a.txt",)
    assert (src / "a.txt").read_bytes() == b"precious"


async def test_delete_source_removes_src_when_clean(tmp_path: Path) -> None:
    src = make_tree(tmp_path / "src", {"a.txt": b"data", "sub/b.txt": b"more"})
    dst = tmp_path / "dst"
    report = await mirror(src, str(dst), delete_source=True)
    assert report.verified is True
    assert list(src.iterdir()) == []
    assert (dst / "a.txt").read_bytes() == b"data"
    assert (dst / "sub" / "b.txt").read_bytes() == b"more"


@pytest.mark.parametrize("missing", ["rsync", "shasum"])
async def test_missing_tool_names_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    src = make_tree(tmp_path / "src", {"a.txt": b"x"})
    monkeypatch.setattr(shutil, "which", lambda name: None if name == missing else f"/usr/bin/{name}")
    with pytest.raises(SyncVerificationError, match=missing):
        await mirror(src, str(tmp_path / "dst"))


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("/local/abs/path", Location(None, "/local/abs/path")),
        ("relative/path", Location(None, "relative/path")),
        ("host:/remote/path", Location("host", "/remote/path")),
        ("user@host:data", Location("user@host", "data")),
        ("./weird:name", Location(None, "./weird:name")),
    ],
    ids=["absolute", "relative", "remote", "user-host", "colon-in-local"],
)
def test_location_parse(target: str, expected: Location) -> None:
    assert Location.parse(target) == expected


def test_location_parse_ipv6_bracket() -> None:
    assert Location.parse("user@[::1]:/dst") == Location("user@[::1]", "/dst")


@pytest.mark.parametrize(
    "dst",
    ["-oProxyCommand=evil", "/dst;touch /tmp/pwn", "host:/safe|reboot", "host:/x`id`"],
    ids=["leading-dash", "semicolon", "pipe", "backtick"],
)
async def test_unsafe_operands_rejected_before_rsync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dst: str) -> None:
    src = make_tree(tmp_path / "src", {"a.txt": b"x"})
    called = False

    async def fail_rsync(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(athome.sync, "rsync", fail_rsync)
    with pytest.raises(SyncVerificationError, match="unsafe"):
        await mirror(src, dst)
    assert called is False


async def test_move_preserves_files_created_after_verification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = make_tree(tmp_path / "src", {"a.txt": b"keep"})
    dst = tmp_path / "dst"
    real_local_files = athome.sync.local_files

    def inject_after_verification(root: Path) -> list[Path]:
        result = real_local_files(root)
        (root / "sneaky.txt").write_bytes(b"appeared-after-verify")
        return result

    monkeypatch.setattr(athome.sync, "local_files", inject_after_verification)
    report = await mirror(src, str(dst), delete_source=True)
    assert report.verified is True
    assert not (src / "a.txt").exists()
    assert (src / "sneaky.txt").read_bytes() == b"appeared-after-verify"
