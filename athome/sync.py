from __future__ import annotations

import shlex
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import click

from athome.cli import coro, emit, json_option
from athome.errors import AthomeError

if TYPE_CHECKING:
    from collections.abc import Sequence

MAX_REPAIR_ROUNDS = 3
REQUIRED_TOOLS = ("rsync", "shasum")
SHELL_METACHARACTERS = frozenset(";|&$`\n")


class SyncVerificationError(AthomeError):
    """A mirror could not be verified clean; carries the unverified file list in ``mismatches``."""

    def __init__(self, message: str, *, mismatches: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.mismatches = tuple(mismatches)


@dataclass(frozen=True, slots=True)
class SyncReport:
    """The outcome of a verified :func:`mirror`: file/byte counts and swept AppleDoubles."""

    files: int
    bytes: int
    verified: bool
    swept_appledoubles: int


@dataclass(frozen=True, slots=True)
class Location:
    host: str | None
    path: str

    @classmethod
    def parse(cls, target: str) -> Location:
        if "[" in target and (end := target.find("]")) != -1 and target[end + 1 : end + 2] == ":":
            return cls(target[: end + 1], target[end + 2 :])
        head, sep, tail = target.partition(":")
        return cls(head, tail) if sep and "/" not in head else cls(None, target)


def require_tools() -> None:
    for tool in REQUIRED_TOOLS:
        if shutil.which(tool) is None:
            raise SyncVerificationError(f"required tool not found on PATH: {tool}")


def guard_operands(*operands: str | None) -> None:
    if bad := next((op for op in operands if op and (op.startswith("-") or SHELL_METACHARACTERS & set(op))), None):
        raise SyncVerificationError(f"refusing unsafe sync operand: {bad!r}")


def parse_manifest(output: str) -> dict[str, str]:
    return {
        path.removeprefix("./"): digest
        for digest, _, path in (line.partition("  ") for line in output.splitlines() if line)
    }


def manifest_command(path: str) -> str:
    return f"cd {shlex.quote(path)} && find . -type f ! -name '._*' -exec shasum -a 256 {{}} +"


def sweep_command(path: str) -> str:
    return f"find {shlex.quote(path)} -type f -name '._*' -print -delete"


def diff_manifests(source: dict[str, str], dest: dict[str, str]) -> list[str]:
    return sorted(name for name in source.keys() | dest.keys() if source.get(name) != dest.get(name))


def local_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file() and not path.name.startswith("._")]


async def shell(command: str, *, host: str | None = None) -> str:
    return (await anyio.run_process(("ssh", host, command) if host else ("/bin/sh", "-c", command))).stdout.decode()


async def rsync(src: Path, dst: str, *, checksum: bool = False, remote: bool = False) -> None:
    await anyio.run_process(
        (
            "rsync",
            "-a",
            *(("--checksum",) if checksum else ()),
            *(("-e", "ssh") if remote else ()),
            "--exclude",
            "._*",
            "--",
            f"{src}/",
            f"{dst}/",
        )
    )


async def sweep(location: Location) -> int:
    return len((await shell(sweep_command(location.path), host=location.host)).splitlines())


async def src_manifest(src: Path) -> dict[str, str]:
    return parse_manifest(await shell(manifest_command(str(src))))


async def dst_manifest(location: Location) -> dict[str, str]:
    return parse_manifest(await shell(manifest_command(location.path), host=location.host))


def prune_empty_dirs(root: Path) -> None:
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if not any(path.iterdir()):
            path.rmdir()


async def delete_verified(src: Path, verified: dict[str, str]) -> None:
    current = await src_manifest(src)
    for name in (name for name, digest in verified.items() if current.get(name) == digest):
        (src / name).unlink()
    prune_empty_dirs(src)


async def mirror(src: Path, dst: str, *, delete_source: bool = False) -> SyncReport:
    """Replicate ``src`` onto ``dst`` (a local path or ``"host:path"``) and verify it by sha256.

    Runs ``rsync -a``, sweeps AppleDouble ``._*`` files after every hop, then diffs a sorted
    sha256 manifest of both trees; a mismatch triggers up to :data:`MAX_REPAIR_ROUNDS` checksum
    re-syncs before raising :class:`SyncVerificationError`. Source contents are removed only when
    ``delete_source`` is set and the tree verifies clean.

    Args:
        src: The local directory to replicate.
        dst: The destination, either a local path or an ``rsync``/``ssh`` ``"host:path"`` spec.
        delete_source: Remove the source contents once, and only once, the mirror verifies.

    Returns:
        A :class:`SyncReport` describing the verified transfer.

    Raises:
        SyncVerificationError: A required tool is missing, or the tree stays unverified.
    """
    require_tools()
    location = Location.parse(dst)
    guard_operands(str(src), dst, location.host, location.path)
    swept = 0
    mismatches: list[str] = []
    source: dict[str, str] = {}
    for attempt in range(MAX_REPAIR_ROUNDS + 1):
        await rsync(src, dst, checksum=attempt > 0, remote=location.host is not None)
        swept += await sweep(location)
        if not (mismatches := diff_manifests(source := await src_manifest(src), await dst_manifest(location))):
            break
    else:
        raise SyncVerificationError(
            f"unverified after {MAX_REPAIR_ROUNDS} repair round(s): {', '.join(mismatches)}",
            mismatches=mismatches,
        )
    files = local_files(src)
    report = SyncReport(
        files=len(files),
        bytes=sum(path.stat().st_size for path in files),
        verified=True,
        swept_appledoubles=swept,
    )
    if delete_source:
        await delete_verified(src, source)
    return report


@click.command(name="sync")
@click.argument("src", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("dst")
@click.option("--move", is_flag=True, help="Delete the source once the mirror verifies clean.")
@json_option
@coro
async def cli(src: Path, dst: str, *, move: bool, as_json: bool) -> None:
    """Mirror SRC onto DST with sha256 verification; --move deletes SRC once verified."""
    emit(asdict(await mirror(src, dst, delete_source=move)), as_json=as_json)
