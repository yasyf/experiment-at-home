from __future__ import annotations

import functools
from pathlib import Path
from typing import ClassVar

import click
from anyio import to_thread
from pydantic import Field, SecretStr

from athome.cli import coro, emit, json_option
from athome.config import SectionSettings, load
from athome.errors import AthomeError

REVISIONS: dict[str, str] = {
    "rrivera1849/LUAR-MUD": "f1db50251805ed69b43cf4f72ea2f0e231f36a1c",
    "StyleDistance/styledistance": "b7df5f0b0480773c097ba3121d83ca32b71015ca",
    "StyleDistance/mstyledistance": "d66ed25e48225a503b21a65bc804caf06c886f96",
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
}


class HfError(AthomeError):
    """Root of every HF hub error: an unpinned repo, or a failed transfer."""


class HfAuthError(HfError):
    """Raised when the write-role preflight fails: a read-scoped ``HF_TOKEN`` that would 403 on push."""


class HfSettings(SectionSettings):
    """HF hub credentials, bound to ``[hf]`` and the canonical ``HF_TOKEN`` env var."""

    section: ClassVar[tuple[str, ...]] = ("hf",)
    token: SecretStr = Field(validation_alias="HF_TOKEN")


def revision_for(repo: str) -> str:
    if (revision := REVISIONS.get(repo)) is None:
        raise HfError(f"unpinned repo {repo!r}: add it to REVISIONS or pass an explicit revision")
    return revision


async def snapshot(repo: str, *, revision: str | None = None) -> Path:
    """Download ``repo`` at its pinned revision into the HF cache and return the local path.

    Args:
        repo: The ``owner/name`` HF repo id.
        revision: An explicit commit SHA; defaults to the :data:`REVISIONS` pin for ``repo``.

    Returns:
        The local filesystem path of the materialized snapshot.

    Raises:
        HfError: ``repo`` carries no :data:`REVISIONS` pin and no ``revision`` was given.
    """
    from huggingface_hub import snapshot_download

    return Path(
        await to_thread.run_sync(
            functools.partial(
                snapshot_download,
                repo,
                revision=revision or revision_for(repo),
                token=load(HfSettings).token.get_secret_value(),
            )
        )
    )


async def ensure_write_auth() -> None:
    """Assert ``HF_TOKEN`` carries write scope, failing loudly before any push reaches the hub.

    A read-scoped token ``whoami``\\ s cleanly yet 403s on upload; this preflight raises on the
    role instead, so the failure surfaces here rather than mid-push.

    Raises:
        HfAuthError: The token resolves to a non-write role.
    """
    from huggingface_hub import whoami

    info = await to_thread.run_sync(functools.partial(whoami, token=load(HfSettings).token.get_secret_value()))
    if (role := info["auth"]["accessToken"]["role"]) != "write":
        raise HfAuthError(f"HF_TOKEN has role {role!r}; a write-scoped token is required to push")


async def push(repo: str, local_dir: Path, *, revision: str = "main") -> None:
    """Upload ``local_dir`` to ``repo`` after a write-role preflight.

    Args:
        repo: The ``owner/name`` HF repo id to upload into.
        local_dir: The local folder whose contents are pushed.
        revision: The target branch or revision.

    Raises:
        HfAuthError: The write-role preflight failed; nothing is uploaded.
    """
    from huggingface_hub import upload_folder

    await ensure_write_auth()
    await to_thread.run_sync(
        functools.partial(
            upload_folder,
            repo_id=repo,
            folder_path=str(local_dir),
            revision=revision,
            token=load(HfSettings).token.get_secret_value(),
        )
    )


@click.group("hf")
def cli() -> None:
    """Pull pinned HF snapshots and push behind a write-role preflight."""


@cli.command("pull")
@click.argument("repo")
@click.option("--revision", default=None, help="Explicit commit SHA; overrides the REVISIONS pin.")
@json_option
@coro
async def hf_pull(repo: str, *, revision: str | None, as_json: bool) -> None:
    """Download REPO at its pinned revision into the HF cache and print the local path."""
    emit(str(await snapshot(repo, revision=revision)), as_json=as_json)
