from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from athome import hf
from athome.config import load
from athome.hf import REVISIONS, HfAuthError, HfError, ensure_write_auth, push, snapshot

if TYPE_CHECKING:
    from collections.abc import Iterator

PINNED_REPO = "StyleDistance/styledistance"
PINNED_SHA = REVISIONS[PINNED_REPO]
SNAPSHOT_PATH = "/hf/cache/models--StyleDistance--styledistance/snapshots/abc123"


def whoami_response(role: str) -> dict[str, object]:
    return {
        "type": "user",
        "name": "yasyf",
        "auth": {"type": "access_token", "accessToken": {"displayName": "t", "role": role}},
    }


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.setenv("HF_TOKEN", "tok_write")
    load.cache_clear()
    module = ModuleType("huggingface_hub")
    module.snapshot_download = MagicMock(return_value=SNAPSHOT_PATH)
    module.whoami = MagicMock(return_value=whoami_response("write"))
    module.upload_folder = MagicMock(return_value=None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    yield module


async def test_snapshot_uses_pinned_revision(fake_hub: ModuleType) -> None:
    assert await snapshot(PINNED_REPO) == Path(SNAPSHOT_PATH)
    fake_hub.snapshot_download.assert_called_once_with(PINNED_REPO, revision=PINNED_SHA, token="tok_write")


async def test_snapshot_explicit_revision_overrides_pin(fake_hub: ModuleType) -> None:
    await snapshot("unpinned/repo", revision="deadbeef")
    fake_hub.snapshot_download.assert_called_once_with("unpinned/repo", revision="deadbeef", token="tok_write")


async def test_snapshot_unpinned_repo_raises(fake_hub: ModuleType) -> None:
    with pytest.raises(HfError):
        await snapshot("unpinned/repo")
    fake_hub.snapshot_download.assert_not_called()


async def test_ensure_write_auth_accepts_write_token(fake_hub: ModuleType) -> None:
    await ensure_write_auth()
    fake_hub.whoami.assert_called_once_with(token="tok_write")


async def test_ensure_write_auth_rejects_read_token(fake_hub: ModuleType) -> None:
    fake_hub.whoami.return_value = whoami_response("read")
    with pytest.raises(HfAuthError):
        await ensure_write_auth()


async def test_push_read_token_fails_preflight_before_upload(fake_hub: ModuleType, tmp_path: Path) -> None:
    fake_hub.whoami.return_value = whoami_response("read")
    with pytest.raises(HfAuthError):
        await push("me/repo", tmp_path)
    fake_hub.upload_folder.assert_not_called()


async def test_push_write_token_uploads_after_preflight(fake_hub: ModuleType, tmp_path: Path) -> None:
    await push("me/repo", tmp_path, revision="main")
    fake_hub.whoami.assert_called_once_with(token="tok_write")
    fake_hub.upload_folder.assert_called_once_with(
        repo_id="me/repo", folder_path=str(tmp_path), revision="main", token="tok_write"
    )


async def test_missing_token_raises(fake_hub: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    load.cache_clear()
    with pytest.raises(ValidationError):
        await snapshot(PINNED_REPO)
    fake_hub.snapshot_download.assert_not_called()


def test_cli_pull_prints_snapshot_path(fake_hub: ModuleType) -> None:
    result = CliRunner().invoke(hf.cli, ["pull", PINNED_REPO])
    assert result.exit_code == 0
    assert SNAPSHOT_PATH in result.output
    fake_hub.snapshot_download.assert_called_once_with(PINNED_REPO, revision=PINNED_SHA, token="tok_write")


@pytest.mark.live
async def test_snapshot_live_public_repo() -> None:
    assert (await snapshot("hf-internal-testing/tiny-random-bert", revision="main")).exists()
