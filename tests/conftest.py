from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from athome.config import load

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def isolate_athome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr("athome.config.CONFIG_FILE", tmp_path / "config.toml")
    monkeypatch.setenv("ATHOME_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("ATHOME_LOGS_ROOT", str(tmp_path / "logs"))
    monkeypatch.setenv("ATHOME_BATCHES_ROOT", str(tmp_path / "batches"))
    load.cache_clear()
    yield
    load.cache_clear()
