from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest

from athome.research.baseline import BaselineConflict, BaselineStore, uplift
from athome.research.spec import Comparability

if TYPE_CHECKING:
    from pathlib import Path


async def test_put_then_get_roundtrip(tmp_path: Path) -> None:
    key = Comparability(config_hash="config", dataset_digest="dataset")
    async with BaselineStore.open(tmp_path / "baselines.db") as baselines:
        await baselines.put(key, "candidate", 0.75)
        assert await baselines.get(key, "candidate") == 0.75


async def test_get_returns_none_for_missing_baseline(tmp_path: Path) -> None:
    key = Comparability(config_hash="config", dataset_digest="dataset")
    async with BaselineStore.open(tmp_path / "baselines.db") as baselines:
        assert await baselines.get(key, "candidate") is None


async def test_put_same_metric_is_idempotent(tmp_path: Path) -> None:
    key = Comparability(config_hash="config", dataset_digest="dataset")
    async with BaselineStore.open(tmp_path / "baselines.db") as baselines:
        await baselines.put(key, "candidate", 0.75)
        await baselines.put(key, "candidate", 0.75)
        assert await baselines.get(key, "candidate") == 0.75


async def test_put_rejects_a_different_metric_for_existing_key(tmp_path: Path) -> None:
    key = Comparability(config_hash="config", dataset_digest="dataset")
    async with BaselineStore.open(tmp_path / "baselines.db") as baselines:
        await baselines.put(key, "candidate", 0.75)
        with pytest.raises(BaselineConflict):
            await baselines.put(key, "candidate", 0.8)


@pytest.mark.parametrize(
    ("candidate", "baseline", "direction", "expected"),
    [
        pytest.param(12.0, 10.0, "max", 0.2, id="max-improvement"),
        pytest.param(8.0, 10.0, "min", 0.2, id="min-improvement"),
        pytest.param(0.0015, 0.0005, "max", 1.0, id="near-zero-baseline-uses-floor"),
        pytest.param(9.0, 10.0, "max", -0.1, id="regression"),
    ],
)
def test_uplift(
    candidate: float,
    baseline: float,
    direction: Literal["min", "max"],
    expected: float,
) -> None:
    assert uplift(candidate, baseline, direction=direction) == expected
