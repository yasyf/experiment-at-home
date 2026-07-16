"""Frozen, content-addressed baseline metrics for comparative sweeps."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from athome.research.errors import ResearchError
from athome.research.spec import Comparability
from athome.store import Store

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

UPLIFT_FLOOR = 1e-3

BASELINES_SCHEMA = """
CREATE TABLE IF NOT EXISTS baselines (
    config_hash    TEXT NOT NULL,
    dataset_digest TEXT NOT NULL,
    arm            TEXT NOT NULL,
    metric         REAL NOT NULL,
    PRIMARY KEY (config_hash, dataset_digest, arm)
) WITHOUT ROWID;
"""


class BaselineConflict(ResearchError):
    """A frozen baseline already exists with a different metric."""


@dataclass(slots=True)
class BaselineStore:
    """Content-addressed store of immutable baseline metrics.

    Example:
        >>> async with BaselineStore.open(path) as baselines:
        ...     await baselines.put(key, "candidate", 0.9)
        ...     metric = await baselines.get(key, "candidate")
    """

    store: Store

    @classmethod
    @asynccontextmanager
    async def open(cls, path: Path) -> AsyncIterator[BaselineStore]:
        """Opens the baseline store at ``path``."""
        async with Store.open(path, schema=BASELINES_SCHEMA) as store:
            yield cls(store)

    async def get(self, key: Comparability, arm: str) -> float | None:
        """Returns the metric for ``key`` and ``arm``, or ``None`` when absent."""
        row = await self.store.fetch_one(
            "SELECT metric FROM baselines WHERE config_hash = ? AND dataset_digest = ? AND arm = ?",
            (key.config_hash, key.dataset_digest, arm),
        )
        return float(row["metric"]) if row is not None else None

    async def put(self, key: Comparability, arm: str, metric: float) -> None:
        """Stores ``metric`` once, rejecting changes to an existing baseline."""
        existing = await self.get(key, arm)
        if existing is not None:
            if existing != metric:
                raise BaselineConflict(
                    f"baseline for config {key.config_hash!r}, dataset {key.dataset_digest!r}, "
                    f"and arm {arm!r} is frozen at {existing}, not {metric}"
                )
            return
        await self.store.execute(
            "INSERT INTO baselines(config_hash, dataset_digest, arm, metric) VALUES (?, ?, ?, ?)",
            (key.config_hash, key.dataset_digest, arm, metric),
        )


def uplift(
    candidate: float,
    baseline: float,
    *,
    direction: Literal["min", "max"],
    floor: float = UPLIFT_FLOOR,
) -> float:
    """Returns candidate improvement over baseline, normalized by baseline magnitude.

    Args:
        candidate: Metric produced by the candidate.
        baseline: Frozen metric used as the comparison anchor.
        direction: Whether lower or higher metric values are better.
        floor: Minimum normalization denominator.
    """
    denominator = max(abs(baseline), floor)
    return (candidate - baseline) / denominator if direction == "max" else (baseline - candidate) / denominator
