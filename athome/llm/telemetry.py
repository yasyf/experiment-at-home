from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path


def drifted(prior: str | None, current: str | None) -> bool:
    return prior is not None and current is not None and prior != current


@dataclass(frozen=True, slots=True)
class CallRecord:
    """One LLM call's model, latency, token counts, cost, and served-model / fingerprint identity."""

    model: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    served_model: str | None
    system_fingerprint: str | None


@dataclass(slots=True)
class CallLog:
    """In-memory log of :class:`CallRecord`s with an optional JSONL sink; warns on served-model / fingerprint drift.

    Example:
        >>> log = CallLog()
        >>> log.add(record)
        >>> log.total_usd
    """

    sink: Path | None = None
    records: list[CallRecord] = field(default_factory=list)

    def add(self, record: CallRecord) -> None:
        """Append ``record`` to the log (and the JSONL sink when set), warning on identity drift."""
        self.warn_on_drift(record)
        self.records.append(record)
        if self.sink is not None:
            with self.sink.open("a") as handle:
                handle.write(json.dumps(asdict(record)) + "\n")

    def warn_on_drift(self, record: CallRecord) -> None:
        prior = next((r for r in reversed(self.records) if r.model == record.model), None)
        if prior is None:
            return
        if drifted(prior.served_model, record.served_model):
            logger.warning("served-model drift for {}: {} -> {}", record.model, prior.served_model, record.served_model)
        if drifted(prior.system_fingerprint, record.system_fingerprint):
            logger.warning(
                "system-fingerprint drift for {}: {} -> {}",
                record.model,
                prior.system_fingerprint,
                record.system_fingerprint,
            )

    @property
    def total_usd(self) -> float:
        """The summed ``cost_usd`` of every recorded call."""
        return sum(record.cost_usd for record in self.records)
