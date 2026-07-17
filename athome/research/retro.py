"""Post-experiment retrospective generation and durable records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple

import anyio
from pydantic import BaseModel, ConfigDict, Field

from athome.progress import RunSink, load_journal
from athome.research.errors import ResearchError
from athome.research.journal import CC_NOTES_BIN, CC_NOTES_LABEL, Verdict
from athome.research.judge import with_backoff

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from spawnllm import TModel
    from spawnllm.backends.base import LlmBackend

    from athome.research.journal import JournalRow
    from athome.research.nightly import MorningReport


class RetroError(ResearchError):
    """A retrospective cannot be generated from the supplied experiment evidence."""


class RetroVerdict(BaseModel):
    """The evidence-bound retrospective returned by structured extraction."""

    outcome: Literal["improved", "flat", "inconclusive"] = Field(
        description="Overall experiment outcome supported by the supplied numeric evidence."
    )
    summary: str = Field(description="Concise account of the experiment outcome without unsupported causal claims.")
    evidence: tuple[str, ...] = Field(description="Numeric or verdict-pattern observations supporting the summary.")
    next_steps: tuple[str, ...] = Field(description="Concrete follow-up experiments justified by the evidence.")


class RetroRecordSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    experiment: str
    baseline: float
    best_metric: float
    uplift: float
    verdict: RetroVerdict


class _MetricSummary(NamedTuple):
    baseline: float
    best_metric: float
    uplift: float


def _metric_summary(rows: Sequence[JournalRow], report: MorningReport) -> _MetricSummary:
    kept_metrics = tuple(row.metric for row in rows if row.verdict is Verdict.KEEP and row.metric is not None)
    match kept_metrics, report.best.metric if report.best is not None else None:
        case (), _:
            raise RetroError("a retrospective requires at least one kept metric")
        case _, None:
            raise RetroError("a retrospective requires a best kept metric")
        case (baseline, *_), best_metric:
            return _MetricSummary(baseline, best_metric, abs(best_metric - baseline))


def build_prompt(rows: Sequence[JournalRow], report: MorningReport, *, baseline: float, uplift: float) -> str:
    """Build a retrospective prompt from an allowlisted projection of experiment evidence.

    Args:
        rows: Ordered journal rows. Only unit, metric, and verdict are projected.
        report: Morning report. Only aggregate numeric counts are projected.
        baseline: First kept metric for the experiment.
        uplift: Absolute distance between the baseline and best kept metric.
    """
    evidence = json.dumps(
        {
            "report": {"units": report.units, "kept": report.kept, "crashes": report.crashes},
            "metrics": {"baseline": baseline, "uplift": uplift},
            "rows": [{"unit": row.unit, "metric": row.metric, "verdict": row.verdict.value} for row in rows],
        },
        allow_nan=False,
        sort_keys=True,
    )
    return "\n\n".join(
        [
            "# Post-experiment retrospective",
            "Use only the structured evidence below. Describe observable metric and verdict patterns; "
            "do not invent implementation details, quote logs, or make unsupported causal claims. "
            "The uplift is an absolute magnitude because metric direction is not part of this report.",
            f"## Structured evidence\n{evidence}",
            "Return an outcome, a concise summary, supporting evidence, and concrete next experiments.",
        ]
    )


async def generate(
    rows: Sequence[JournalRow],
    report: MorningReport,
    *,
    backend: LlmBackend | str,
    tier: TModel = "large",
    timeout: int = 240,
    label: str = "retro",
) -> RetroVerdict:
    """Generate an informational retrospective on a directly bound backend.

    Args:
        rows: Ordered journal rows used for the numeric trajectory.
        report: Aggregate report stats and selected best kept row.
        backend: An ``LlmBackend`` instance, or a spawnllm backend registry name.
        tier: Abstract spawnllm model tier to request.
        timeout: Seconds before the backend call is killed.
        label: Retry label used by the shared backoff wrapper.

    Raises:
        RetroError: The evidence contains no kept metric to compare.
        JudgeError: The backend call exhausts its retries.
    """
    from spawnllm import extract
    from spawnllm.backends.registry import BACKENDS_BY_NAME

    metrics = _metric_summary(rows, report)
    bound = BACKENDS_BY_NAME[backend] if isinstance(backend, str) else backend
    prompt = build_prompt(rows, report, baseline=metrics.baseline, uplift=metrics.uplift)
    return await with_backoff(
        lambda: extract(prompt, RetroVerdict, backend=bound, model=tier, timeout=timeout),
        label=label,
    )


@dataclass(frozen=True, slots=True)
class RetroRecord:
    """One durable retrospective and its numeric comparison context."""

    experiment: str
    baseline: float
    best_metric: float
    uplift: float
    verdict: RetroVerdict

    @classmethod
    def from_report(cls, report: MorningReport, verdict: RetroVerdict) -> RetroRecord:
        """Create a record from a morning report and extracted verdict."""
        metrics = _metric_summary(report.rows, report)
        return cls(report.experiment, metrics.baseline, metrics.best_metric, metrics.uplift, verdict)

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> RetroRecord:
        """Load one strict-schema persisted record."""
        validated = RetroRecordSchema.model_validate(record, extra="forbid")
        return cls(
            experiment=validated.experiment,
            baseline=validated.baseline,
            best_metric=validated.best_metric,
            uplift=validated.uplift,
            verdict=validated.verdict,
        )

    def to_record(self) -> dict[str, object]:
        """Return the JSON-serializable persistence record."""
        return {
            "experiment": self.experiment,
            "baseline": self.baseline,
            "best_metric": self.best_metric,
            "uplift": self.uplift,
            "verdict": self.verdict.model_dump(mode="json"),
        }


@dataclass(slots=True)
class RetroJournal:
    """Crash-safe retrospective records with cc-notes mirroring enabled by default."""

    sink: RunSink
    mirror_cc_notes: bool
    _records: list[RetroRecord]

    @classmethod
    def open(cls, path: Path, *, mirror_cc_notes: bool = True) -> RetroJournal:
        """Open a retrospective journal, enabling cc-notes mirroring by default."""
        return cls(
            RunSink.open(path),
            mirror_cc_notes,
            [RetroRecord.from_record(record) for record in load_journal(path)],
        )

    async def append(self, record: RetroRecord) -> None:
        """Durably append a retrospective, then mirror it to cc-notes when enabled."""
        await self.sink.append(record.to_record())
        self._records.append(record)
        if self.mirror_cc_notes:
            await self._mirror(record)

    def records(self) -> list[RetroRecord]:
        """Return a snapshot of the persisted retrospective records."""
        return list(self._records)

    async def _mirror(self, record: RetroRecord) -> None:
        await anyio.run_process(
            [
                str(CC_NOTES_BIN),
                "note",
                "add",
                f"athome retro {record.experiment} [{record.verdict.outcome}]",
                "--body",
                "\n".join(
                    [
                        f"baseline {record.baseline}",
                        f"best metric {record.best_metric}",
                        f"uplift {record.uplift}",
                        f"summary {record.verdict.summary}",
                        f"evidence {json.dumps(record.verdict.evidence)}",
                        f"next steps {json.dumps(record.verdict.next_steps)}",
                    ]
                ),
                "--label",
                CC_NOTES_LABEL,
            ]
        )
