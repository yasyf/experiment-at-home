from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    from athome.research.journal import Journal, JournalRow
    from athome.research.spec import ExperimentSpec

RECENT_UNITS = 10

HISTORY_LINE_MAX = 240
NEWLINE_PLACEHOLDER = "⏎"

BUDGET_LOW_WARNING = (
    "## Budget is nearly exhausted\n"
    "Few work-units remain. Make one high-confidence, minimal change now; do not begin "
    "a large refactor you cannot finish and score before the budget runs out."
)


def bullets(globs: Sequence[str]) -> str:
    return "\n".join(f"- `{glob}`" for glob in globs)


def show_metric(value: float | None) -> str:
    return "unscored" if value is None else str(value)


def sanitize_history(text: str) -> str:
    clean = "".join(
        NEWLINE_PLACEHOLDER if ch in "\r\n" else " " if unicodedata.category(ch) == "Cc" else "'" if ch == "`" else ch
        for ch in text
    )
    return clean if len(clean) <= HISTORY_LINE_MAX else f"{clean[: HISTORY_LINE_MAX - 1]}…"


def history_line(row: JournalRow) -> str:
    return sanitize_history(
        f"unit {row.unit} [{row.verdict.value}] metric={show_metric(row.metric)} — {row.description}"
    )


def render_history(memory: Memory) -> str:
    return "\n".join(
        [
            "## History",
            "Prior attempts, oldest first. A `discard` line lost to the incumbent — learn from it, do not repeat it.",
            f"- baseline (untouched tree): {show_metric(memory.baseline)}",
            f"- incumbent: {show_metric(memory.incumbent)}",
            f"- best kept: {'none' if memory.best is None else show_metric(memory.best.metric)}",
            *(history_line(row) for row in memory.recent),
        ]
    )


@dataclass(frozen=True, slots=True)
class Memory:
    """The harness-authored history threaded into the next proposal's contract.

    Every field is written by the harness, never by candidate code: the baseline and
    metrics come from the structured metric file (the run log is withheld), and each
    recent row's description is the trusted tree-diff summary or a harness-written
    crash or immutability reason. Rendering it into a contract therefore cannot leak
    the untrusted run log back to the agent, while the discard reasons it does carry
    are the deliberate "why my last attempt failed" feedback.

    Attributes:
        baseline: The untouched incumbent tree's frozen score, or ``None`` when unscored.
        incumbent: The current incumbent's metric, or ``None`` before the first keep.
        best: The best kept row for the metric direction, or ``None`` if nothing was kept.
        recent: The last ``RECENT_UNITS`` journaled rows, oldest first.
    """

    baseline: float | None
    incumbent: float | None
    best: JournalRow | None
    recent: tuple[JournalRow, ...]

    @classmethod
    def from_journal(
        cls, journal: Journal, *, baseline: float | None, incumbent: float | None, direction: Literal["min", "max"]
    ) -> Memory:
        """Builds the history view: best-so-far plus the last ``RECENT_UNITS`` journaled rows, oldest first."""
        return cls(
            baseline=baseline,
            incumbent=incumbent,
            best=journal.best(direction),
            recent=tuple(journal.rows()[-RECENT_UNITS:]),
        )


def build_contract(spec: ExperimentSpec, *, budget_low: bool, memory: Memory) -> str:
    """Generates the agent instruction contract (the ``program.md`` role) for a proposal.

    The contract states the mutable/immutable manifest (the scoring boundary), the
    proposer's hypothesis when the spec carries one, the metric command and its
    structured JSON file, the keep/discard rule, the simplicity criterion, a
    ``## History`` section of prior units (baseline, current incumbent, best so far,
    and the most recent attempts with their verdicts — discards included as failure
    feedback), and — when ``budget_low`` — a warning to make a small, finishable
    change. Every history field is harness-authored, so the withheld metric run log
    never leaks back to the agent. The agent reports its own metric only through the
    JSON file.

    Args:
        spec: The experiment whose manifest, metric, and direction shape the contract.
        budget_low: Whether few work-units (or little wall-clock) remain.
        memory: The harness-authored history rendered into the ``## History`` section.
    """
    match spec.direction:
        case "min":
            goal, beats = "minimize", "must fall strictly below"
        case "max":
            goal, beats = "maximize", "must strictly exceed"
    sections = [
        f"# Experiment: {spec.name}",
        f"Improve the system to **{goal}** `{spec.metric_key}`. Change only what earns a better metric.",
        *([f"## Hypothesis\n{spec.hypothesis}"] if spec.hypothesis else []),
        f"## Files you MAY edit\n{bullets(spec.mutable_paths)}",
        f"## Files you MUST NOT edit (the scoring boundary)\n{bullets(spec.immutable_paths)}\n"
        "Any proposal that changes an immutable path is discarded without being scored.",
        f"## Metric\nRun `{' '.join(spec.metric_command)}`. It writes `{spec.metric_file}`, a JSON file; "
        f"your score is its `{spec.metric_key}` field. Report the metric only through that file — "
        "never print it to stdout or rely on the run log.",
        f"## Keep or discard\nA proposal is kept only if its `{spec.metric_key}` {beats} the incumbent's; "
        "otherwise it is discarded and the incumbent stands.",
        "## Simplicity\nPrefer the smallest change that moves the metric. Do not add dead code, speculative "
        "abstraction, or complexity; at an equal metric the simpler proposal wins.",
        render_history(memory),
    ]
    return "\n\n".join([*sections, BUDGET_LOW_WARNING] if budget_low else sections)
