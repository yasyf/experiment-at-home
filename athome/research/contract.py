from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from athome.research.spec import ExperimentSpec

BUDGET_LOW_WARNING = (
    "## Budget is nearly exhausted\n"
    "Few work-units remain. Make one high-confidence, minimal change now; do not begin "
    "a large refactor you cannot finish and score before the budget runs out."
)


def bullets(globs: Sequence[str]) -> str:
    return "\n".join(f"- `{glob}`" for glob in globs)


def build_contract(spec: ExperimentSpec, *, budget_low: bool) -> str:
    """Generates the agent instruction contract (the ``program.md`` role) for a proposal.

    The contract states the mutable/immutable manifest (the scoring boundary), the
    proposer's hypothesis when the spec carries one, the metric command and its
    structured JSON file, the keep/discard rule, the simplicity criterion, and —
    when ``budget_low`` — a warning to make a small, finishable change. The agent
    reports its metric only through the JSON file.

    Args:
        spec: The experiment whose manifest, metric, and direction shape the contract.
        budget_low: Whether few work-units (or little wall-clock) remain.
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
    ]
    return "\n\n".join([*sections, BUDGET_LOW_WARNING] if budget_low else sections)
