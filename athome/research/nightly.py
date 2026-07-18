from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from athome import launchd
from athome.research import failures
from athome.research.journal import Journal, Verdict
from athome.research.loop import run_git
from athome.research.spec import ExperimentSpec

if TYPE_CHECKING:
    from athome.research.journal import JournalRow

RESEARCH_LABEL_PREFIX = f"{launchd.LABEL_NAMESPACE}research."
NIGHTLY_CALENDAR = launchd.Calendar(hour=2, minute=0)


@dataclass(frozen=True, slots=True)
class MorningReport:
    """The overnight run's morning summary, read back from the loop's journal.

    Attributes:
        experiment: The experiment name.
        units: How many units the journal recorded.
        kept: How many candidates were kept.
        crashes: How many candidates crashed.
        best: The best kept row for the metric direction, or ``None`` if nothing was kept.
        rows: Every journaled unit, in order.
        infra_retries: How many infra retries the sidecar recorded (``0`` when absent).
        accounting_aborts: How many accounting aborts the sidecar recorded (``0`` when absent).
    """

    experiment: str
    units: int
    kept: int
    crashes: int
    best: JournalRow | None
    rows: tuple[JournalRow, ...]
    infra_retries: int
    accounting_aborts: int


async def repo_root(spec_path: Path) -> Path:
    return Path((await run_git(spec_path.resolve().parent, "rev-parse", "--show-toplevel")).strip())


async def journal_dir(repo: Path) -> Path:
    common = Path((await run_git(repo, "rev-parse", "--git-common-dir")).strip())
    return (common if common.is_absolute() else repo / common) / "athome"


async def journal_path(repo: Path, name: str) -> Path:
    return await journal_dir(repo) / f"{name}.jsonl"


async def install(
    spec_path: Path,
    *,
    calendar: launchd.Calendar = NIGHTLY_CALENDAR,
    mirror_cc_notes: bool = False,
) -> Path:
    """Installs a launchd Calendar agent that runs the experiment's loop overnight.

    The agent runs ``athome research run <spec>`` at ``calendar`` in the experiment's
    git repository; a re-install replaces the existing agent for the same experiment.

    Args:
        spec_path: The experiment TOML the nightly agent runs.
        calendar: When to fire; defaults to 02:00 nightly.
        mirror_cc_notes: Whether to mirror journal rows to the installed ``cc-notes`` service.

    Returns:
        The path of the written launchd plist.
    """
    spec = ExperimentSpec.load(spec_path)
    agent = launchd.AgentSpec(
        label=f"{RESEARCH_LABEL_PREFIX}{spec.name}",
        command=(
            "athome",
            "research",
            "run",
            str(spec_path.resolve()),
            *(("--mirror-cc-notes",) if mirror_cc_notes else ()),
        ),
        schedule=calendar,
        working_dir=await repo_root(spec_path),
    )
    return await launchd.install(agent)


async def report(spec: ExperimentSpec, *, repo: Path) -> MorningReport:
    """Summarizes the overnight loop from its journal: counts, the best kept row, and every unit.

    Args:
        spec: The experiment whose journal and metric direction are summarized.
        repo: The git repository the loop ran against.
    """
    path = await journal_path(repo, spec.name)
    journal = Journal.open(path)
    rows = journal.rows()
    return MorningReport(
        experiment=spec.name,
        units=len(rows),
        kept=sum(row.verdict is Verdict.KEEP for row in rows),
        crashes=sum(row.verdict is Verdict.CRASH for row in rows),
        best=journal.best(spec.direction),
        rows=tuple(rows),
        infra_retries=failures.infra_retries(path.with_name(f"{spec.name}.events.jsonl")),
        accounting_aborts=failures.accounting_aborts(path.with_name(f"{spec.name}.events.jsonl")),
    )
