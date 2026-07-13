from __future__ import annotations

import json
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from subprocess import PIPE, STDOUT
from typing import TYPE_CHECKING, Literal

import anyio
from loguru import logger

from athome.research.contract import build_contract
from athome.research.gate import monotone_gate
from athome.research.journal import Journal, JournalRow, Verdict

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from athome.research.driver import Driver
    from athome.research.spec import Budget, ExperimentSpec

EXPERIMENT_BRANCH_PREFIX = "athome"
BUDGET_LOW_WALL_FRACTION = 0.75


@dataclass(frozen=True, slots=True)
class LoopResult:
    """The outcome of a greedy keep/discard run.

    Attributes:
        kept: How many journaled units were kept (over the whole, resumable journal).
        best: The best kept row for the metric direction, or ``None`` if nothing was kept.
    """

    kept: int
    best: JournalRow | None


@dataclass(frozen=True, slots=True)
class UnitOutcome:
    verdict: Verdict
    metric: float | None
    commit: str
    description: str


async def run_git(repo: Path, *args: str, check: bool = True) -> str:
    return (await anyio.run_process(["git", "-C", str(repo), *args], check=check)).stdout.decode()


async def rev(repo: Path, ref: str) -> str | None:
    return (await run_git(repo, "rev-parse", "--verify", "--quiet", ref, check=False)).strip() or None


def touches_immutable(path: str, patterns: tuple[str, ...]) -> bool:
    return any(PurePosixPath(path).match(pattern) for pattern in patterns)


async def run_metric(command: tuple[str, ...], workdir: Path, *, hard_kill_s: float | None) -> tuple[int | None, bytes]:
    with anyio.move_on_after(hard_kill_s):
        result = await anyio.run_process(list(command), cwd=str(workdir), check=False, stdout=PIPE, stderr=STDOUT)
        return result.returncode, result.stdout
    return None, b""


async def measure(spec: ExperimentSpec, workdir: Path) -> tuple[float | None, bytes]:
    match await run_metric(spec.metric_command, workdir, hard_kill_s=spec.budget.hard_kill_s):
        case (0, log):
            payload = json.loads(await (anyio.Path(workdir) / spec.metric_file).read_text())
            return float(payload[spec.metric_key]), log
        case (_, log):
            return None, log


def decide(metric: float | None, incumbent: float | None, *, direction: Literal["min", "max"]) -> Verdict:
    match metric:
        case None:
            return Verdict.CRASH
        case _ if monotone_gate(metric, incumbent, direction=direction):
            return Verdict.KEEP
        case _:
            return Verdict.DISCARD


@asynccontextmanager
async def worktree(repo: Path, workdir: Path, base: str) -> AsyncIterator[None]:
    await run_git(repo, "worktree", "add", "--detach", str(workdir), base)
    try:
        yield
    finally:
        await run_git(repo, "worktree", "remove", "--force", str(workdir), check=False)


async def commit_candidate(workdir: Path, description: str) -> str:
    await run_git(
        workdir,
        "-c",
        "user.name=athome",
        "-c",
        "user.email=athome@localhost",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--allow-empty-message",
        "-m",
        description,
    )
    return (await run_git(workdir, "rev-parse", "HEAD")).strip()


async def evaluate_unit(
    spec: ExperimentSpec,
    *,
    repo: Path,
    workdir: Path,
    incumbent: str,
    incumbent_metric: float | None,
    contract: str,
    driver: Driver,
) -> UnitOutcome:
    async with worktree(repo, workdir, incumbent):
        description = await driver.propose(contract, workdir)
        await run_git(workdir, "add", "-A")
        listing = await run_git(workdir, "diff", "--cached", "--name-only", "-z")
        changed = [path for path in listing.split("\0") if path]
        if violated := [path for path in changed if touches_immutable(path, spec.immutable_paths)]:
            reason = f"{description} | ImmutableViolation: {sorted(violated)}"
            return UnitOutcome(Verdict.DISCARD, None, incumbent, reason)
        commit = await commit_candidate(workdir, description)
        metric, log = await measure(spec, workdir)
        logger.debug("unit metric log ({} bytes) captured, withheld from the next contract", len(log))
        return UnitOutcome(decide(metric, incumbent_metric, direction=spec.direction), metric, commit, description)


def budget_low(budget: Budget, *, unit: int, elapsed: float) -> bool:
    return budget.max_units - unit <= 1 or (
        budget.max_wall_s is not None and elapsed >= BUDGET_LOW_WALL_FRACTION * budget.max_wall_s
    )


async def resume(
    repo: Path, branch: str, journal: Journal, *, direction: Literal["min", "max"]
) -> tuple[str, float | None]:
    if (tip := await rev(repo, branch)) is None:
        tip = (await run_git(repo, "rev-parse", "HEAD")).strip()
        await run_git(repo, "branch", branch, tip)
    best = journal.best(direction)
    return tip, best.metric if best is not None else None


async def run(spec: ExperimentSpec, *, driver: Driver, repo: Path) -> LoopResult:
    """Runs the greedy keep/discard loop in git worktrees until the budget is spent.

    Each work-unit isolates a candidate in a detached worktree off the incumbent, lets
    the driver edit the mutable files, and enforces the scoring boundary structurally:
    a ``git diff`` that touches an immutable path is reset and journaled ``DISCARD``
    without being scored. Surviving candidates run ``spec.metric_command`` under the
    ``hard_kill_s`` timeout; the metric is read from ``spec.metric_file`` (never stdout),
    and the run log is captured but withheld from the next contract. The monotone gate
    keeps a strict improvement — committing it onto the experiment branch and advancing
    the incumbent — or discards it. Every unit is journaled before the branch moves, so
    a restart resumes from :meth:`Journal.resume_unit`.

    Args:
        spec: The experiment: metric, direction, budget, and the scoring boundary.
        driver: The proposer that edits mutable files and returns a description.
        repo: The git repository the experiment runs against.

    Returns:
        The kept count and the best kept row over the whole (resumable) journal.
    """
    common = Path((await run_git(repo, "rev-parse", "--git-common-dir")).strip())
    athome_dir = anyio.Path(common if common.is_absolute() else repo / common) / "athome"
    await athome_dir.mkdir(parents=True, exist_ok=True)
    journal = Journal.open(Path(athome_dir) / f"{spec.name}.jsonl")
    branch = f"{EXPERIMENT_BRANCH_PREFIX}/{spec.name}"
    incumbent, incumbent_metric = await resume(repo, branch, journal, direction=spec.direction)

    worktrees = Path(tempfile.mkdtemp(prefix=f"athome-{spec.name}-")).resolve()
    started = time.monotonic()
    try:
        for unit in range(journal.resume_unit(), spec.budget.max_units):
            elapsed = time.monotonic() - started
            if spec.budget.max_wall_s is not None and elapsed >= spec.budget.max_wall_s:
                break
            unit_started = time.monotonic()
            contract = build_contract(spec, budget_low=budget_low(spec.budget, unit=unit, elapsed=elapsed))
            outcome = await evaluate_unit(
                spec,
                repo=repo,
                workdir=worktrees / f"unit-{unit}",
                incumbent=incumbent,
                incumbent_metric=incumbent_metric,
                contract=contract,
                driver=driver,
            )
            await journal.append(
                JournalRow(
                    unit=unit,
                    commit=outcome.commit,
                    metric=outcome.metric,
                    verdict=outcome.verdict,
                    resources={"wall_s": time.monotonic() - unit_started},
                    description=outcome.description,
                )
            )
            if outcome.verdict is Verdict.KEEP:
                await run_git(repo, "branch", "-f", branch, outcome.commit)
                incumbent, incumbent_metric = outcome.commit, outcome.metric
    finally:
        await run_git(repo, "worktree", "prune", check=False)
        shutil.rmtree(worktrees, ignore_errors=True)

    kept = sum(row.verdict is Verdict.KEEP for row in journal.rows())
    return LoopResult(kept=kept, best=journal.best(spec.direction))
