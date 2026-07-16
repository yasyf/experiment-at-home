from __future__ import annotations

import fcntl
import io
import json
import os
import shutil
import stat
import tarfile
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from subprocess import PIPE, STDOUT
from typing import TYPE_CHECKING, Literal

import anyio
from loguru import logger

from athome.config import base_environ
from athome.research.common import Hasher
from athome.research.contract import Memory, build_contract
from athome.research.driver import describe_change, read_reported_metric
from athome.research.gate import immutable_violations, monotone_gate, parse_diff_tree
from athome.research.journal import Journal, JournalRow, Verdict
from athome.research.spec import BudgetExhausted, ConcurrentRun, PoisonedJournal, ProposalTimeout, finite_number

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from athome.research.driver import Driver
    from athome.research.spec import Budget, ExperimentSpec

EXPERIMENT_BRANCH_PREFIX = "athome"
BUDGET_LOW_WALL_FRACTION = 0.75
HERMETIC_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}


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
    cost: float


@dataclass(frozen=True, slots=True)
class Baseline:
    commit: str
    metric: float | None
    spec_digest: str


def hermetic_env() -> dict[str, str]:
    return base_environ() | HERMETIC_ENV


async def run_git(
    repo: Path, *args: str, check: bool = True, index: Path | None = None, work_tree: Path | None = None
) -> str:
    prefix = ["git", "-c", "core.hooksPath=/dev/null", "-C", str(repo)]
    if work_tree is not None:
        prefix += ["--work-tree", str(work_tree)]
    env = hermetic_env() | ({"GIT_INDEX_FILE": str(index)} if index is not None else {})
    return (await anyio.run_process([*prefix, *args], check=check, env=env)).stdout.decode()


def _extract_tar(archive: bytes, dest: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")


async def extract_tree(repo: Path, treeish: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    command = ["git", "-c", "core.hooksPath=/dev/null", "-C", str(repo), "archive", "--format=tar", treeish]
    archive = (await anyio.run_process(command, env=hermetic_env())).stdout
    await anyio.to_thread.run_sync(partial(_extract_tar, archive, dest))


async def rev(repo: Path, ref: str) -> str | None:
    return (await run_git(repo, "rev-parse", "--verify", "--quiet", ref, check=False)).strip() or None


async def run_metric(command: tuple[str, ...], workdir: Path, *, hard_kill_s: float | None) -> tuple[int | None, bytes]:
    with anyio.move_on_after(hard_kill_s):
        result = await anyio.run_process(
            list(command), cwd=str(workdir), check=False, stdout=PIPE, stderr=STDOUT, env=hermetic_env()
        )
        return result.returncode, result.stdout
    return None, b""



def finite_metric(text: str, key: str) -> float | None:
    try:
        value = json.loads(text)[key]
    except (json.JSONDecodeError, KeyError):
        return None
    return float(value) if finite_number(value) else None


async def regular_file(path: anyio.Path) -> bool:
    try:
        info = await path.stat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode)


async def measure(spec: ExperimentSpec, workdir: Path) -> tuple[float | None, bytes]:
    metric_path = anyio.Path(workdir) / spec.metric_file
    await metric_path.unlink(missing_ok=True)
    returncode, log = await run_metric(spec.metric_command, workdir, hard_kill_s=spec.budget.hard_kill_s)
    if returncode != 0 or not await regular_file(metric_path):
        return None, log
    return finite_metric(await metric_path.read_text(), spec.metric_key), log


def decide(metric: float | None, incumbent: float | None, *, direction: Literal["min", "max"]) -> Verdict:
    match metric:
        case None:
            return Verdict.CRASH
        case _ if monotone_gate(metric, incumbent, direction=direction):
            return Verdict.KEEP
        case _:
            return Verdict.DISCARD


async def score_commit(spec: ExperimentSpec, *, repo: Path, score_dir: Path, commit: str) -> tuple[float | None, bytes]:
    await extract_tree(repo, commit, score_dir)
    return await measure(spec, score_dir)


def baseline_digest(spec: ExperimentSpec) -> str:
    return Hasher.digest([spec.metric_command, spec.metric_key, spec.metric_file])


async def read_baseline(path: anyio.Path) -> Baseline | None:
    if not await path.exists():
        return None
    record = json.loads(await path.read_text())
    return Baseline(commit=record["commit"], metric=record["metric"], spec_digest=record["spec_digest"])


async def evaluate_unit(
    spec: ExperimentSpec,
    *,
    repo: Path,
    workdir: Path,
    score_dir: Path,
    index: Path,
    incumbent: str,
    incumbent_metric: float | None,
    contract: str,
    driver: Driver,
) -> UnitOutcome:
    await extract_tree(repo, incumbent, workdir)
    cost = await driver.propose(contract, workdir)
    reported = await read_reported_metric(spec, workdir)
    await (anyio.Path(workdir) / spec.metric_file).unlink(missing_ok=True)
    await run_git(repo, "read-tree", incumbent, index=index)
    await run_git(repo, "add", "-A", index=index, work_tree=workdir)
    changes = parse_diff_tree(
        await run_git(repo, "diff-index", "--cached", "--no-renames", "-r", "-z", "--raw", incumbent, index=index)
    )
    description = describe_change(driver.label, spec, changes, reported)
    if not changes:
        return UnitOutcome(Verdict.CRASH, None, incumbent, f"{description} | empty proposal, nothing to commit", cost)
    if violations := immutable_violations(changes, mutable=spec.mutable_paths, immutable=spec.immutable_paths):
        reason = f"{description} | ImmutableViolation: {sorted(violations)}"
        return UnitOutcome(Verdict.DISCARD, None, incumbent, reason, cost)
    tree = (await run_git(repo, "write-tree", index=index)).strip()
    commit = (
        await run_git(
            repo,
            "-c",
            "user.name=athome",
            "-c",
            "user.email=athome@localhost",
            "commit-tree",
            tree,
            "-p",
            incumbent,
            "-m",
            description,
            index=index,
        )
    ).strip()
    metric, log = await score_commit(spec, repo=repo, score_dir=score_dir, commit=commit)
    logger.debug("unit metric log ({} bytes) captured, withheld from the next contract", len(log))
    return UnitOutcome(decide(metric, incumbent_metric, direction=spec.direction), metric, commit, description, cost)


def budget_low(budget: Budget, *, unit: int, elapsed: float) -> bool:
    return budget.max_units - unit <= 1 or (
        budget.max_wall_s is not None and elapsed >= BUDGET_LOW_WALL_FRACTION * budget.max_wall_s
    )


def validate_journal(rows: list[JournalRow]) -> None:
    for row in rows:
        if row.metric is not None and not finite_number(row.metric):
            raise PoisonedJournal(f"unit {row.unit}: non-finite metric {row.metric!r}")
        if not (finite_number(usd := row.resources.get("usd", 0.0)) and usd >= 0):
            raise PoisonedJournal(f"unit {row.unit}: invalid usd {usd!r}")


@asynccontextmanager
async def experiment_lock(path: Path) -> AsyncIterator[None]:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConcurrentRun(f"another run already holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


async def resume(
    repo: Path, branch: str, journal: Journal, *, direction: Literal["min", "max"]
) -> tuple[str, float | None]:
    if (best := journal.best(direction)) is not None:
        await run_git(repo, "branch", "-f", branch, best.commit)
        return best.commit, best.metric
    if (tip := await rev(repo, branch)) is None:
        tip = (await run_git(repo, "rev-parse", "HEAD")).strip()
        await run_git(repo, "branch", branch, tip)
    return tip, None


async def run(spec: ExperimentSpec, *, driver: Driver, repo: Path) -> LoopResult:
    """Runs the greedy keep/discard loop in throwaway plain checkouts until the budget is spent.

    Each work-unit materializes the incumbent into a plain directory via ``git archive``
    (no ``.git``), lets the driver edit the mutable files, and stages the result into the
    trusted store with a throwaway index — so no candidate git config is ever shared with
    the harness. Immutability is enforced structurally by diffing the incumbent tree
    against that staged index: every changed path must fall inside ``mutable_paths``,
    outside ``immutable_paths``, and never be a symlink or a Python auto-loader (a rename
    or deletion of an immutable file, an undeclared new file, or an added ``conftest.py``
    is rejected), else the candidate is journaled ``DISCARD`` without being scored.
    Surviving candidates are committed with ``commit-tree`` plumbing and scored from a
    *plain checkout materialized via ``git archive``* of the commit — only committed
    content runs, under a hard-kill timeout and hermetic git config — with the metric
    read from a regular ``spec.metric_file`` (freshly written by that run, never stdout,
    never a candidate-planted FIFO) and the run log withheld from the next contract. On a
    fresh run the untouched incumbent tree is scored once into a frozen baseline —
    persisted to ``<git-common>/athome/<name>.baseline.json`` and reused only when the
    incumbent commit and the scorer digest both match — which seeds the incumbent metric,
    so even the first candidate must *strictly beat the untouched tree* rather than being
    auto-kept. Each contract threads a harness-authored ``## History`` (baseline, current
    incumbent, best-so-far, and the most recent units with their verdicts and discard
    reasons), never the withheld run log. A strict monotone improvement is kept (branch
    advanced, incumbent updated); anything else discards, and a crash or over-time unit is
    journaled and the loop continues.
    Spend is measured per unit — including the spend recovered from a hung unit killed on
    timeout — and the run aborts when it crosses ``max_usd``; work is bounded within a
    unit by the remaining wall budget. A per-experiment ``flock`` serializes concurrent
    runs. Every unit is journaled, so a restart resumes from :meth:`Journal.resume_unit`
    and reconciles the branch with the (validated) journaled best.

    Args:
        spec: The experiment: metric, direction, budget, and the scoring boundary.
        driver: The proposer that edits mutable files and returns the proposal's cost.
        repo: The git repository the experiment runs against.

    Returns:
        The kept count and the best kept row over the whole (resumable) journal.

    Raises:
        BudgetExhausted: Cumulative spend crossed ``spec.budget.max_usd``.
        ConcurrentRun: Another live run holds the per-experiment lock.
        PoisonedJournal: The resumed journal carried a non-finite metric or bad spend.
    """
    from athome.research.preflight import preflight

    common = Path((await run_git(repo, "rev-parse", "--git-common-dir")).strip())
    athome_dir = anyio.Path(common if common.is_absolute() else repo / common) / "athome"
    await athome_dir.mkdir(parents=True, exist_ok=True)
    async with experiment_lock(Path(athome_dir) / f"{spec.name}.lock"):
        journal = Journal.open(Path(athome_dir) / f"{spec.name}.jsonl")
        validate_journal(journal.rows())
        branch = f"{EXPERIMENT_BRANCH_PREFIX}/{spec.name}"
        incumbent, incumbent_metric = await resume(repo, branch, journal, direction=spec.direction)
        resumed = bool(journal.rows())
        spent = sum(row.resources.get("usd", 0.0) for row in journal.rows())

        worktrees = Path(tempfile.mkdtemp(prefix=f"athome-{spec.name}-")).resolve()
        try:
            report = await preflight(
                spec,
                repo=repo,
                incumbent=incumbent,
                scratch_dir=worktrees / "preflight",
                baseline_path=Path(athome_dir) / f"{spec.name}.baseline.json",
                resume=resumed,
            )
            if incumbent_metric is None:
                incumbent_metric = report.baseline
            baseline = report.baseline
            started = time.monotonic()
            for unit in range(journal.resume_unit(), spec.budget.max_units):
                elapsed = time.monotonic() - started
                if spec.budget.max_wall_s is not None and elapsed >= spec.budget.max_wall_s:
                    break
                remaining = None if spec.budget.max_wall_s is None else spec.budget.max_wall_s - elapsed
                unit_started = time.monotonic()
                contract = build_contract(
                    spec,
                    budget_low=budget_low(spec.budget, unit=unit, elapsed=elapsed),
                    memory=Memory.from_journal(
                        journal, baseline=baseline, incumbent=incumbent_metric, direction=spec.direction
                    ),
                )
                outcome: UnitOutcome | None = None
                with anyio.move_on_after(remaining):
                    try:
                        outcome = await evaluate_unit(
                            spec,
                            repo=repo,
                            workdir=worktrees / f"unit-{unit}",
                            score_dir=worktrees / f"score-{unit}",
                            index=worktrees / f"unit-{unit}.index",
                            incumbent=incumbent,
                            incumbent_metric=incumbent_metric,
                            contract=contract,
                            driver=driver,
                        )
                    except ProposalTimeout as exc:
                        # A hung proposal killed on timeout still counts its recovered spend.
                        logger.warning("unit {} proposal timed out: {}", unit, exc)
                        outcome = UnitOutcome(Verdict.CRASH, None, incumbent, f"proposal timeout: {exc}", exc.cost)
                    except Exception as exc:
                        # The overnight contract: a candidate-caused crash is journaled and the
                        # loop continues, it never aborts the run (BudgetExhausted still propagates).
                        logger.warning("unit {} crashed: {!r}", unit, exc)
                        outcome = UnitOutcome(Verdict.CRASH, None, incumbent, f"crash: {exc!r}", 0.0)
                if outcome is None:
                    break  # the unit blew past the remaining wall budget and was cancelled
                await journal.append(
                    JournalRow(
                        unit=unit,
                        commit=outcome.commit,
                        metric=outcome.metric,
                        verdict=outcome.verdict,
                        resources={"wall_s": time.monotonic() - unit_started, "usd": outcome.cost},
                        description=outcome.description,
                    )
                )
                if outcome.verdict is Verdict.KEEP:
                    await run_git(repo, "branch", "-f", branch, outcome.commit)
                    incumbent, incumbent_metric = outcome.commit, outcome.metric
                spent += outcome.cost
                if spec.budget.max_usd is not None and spent > spec.budget.max_usd:
                    raise BudgetExhausted(
                        f"spend ${spent:.4f} crossed max_usd ${spec.budget.max_usd:.4f} at unit {unit}"
                    )
        finally:
            shutil.rmtree(worktrees, ignore_errors=True)

        kept = sum(row.verdict is Verdict.KEEP for row in journal.rows())
        return LoopResult(kept=kept, best=journal.best(spec.direction))
