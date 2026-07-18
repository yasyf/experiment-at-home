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
from subprocess import PIPE, STDOUT, CalledProcessError
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import anyio
from loguru import logger

from athome.config import base_environ
from athome.research.common import Hasher
from athome.research.contract import Memory, build_contract
from athome.research.driver import describe_change, read_reported_metric
from athome.research.errors import ResearchError
from athome.research.failures import (
    MAX_INFRA_RETRIES,
    AccountingIntegrityError,
    CandidateFault,
    InfraFailure,
    classify,
    infra_cost,
    infra_log,
    record_accounting_abort,
    record_infra_event,
    safe_describe,
)
from athome.research.gate import immutable_violations, monotone_gate, parse_diff_tree
from athome.research.journal import Journal, JournalRow, Verdict
from athome.research.spec import BudgetExhausted, ConcurrentRun, PoisonedJournal, ProposalTimeout, finite_number

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from athome.research.driver import Driver
    from athome.research.spec import Budget, ExperimentSpec

EXPERIMENT_BRANCH_PREFIX = "athome"
BUDGET_LOW_WALL_FRACTION = 0.75
LOCK_RETRY_DELAY_S = 1.0
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


@dataclass(frozen=True, slots=True)
class Measurement:
    metric: float | None
    log: bytes
    produced: bool


@dataclass(frozen=True, slots=True)
class Candidate:
    commit: str
    description: str
    cost: float


class InvalidBaseline(ResearchError):
    pass


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


def validate_driver_cost(unit: int, cost: object) -> float:
    match cost:
        case bool():
            pass
        case int() | float() as value:
            try:
                converted = float(value)
            except Exception:
                pass
            else:
                if finite_number(converted) and converted >= 0:
                    return converted
    raise AccountingIntegrityError(f"unit {unit}: driver returned invalid cost of type {type(cost).__name__}")


def finite_metric(text: str, key: str) -> float | None:
    try:
        value = json.loads(text)[key]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return float(value) if finite_number(value) else None


async def regular_file(path: anyio.Path) -> bool:
    try:
        info = await path.stat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode)


async def measure(spec: ExperimentSpec, workdir: Path) -> Measurement:
    metric_path = anyio.Path(workdir) / spec.metric_file
    await metric_path.unlink(missing_ok=True)
    returncode, log = await run_metric(spec.metric_command, workdir, hard_kill_s=spec.budget.hard_kill_s)
    produced = await regular_file(metric_path)
    return Measurement(
        finite_metric(await metric_path.read_text(), spec.metric_key) if returncode == 0 and produced else None,
        log,
        produced,
    )


def decide(metric: float | None, incumbent: float | None, *, direction: Literal["min", "max"]) -> Verdict:
    match metric:
        case None:
            return Verdict.CRASH
        case _ if monotone_gate(metric, incumbent, direction=direction):
            return Verdict.KEEP
        case _:
            return Verdict.DISCARD


async def score_commit(spec: ExperimentSpec, *, repo: Path, score_dir: Path, commit: str) -> Measurement:
    await extract_tree(repo, commit, score_dir)
    return await measure(spec, score_dir)


def baseline_digest(spec: ExperimentSpec) -> str:
    return Hasher.digest([spec.metric_command, spec.metric_key, spec.metric_file])


async def read_baseline(path: anyio.Path) -> Baseline | None:
    if not await path.exists():
        return None
    try:
        record = json.loads(await path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise InvalidBaseline(path) from error
    try:
        return Baseline(commit=record["commit"], metric=record["metric"], spec_digest=record["spec_digest"])
    except (KeyError, TypeError) as error:
        raise InvalidBaseline(path) from error


async def stage_candidate(
    spec: ExperimentSpec,
    *,
    repo: Path,
    workdir: Path,
    index: Path,
    incumbent: str,
    label: str,
    cost: float,
    reported: float | None,
) -> UnitOutcome | Candidate:
    await run_git(repo, "read-tree", incumbent, index=index)
    try:
        await run_git(repo, "add", "-A", index=index, work_tree=workdir)
        changes = parse_diff_tree(
            await run_git(repo, "diff-index", "--cached", "--no-renames", "-r", "-z", "--raw", incumbent, index=index)
        )
    except CalledProcessError as exc:
        raise CandidateFault(f"git rejected the staged candidate tree: {exc}") from exc
    description = describe_change(label, spec, changes, reported)
    if not changes:
        return UnitOutcome(Verdict.CRASH, None, incumbent, f"{description} | empty proposal, nothing to commit", cost)
    if violations := immutable_violations(changes, mutable=spec.mutable_paths, immutable=spec.immutable_paths):
        reason = f"{description} | ImmutableViolation: {sorted(violations)}"
        return UnitOutcome(Verdict.DISCARD, None, incumbent, reason, cost)
    try:
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
    except CalledProcessError as exc:
        raise CandidateFault(f"git rejected the candidate commit: {exc}") from exc
    return Candidate(commit=commit, description=description, cost=cost)


async def score_candidate(
    spec: ExperimentSpec,
    *,
    repo: Path,
    score_dir: Path,
    candidate: Candidate,
    incumbent_metric: float | None,
    cost: float,
) -> UnitOutcome:
    measurement = await score_commit(spec, repo=repo, score_dir=score_dir, commit=candidate.commit)
    if measurement.metric is None and not measurement.produced and infra_log(measurement.log):
        raise InfraFailure("scorer produced no metric file and its run log matched an infra marker")
    logger.debug("unit metric log ({} bytes) captured, withheld from the next contract", len(measurement.log))
    return UnitOutcome(
        decide(measurement.metric, incumbent_metric, direction=spec.direction),
        measurement.metric,
        candidate.commit,
        candidate.description,
        cost,
    )


def budget_low(budget: Budget, *, unit: int, elapsed: float) -> bool:
    return budget.max_units - unit <= 1 or (
        budget.max_wall_s is not None and elapsed >= BUDGET_LOW_WALL_FRACTION * budget.max_wall_s
    )


def validate_journal(rows: list[JournalRow]) -> None:
    for row in rows:
        if row.metric is not None and not finite_number(row.metric):
            raise PoisonedJournal(f"unit {row.unit}: non-finite metric {row.metric!r}")
        if "usd" not in row.resources:
            raise PoisonedJournal(f"unit {row.unit}: missing usd")
        if not (finite_number(usd := row.resources["usd"]) and usd >= 0):
            raise PoisonedJournal(f"unit {row.unit}: invalid usd {usd!r}")


@asynccontextmanager
async def experiment_lock(path: Path) -> AsyncIterator[None]:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # The watchdog's momentary LOCK_SH probe can shadow the first attempt; one
            # bounded retry outlasts it, while a genuine run holds LOCK_EX far longer.
            await anyio.sleep(LOCK_RETRY_DELAY_S)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ConcurrentRun(f"another run already holds {path}") from exc
        try:
            await anyio.to_thread.run_sync(os.pwrite, fd, uuid4().hex.encode(), 0)
            await anyio.to_thread.run_sync(os.fsync, fd)
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


async def run_unit(
    spec: ExperimentSpec,
    *,
    unit: int,
    repo: Path,
    worktrees: Path,
    incumbent: str,
    incumbent_metric: float | None,
    contract: str,
    driver: Driver,
    events: Path,
    abort: Path,
    deadline: float | None,
    spent: float,
) -> UnitOutcome | None:
    try:
        return await execute_unit(
            spec,
            unit=unit,
            repo=repo,
            worktrees=worktrees,
            incumbent=incumbent,
            incumbent_metric=incumbent_metric,
            contract=contract,
            driver=driver,
            events=events,
            abort=abort,
            deadline=deadline,
            spent=spent,
        )
    except AccountingIntegrityError as exc:
        await record_accounting_abort(abort, events, unit=unit, reason=safe_describe(exc))
        raise


async def execute_unit(
    spec: ExperimentSpec,
    *,
    unit: int,
    repo: Path,
    worktrees: Path,
    incumbent: str,
    incumbent_metric: float | None,
    contract: str,
    driver: Driver,
    events: Path,
    abort: Path,
    deadline: float | None,
    spent: float,
) -> UnitOutcome | None:
    committed: Candidate | None = None
    billed = False
    if spec.budget.max_usd is not None and spent > spec.budget.max_usd:
        raise BudgetExhausted(f"spend ${spent:.4f} crossed max_usd ${spec.budget.max_usd:.4f} at unit {unit}")
    for attempt in range(MAX_INFRA_RETRIES + 1):
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            return None
        cost = 0.0
        proposal_started = False
        proposal_completed = False
        infra: BaseException | None = None
        recorded = False
        try:
            try:
                with anyio.move_on_after(remaining) as scope:
                    try:
                        if committed is not None:
                            candidate = committed  # a prior attempt already built it: re-score the same commit
                        else:
                            workdir = worktrees / f"unit-{unit}-{attempt}"
                            await extract_tree(repo, incumbent, workdir)
                            proposal_started = True
                            grant = None if spec.budget.max_usd is None else max(spec.budget.max_usd - spent, 0.0)
                            cost = validate_driver_cost(unit, await driver.propose(contract, workdir, budget_usd=grant))
                            proposal_completed = True
                            reported = await read_reported_metric(spec, workdir)
                            await (anyio.Path(workdir) / spec.metric_file).unlink(missing_ok=True)
                            match await stage_candidate(
                                spec,
                                repo=repo,
                                workdir=workdir,
                                index=worktrees / f"unit-{unit}-{attempt}.index",
                                incumbent=incumbent,
                                label=driver.label,
                                cost=cost,
                                reported=reported,
                            ):
                                case UnitOutcome() as terminal:
                                    recorded = True
                                    return terminal  # empty proposal or immutable violation: journaled with its cost
                                case Candidate() as candidate:
                                    committed = candidate
                        outcome = await score_candidate(
                            spec,
                            repo=repo,
                            score_dir=worktrees / f"score-{unit}-{attempt}",
                            candidate=candidate,
                            incumbent_metric=incumbent_metric,
                            cost=0.0 if billed else candidate.cost,
                        )
                        recorded = True
                        return outcome
                    except ProposalTimeout as exc:
                        # A hung proposal killed on timeout still counts its recovered spend.
                        logger.warning("unit {} proposal timed out: {}", unit, exc)
                        outcome = UnitOutcome(
                            Verdict.CRASH,
                            None,
                            incumbent,
                            f"proposal timeout: {exc}",
                            validate_driver_cost(unit, exc.cost),
                        )
                        recorded = True
                        return outcome
                    except Exception as exc:
                        classification = classify(exc)
                        if classification != "accounting" and proposal_started and not proposal_completed:
                            cost = validate_driver_cost(unit, await driver.recover_cost())
                        match classification:
                            case "accounting":
                                recorded = True
                                raise
                            case "candidate":
                                # Candidate crash: journaled with its proposal cost; the loop continues.
                                logger.warning("unit {} crashed: {!r}", unit, exc)
                                crash_cost = 0.0 if billed else committed.cost if committed is not None else cost
                                outcome = UnitOutcome(Verdict.CRASH, None, incumbent, f"crash: {exc!r}", crash_cost)
                                recorded = True
                                return outcome
                            case "infra":
                                infra = exc
                if scope.cancel_called:
                    with anyio.CancelScope(shield=True):
                        if proposal_started and not proposal_completed:
                            cost = validate_driver_cost(unit, await driver.recover_cost())
                        if proposal_completed or cost > 0:
                            await record_infra_event(
                                events,
                                unit=unit,
                                attempt=attempt,
                                reason="wall deadline cancelled after proposal",
                                cost=cost,
                                kind="wall_cancel",
                            )
                    recorded = True
                    return None
                # A committed candidate re-scores (bill its cost once); a pre-commit failure re-proposes.
                attempt_cost = (committed.cost if not billed else 0.0) if committed is not None else cost
                billed = billed or committed is not None
                await record_infra_event(
                    events, unit=unit, attempt=attempt, reason=repr(infra), cost=attempt_cost, kind="retry"
                )
                recorded = True
                spent += attempt_cost
                logger.warning("unit {} infra failure, attempt {}/{}: {!r}", unit, attempt, MAX_INFRA_RETRIES, infra)
                if spec.budget.max_usd is not None and spent > spec.budget.max_usd:
                    raise BudgetExhausted(
                        f"spend ${spent:.4f} crossed max_usd ${spec.budget.max_usd:.4f} at unit {unit}"
                    )
                if attempt == MAX_INFRA_RETRIES:
                    raise InfraFailure(f"unit {unit} aborted after {MAX_INFRA_RETRIES} infra retries") from infra
            except AccountingIntegrityError:
                recorded = True
                raise
        finally:
            if not recorded:
                with anyio.CancelScope(shield=True):
                    try:
                        if proposal_started and not proposal_completed:
                            cost = validate_driver_cost(unit, await driver.recover_cost())
                        if proposal_completed or cost > 0:
                            await record_infra_event(
                                events,
                                unit=unit,
                                attempt=attempt,
                                reason="attempt interrupted after proposal",
                                cost=cost,
                                kind="wall_cancel",
                            )
                    except (Exception, anyio.get_cancelled_exc_class()) as exc:
                        await record_accounting_abort(abort, events, unit=unit, reason=safe_describe(exc))


async def run(spec: ExperimentSpec, *, driver: Driver, repo: Path, mirror_cc_notes: bool = False) -> LoopResult:
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
    advanced, incumbent updated); anything else discards, and a candidate-caused crash or
    over-time unit is journaled and the loop continues — including git rejecting the
    candidate's own tree (a FIFO swapped into a mutable path), which is a candidate fault, not
    machine trouble. An infrastructure failure — an OS or harness-side git error on the
    incumbent, or a scorer that writes no metric while its withheld run log matches an infra
    marker — is never journaled: the same unit index is retried up to ``MAX_INFRA_RETRIES``
    times (a committed candidate re-scores that immutable commit rather than re-proposing) and,
    if it keeps failing, the run aborts loudly with :class:`InfraFailure`, so one flaky night is
    never recorded as failed research directions. Every attempt's proposal spend is accounted —
    the journaled outcome carries its own cost, a retried or aborted attempt records its cost in
    the sidecar — and ``max_usd`` sums both.
    Spend is measured per unit — including the spend recovered from a hung unit killed on
    timeout — and the run aborts when it crosses ``max_usd``; work is bounded within a
    unit by the remaining wall budget. A per-experiment ``flock`` serializes concurrent
    runs, retrying acquisition once after a bounded delay so a momentary watchdog probe
    never masquerades as one. Every unit is journaled, so a restart resumes from :meth:`Journal.resume_unit`
    and reconciles the branch with the (validated) journaled best.

    Args:
        spec: The experiment: metric, direction, budget, and the scoring boundary.
        driver: The proposer that edits mutable files and returns the proposal's cost.
        repo: The git repository the experiment runs against.
        mirror_cc_notes: Whether to mirror journal rows to ``cc-notes``.

    Returns:
        The kept count and the best kept row over the whole (resumable) journal.

    Raises:
        BudgetExhausted: Cumulative spend crossed ``spec.budget.max_usd``.
        AccountingIntegrityError: Proposal spend could not be recovered or trusted.
        ConcurrentRun: Another live run holds the per-experiment lock.
        PoisonedJournal: The resumed journal was unreadable, malformed, or carried an invalid metric or spend.
        InfraFailure: A unit hit machine trouble that outlasted ``MAX_INFRA_RETRIES`` retries;
            the unit is never journaled, so a restart resumes the same unit index.
    """
    from athome.research.preflight import preflight

    common = Path((await run_git(repo, "rev-parse", "--git-common-dir")).strip())
    athome_dir = anyio.Path(common if common.is_absolute() else repo / common) / "athome"
    await athome_dir.mkdir(parents=True, exist_ok=True)
    async with experiment_lock(Path(athome_dir) / f"{spec.name}.lock"):
        journal = Journal.open(Path(athome_dir) / f"{spec.name}.jsonl", mirror_cc_notes=mirror_cc_notes)
        events = Path(athome_dir) / f"{spec.name}.events.jsonl"
        abort = Path(athome_dir) / f"{spec.name}.abort.json"
        if abort.exists():
            raise AccountingIntegrityError(
                f"unreconciled accounting abort latch in {abort}; check the provider ledger, reconcile the spend, "
                "then delete the latch file"
            )
        validate_journal(journal.rows())
        branch = f"{EXPERIMENT_BRANCH_PREFIX}/{spec.name}"
        incumbent, incumbent_metric = await resume(repo, branch, journal, direction=spec.direction)
        resumed = bool(journal.rows())
        spent = sum(row.resources["usd"] for row in journal.rows()) + infra_cost(events)

        worktrees = Path(tempfile.mkdtemp(prefix=f"athome-{spec.name}-")).resolve()
        try:
            await driver.preflight()
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
            deadline = None if spec.budget.max_wall_s is None else started + spec.budget.max_wall_s
            for unit in range(journal.resume_unit(), spec.budget.max_units):
                elapsed = time.monotonic() - started
                if spec.budget.max_wall_s is not None and elapsed >= spec.budget.max_wall_s:
                    break
                unit_started = time.monotonic()
                contract = build_contract(
                    spec,
                    budget_low=budget_low(spec.budget, unit=unit, elapsed=elapsed),
                    memory=Memory.from_journal(
                        journal, baseline=baseline, incumbent=incumbent_metric, direction=spec.direction
                    ),
                )
                outcome = await run_unit(
                    spec,
                    unit=unit,
                    repo=repo,
                    worktrees=worktrees,
                    incumbent=incumbent,
                    incumbent_metric=incumbent_metric,
                    contract=contract,
                    driver=driver,
                    events=events,
                    abort=abort,
                    deadline=deadline,
                    spent=spent,
                )
                if outcome is None:
                    break  # the unit blew past the remaining wall budget and was cancelled
                row = JournalRow(
                    unit=unit,
                    commit=outcome.commit,
                    metric=outcome.metric,
                    verdict=outcome.verdict,
                    resources={"wall_s": time.monotonic() - unit_started, "usd": outcome.cost},
                    description=outcome.description,
                )
                try:
                    await journal.append(row)
                except OSError as exc:
                    reason = f"could not append journal row for unit {unit} to {journal.sink.path}"
                    await record_accounting_abort(abort, events, unit=unit, reason=reason)
                    raise AccountingIntegrityError(reason) from exc
                if outcome.verdict is Verdict.KEEP:
                    await run_git(repo, "branch", "-f", branch, outcome.commit)
                    incumbent, incumbent_metric = outcome.commit, outcome.metric
                spent = sum(row.resources["usd"] for row in journal.rows()) + infra_cost(events)
                if spec.budget.max_usd is not None and spent > spec.budget.max_usd:
                    raise BudgetExhausted(
                        f"spend ${spent:.4f} crossed max_usd ${spec.budget.max_usd:.4f} at unit {unit}"
                    )
        finally:
            shutil.rmtree(worktrees, ignore_errors=True)

        kept = sum(row.verdict is Verdict.KEEP for row in journal.rows())
        return LoopResult(kept=kept, best=journal.best(spec.direction))
