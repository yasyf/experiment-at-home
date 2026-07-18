from __future__ import annotations

import json
import os
import re
import shlex
import signal
from dataclasses import dataclass, field
from math import isfinite
from subprocess import PIPE, STDOUT
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

import anyio

from athome.detach import launch, run_exitfile, run_log, run_pidfile, running
from athome.research.errors import AccountingIntegrityError, PreflightFailure, ResearchError
from athome.research.spec import ProposalTimeout

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

    from athome.detach import DetachedRun
    from athome.research.gate import TreeChange
    from athome.research.procs import ExperimentProcs
    from athome.research.spec import ExperimentSpec

RUN_PREFIX = "athome-research"
DEFAULT_CLAUDE_COMMAND = ("claude", "-p", "--output-format", "json", "--dangerously-skip-permissions")
DEFAULT_PROPOSAL_TIMEOUT_S = 3600.0
# total_cost became total_cost_usd in 1.0.22; older result envelopes cannot be accounted.
CLAUDE_CLI_ENVELOPE_FLOOR = (1, 0, 22)
CLAUDE_CLI_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+) \(Claude Code\)\r?\n?")


class CostError(AccountingIntegrityError):
    """A run log could not provide a valid ``total_cost_usd`` envelope."""


class MetricShapeError(ResearchError):
    """A candidate metric file reported a value of the wrong shape for the metric key.

    The message names only the offending value's type and the metric key, never the
    candidate-controlled value itself, so it stays safe to journal and render into the
    next proposal's contract.
    """


class Driver(Protocol):
    """A proposer that edits the mutable files in a plain candidate directory.

    An implementation receives the generated contract, the candidate directory, and
    ``budget_usd`` — the experiment's remaining spend budget at this invocation
    (``max_usd`` minus spend already recorded, never the full cap; ``None`` when the
    spec declares no ``max_usd``), mutates the mutable files in place, and returns
    the dollar cost that proposal incurred. It never writes the metric file — the immutable ``metric_command`` owns
    that channel — and it never inspects git state: the harness owns all git plumbing
    and derives the change description from the trusted tree diff (:func:`describe_change`).
    Before proposals begin, :meth:`preflight` validates any driver-specific external
    prerequisites without incurring proposal spend.
    If cancellation interrupts :meth:`propose`, :meth:`recover_cost` returns any billed
    cost the interrupted proposal persisted before it could return.
    Implementations should raise an :class:`AccountingIntegrityError` subtype for unknown
    or unrecoverable spend. The loop converts any ordinary recovery exception or backend
    cancellation into a durable abort latch instead of depending on that convention.
    Once the loop has durably recorded a proposal's spend — a journal row or a sidecar
    event — it calls :meth:`settle`; an implementation holding a durable process
    registration marks it terminal there and never earlier, so a kill between the two
    writes leaves a non-terminal record the startup orphan scan latches instead of
    invisible spend. :meth:`pending_run` names the registered detached run whose spend
    is captured or recovered but not yet settled (``None`` for drivers without one);
    the loop stamps that identity onto the durable accounting artifact so the orphan
    scan can recognize already-accounted spend instead of raising a false latch.
    """

    label: str

    async def preflight(self) -> None: ...

    async def propose(self, contract: str, workdir: Path, *, budget_usd: float | None) -> float: ...

    async def recover_cost(self) -> float: ...

    def pending_run(self) -> str | None: ...

    def settle(self) -> None: ...


async def read_reported_metric(spec: ExperimentSpec, workdir: Path) -> float | None:
    metric_file = anyio.Path(workdir) / spec.metric_file
    if not await metric_file.exists():
        return None
    try:
        text = await metric_file.read_text()
    except UnicodeDecodeError as exc:
        raise MetricShapeError(f"reported metric file for {spec.metric_key!r} has invalid encoding") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MetricShapeError(f"reported metric file for {spec.metric_key!r} is not valid JSON") from exc
    return reported_metric(payload, spec.metric_key)


def reported_metric(payload: object, key: str) -> float:
    if not (isinstance(payload, dict) and key in payload):
        raise MetricShapeError(f"reported metric file must be a JSON object with a {key!r} field")
    match payload[key]:
        case bool() as value:
            raise MetricShapeError(f"reported {key!r} must be a real number, not {type(value).__name__}")
        case int() | float() as value:
            return coerce_finite(value, key)
        case value:
            raise MetricShapeError(f"reported {key!r} must be a finite real number, not {type(value).__name__}")


def coerce_finite(value: int | float, key: str) -> float:
    reason = f"reported {key!r} must be a finite real number, not {type(value).__name__}"
    try:
        coerced = float(value)
    except OverflowError as exc:
        raise MetricShapeError(reason) from exc
    if isfinite(coerced):
        return coerced
    raise MetricShapeError(reason)


def describe_change(label: str, spec: ExperimentSpec, changes: Sequence[TreeChange], reported: float | None) -> str:
    files = ", ".join(sorted(change.path for change in changes)) or "no files"
    tail = "" if reported is None else f" (reported {spec.metric_key}={reported})"
    return f"{label} edited {files}{tail}"


def json_objects(text: str) -> Iterator[dict[str, object]]:
    decoder = json.JSONDecoder()
    index = text.find("{")
    while index != -1:
        try:
            obj, end = decoder.raw_decode(text, index)
        except ValueError:
            index = text.find("{", index + 1)
            continue
        if isinstance(obj, dict):
            yield obj
        index = text.find("{", end)


def envelope_cost(text: str) -> float:
    if not (costs := [obj["total_cost_usd"] for obj in json_objects(text) if "total_cost_usd" in obj]):
        raise CostError("run log carried no total_cost_usd envelope")
    match costs[-1]:
        case bool() | str() as value:
            raise CostError(f"total_cost_usd must be a real number, got {value!r}")
        case int() | float() as value:
            try:
                cost = float(value)
            except Exception as exc:
                raise CostError("total_cost_usd must be finite and non-negative") from exc
            if isfinite(cost) and cost >= 0:
                return cost
            raise CostError("total_cost_usd must be finite and non-negative")
        case value:
            raise CostError(f"total_cost_usd must be finite and non-negative, got {value!r}")


async def read_cost_log(run: DetachedRun) -> str:
    try:
        return await anyio.Path(run.log_path).read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise CostError("could not read detached proposal log; billing state is unknown") from exc


def claude_cli_version(output: bytes) -> tuple[int, ...]:
    try:
        text = output.decode()
    except UnicodeDecodeError as exc:
        raise PreflightFailure("claude --version output is not valid UTF-8") from exc
    if match := CLAUDE_CLI_VERSION.fullmatch(text):
        return tuple(int(part) for part in match.groups())
    raise PreflightFailure("claude --version output did not match '<version> (Claude Code)'")


@dataclass(frozen=True, slots=True)
class StubProposal:
    """One scripted edit the :class:`StubDriver` replays into a candidate directory.

    Attributes:
        files: Candidate-relative path to file content, written verbatim.
    """

    files: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class StubDriver:
    """A deterministic, LLM-free driver that replays a scripted sequence of edits.

    Each :meth:`propose` writes the next :class:`StubProposal`'s files into the
    candidate directory, so the greedy loop runs end to end without a live agent. The
    iterator is the driver's only mutable state; the dataclass itself stays frozen. It
    never spends money, so every proposal reports a cost of ``0.0``.

    Example:
        >>> driver = StubDriver(iter([StubProposal({"train.py": "LOSS = 0.3\\n"})]))
    """

    proposals: Iterator[StubProposal]
    label: str = "stub"

    async def preflight(self) -> None:
        return None

    async def propose(self, contract: str, workdir: Path, *, budget_usd: float | None) -> float:
        for relative, content in next(self.proposals).files.items():
            target = anyio.Path(workdir) / relative
            await target.parent.mkdir(parents=True, exist_ok=True)
            await target.write_text(content)
        return 0.0

    async def recover_cost(self) -> float:
        return 0.0

    def pending_run(self) -> str | None:
        return None

    def settle(self) -> None:
        return None


def stop(run: DetachedRun) -> None:
    try:
        os.killpg(os.getpgid(run.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        raise AccountingIntegrityError("could not stop detached proposal; billing state is unknown") from exc


@dataclass(slots=True)
class ProposalRecovery:
    run: DetachedRun | None = None
    pending: DetachedRun | None = None

    def retain(self, run: DetachedRun) -> None:
        self.run = run

    def stop(self) -> None:
        match self.run:
            case None:
                return
            case run:
                stop(run)


@dataclass(frozen=True, slots=True)
class ClaudeCodeDriver:
    """Drives one proposal per work-unit by running the ``claude`` CLI in the candidate directory.

    The generated contract is handed to a detached ``claude`` process
    (:func:`athome.detach.launch`) whose working directory is the plain candidate
    directory. :meth:`preflight` rejects CLI versions predating the
    ``total_cost_usd`` result envelope. :meth:`propose` passes the invocation's
    remaining experiment budget (never the full ``max_usd`` cap) as
    ``--max-budget-usd``, and bounds that run — ``timeout_s`` when set, else the
    spec's ``hard_kill_s``, else :data:`DEFAULT_PROPOSAL_TIMEOUT_S`, so a proposal is
    always bounded even when no wall-clock budget is configured — and waits on the
    process's actual exit (``detach.running`` reporting the pid gone, never a grepped
    log sentinel the agent can forge), killing the detached process group on timeout or
    cancellation so a hung or over-budget agent stops billing. It returns only the
    run's dollar cost, captured from the CLI's own ``total_cost_usd`` envelope (the last
    complete JSON object in the log, validated to a finite non-negative number). On
    cancellation, :meth:`recover_cost` reads that envelope from the retained run. On
    timeout it raises :class:`ProposalTimeout` carrying spend from a complete envelope;
    an incomplete log instead raises :class:`CostError` because spawned-run spend is
    unknown. The change description is derived harness-side from the trusted tree diff,
    never the agent's stdout. The ``claude`` CLI is shelled out to, never imported, so
    this driver loads without the ``llm``/``research`` extras. With ``procs`` bound
    (campaign mode), every proposal is durably registered before it launches, bound to
    its pid on spawn, and marked terminal only on :meth:`settle` — after the loop has
    durably recorded the spend that :meth:`propose`, :class:`ProposalTimeout`, or
    :meth:`recover_cost` handed it — so a harness-level kill at any earlier point
    leaves a non-terminal record the campaign's startup scan and the A7 watchdog turn
    into an alarm or an abort latch (:mod:`athome.research.procs`).

    Example:
        >>> driver = ClaudeCodeDriver(spec)
        >>> await run(spec, driver=driver, repo=repo)
    """

    spec: ExperimentSpec
    label: str = "claude"
    command: tuple[str, ...] = DEFAULT_CLAUDE_COMMAND
    poll: float = 5.0
    timeout_s: float | None = None
    procs: ExperimentProcs | None = None
    _recovery: ProposalRecovery = field(default_factory=ProposalRecovery, init=False, repr=False, compare=False)

    async def preflight(self) -> None:
        try:
            result = await anyio.run_process(
                [self.command[0], "--version"],
                check=False,
                stdout=PIPE,
                stderr=STDOUT,
            )
        except OSError as exc:
            raise PreflightFailure(f"claude --version could not run: {exc}") from exc
        if result.returncode != 0:
            raise PreflightFailure(f"claude --version exited with status {result.returncode}")
        if (version := claude_cli_version(result.stdout)) < CLAUDE_CLI_ENVELOPE_FLOOR:
            raise PreflightFailure(
                f"claude CLI {'.'.join(map(str, version))} is below required "
                f"{'.'.join(map(str, CLAUDE_CLI_ENVELOPE_FLOOR))}; upgrade Claude Code"
            )

    async def propose(self, contract: str, workdir: Path, *, budget_usd: float | None) -> float:
        budget = () if budget_usd is None else ("--max-budget-usd", str(budget_usd))
        inner = f"cd {shlex.quote(str(workdir))} && exec {shlex.join([*self.command, *budget, contract])}"
        name = f"{RUN_PREFIX}-{self.spec.name}-{uuid4().hex[:12]}"
        self._recovery.run = None
        if self.procs is not None:
            self.procs.register(name, log=run_log(name), pid_file=run_pidfile(name), exit_file=run_exitfile(name))
        try:
            run = await launch(["/bin/sh", "-c", inner], name=name, on_spawn=self.track)
        except BaseException:
            if self.procs is not None and self._recovery.run is None:
                self.procs.finalize(name, outcome="unspawned")
            self._recovery.stop()
            raise
        try:
            await self.await_exit(run)
        except TimeoutError as exc:
            stop(run)
            cost = await self.recovered_cost(run)
            self.account(run)
            raise ProposalTimeout(str(exc), cost=cost) from exc
        except BaseException:
            stop(run)
            raise
        cost = await self.captured_cost(run)
        self.account(run)
        self._recovery.run = None
        return cost

    def track(self, run: DetachedRun) -> None:
        self._recovery.retain(run)
        if self.procs is not None:
            self.procs.bind(run.name, pid=run.pid, pgid=os.getpgid(run.pid))

    def account(self, run: DetachedRun) -> None:
        self._recovery.pending = run

    def pending_run(self) -> str | None:
        match self._recovery.pending:
            case None:
                return None
            case run:
                return run.name

    def settle(self) -> None:
        match self._recovery.pending:
            case None:
                return
            case run:
                if self.procs is not None:
                    self.procs.finalize(run.name, outcome="accounted")
                self._recovery.pending = None

    async def recover_cost(self) -> float:
        match self._recovery.run:
            case None:
                return 0.0
            case run:
                cost = await self.recovered_cost(run)
                self.account(run)
                return cost

    async def await_exit(self, run: DetachedRun) -> None:
        fallback = self.spec.budget.hard_kill_s or DEFAULT_PROPOSAL_TIMEOUT_S
        timeout = self.timeout_s if self.timeout_s is not None else fallback
        deadline = anyio.current_time() + timeout
        while running(run.name) is not None:
            if anyio.current_time() >= deadline:
                raise TimeoutError(f"claude driver {run.name} exceeded its {timeout}s timeout")
            await anyio.sleep(self.poll)

    async def captured_cost(self, run: DetachedRun) -> float:
        return envelope_cost(await read_cost_log(run))

    async def recovered_cost(self, run: DetachedRun) -> float:
        return envelope_cost(await read_cost_log(run))
