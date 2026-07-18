"""The campaign outer loop: reservation-capped launches, an append-only ledger, and a kill switch.

Each round asks the proposer (A6a) for one admitted spec, reserves that spec's
declared worst case (``max_usd``/``max_wall_s``) against the
:class:`~athome.research.policy.CampaignBudget` before any paid work, runs the
inner loop with the A2 preflight and A3 failure taxonomy unchanged, and releases
the reservation to journal-measured actuals in the ledger. An infra-aborted
experiment never consumes a candidate-experiment slot — it counts toward the
consecutive-failure stop instead. Gated mode writes the admitted spec to
``pending/`` and never runs it; :func:`approve` digest-verifies the file into the
queue, and the gated drain re-verifies every queued spec against its ``approved``
ledger row — refusing a drifted or unledgered file, and refusing to replay a
sequence whose launch already began — after which the codepath is identical. The stop file halts the campaign at
the experiment boundary, like the A3 restart latch: the current experiment
finishes, nothing new launches, and the file persists until the operator removes
it; the runner re-checks it immediately before each proposer call and again
immediately before each reservation/launch, so a stop armed mid-round blocks the
next paid step. Reservations are admission control, not a hard ceiling: metering
is post-hoc, so a single in-flight driver call can overshoot its grant before
actuals land. Each invocation is therefore granted only the experiment's
remaining budget (never the full ``max_usd`` cap), terminal ledger rows record
true journal-measured actuals overshoot included, and the moment recorded
actuals reach ``max_total_usd`` the boundary latches refusal of all new work —
the worst-case breach is bounded by one invocation's overshoot, and the cap is
never stretched. On start and resume the runner reconciles ``completed`` ledger
rows against ``retros.jsonl`` and generates catch-up retros for any gap before
the next proposal round; a catch-up failure aborts as loudly as a live one.
Proposer and retro LLM spend is count-bounded (at most
``MAX_PROPOSAL_ATTEMPTS`` extract calls plus one retro per round), not measured;
experiment spend is the precisely-measured cap.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import StrEnum, nonmember
from hashlib import sha256
from json import JSONDecodeError, loads
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import anyio

from athome import launchd
from athome.cache import atomic_write_text
from athome.config import SectionSettings
from athome.progress import RunSink, load_journal
from athome.research import nightly, retro
from athome.research.common import Hasher
from athome.research.contract import sanitize_history
from athome.research.driver import ClaudeCodeDriver
from athome.research.errors import AccountingIntegrityError, PreflightFailure, ResearchError
from athome.research.failures import InfraFailure, infra_cost
from athome.research.journal import Journal
from athome.research.loop import experiment_lock
from athome.research.loop import run as run_experiment
from athome.research.policy import ProposalPolicy
from athome.research.propose import ProposalViolation, ProposerContext, propose
from athome.research.retro import RetroJournal, RetroRecord
from athome.research.spec import BudgetExhausted, ExperimentSpec, finite_number

if TYPE_CHECKING:
    from collections.abc import Callable

    from spawnllm import LlmBackend, TModel

    from athome.research.driver import Driver
    from athome.research.propose import ProposalRound

LEDGER_NAME = "ledger.jsonl"
RETROS_NAME = "retros.jsonl"
STOP_NAME = "stop.json"
LOCK_NAME = "meta.lock"
SPEC_NAME = "experiment.toml"
PROPOSAL_NAME = "proposal.json"
PENDING_DIR = "pending"
QUEUE_DIR = "queue"
EXPERIMENTS_DIR = "experiments"
FAILURE_CONTEXT_LIMIT = 10


class CampaignError(ResearchError):
    """A campaign-level refusal: a missing or drifted pending spec, or an unserializable audit record."""


class PoisonedLedger(AccountingIntegrityError):
    """The campaign ledger is unreadable, malformed, or carries invalid accounting."""


class CampaignEvent(StrEnum):
    """One kind of append-only campaign ledger row.

    ``TERMINAL`` events release a reservation to actuals, ``FAILURES`` feed the
    consecutive-failure stop, ``RUNS`` consume a candidate-experiment slot
    (and reset the failure streak), and ``LAUNCHED`` events mark a sequence
    whose launch already began, barring a queue replay.
    """

    PROPOSED = "proposed"
    REJECTED = "rejected"
    PENDING = "pending"
    APPROVED = "approved"
    RESERVED = "reserved"
    STARTED = "started"
    PREFLIGHT_FAILED = "preflight_failed"
    ABORTED = "aborted"
    INFRA_ABORTED = "infra_aborted"
    COMPLETED = "completed"
    STOPPED = "stopped"

    TERMINAL = nonmember(frozenset({COMPLETED, ABORTED, INFRA_ABORTED, PREFLIGHT_FAILED}))
    FAILURES = nonmember(frozenset({REJECTED, PREFLIGHT_FAILED, INFRA_ABORTED}))
    LAUNCHED = nonmember(frozenset({RESERVED, STARTED, COMPLETED, ABORTED, INFRA_ABORTED, PREFLIGHT_FAILED}))
    RUNS = nonmember(frozenset({COMPLETED, ABORTED}))


@dataclass(frozen=True, slots=True)
class LedgerRow:
    """One campaign event: what happened, to which experiment, and what it cost.

    Attributes:
        seq: The experiment's ledger sequence number; ``0`` for campaign-scope events.
        event: The campaign event kind.
        usd: Reserved worst case on ``reserved`` rows, journal-measured actuals on
            terminal rows, ``0`` otherwise.
        wall_s: Reserved worst case on ``reserved`` rows, measured wall clock on
            terminal rows, ``0`` otherwise.
        reason: The harness-authored reason on refusals, failures, and stops.
        extra: Audit payload (experiment name, template, digests, result stats).
    """

    seq: int
    event: CampaignEvent
    usd: float
    wall_s: float
    reason: str
    extra: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, object]) -> LedgerRow:
        return cls(
            seq=record["seq"],
            event=CampaignEvent(record["event"]),
            usd=record["usd"],
            wall_s=record["wall_s"],
            reason=record["reason"],
            extra=record["extra"],
        )

    def to_record(self) -> dict[str, object]:
        return {
            "seq": self.seq,
            "event": self.event.value,
            "usd": self.usd,
            "wall_s": self.wall_s,
            "reason": self.reason,
            "extra": self.extra,
        }


@dataclass(slots=True)
class Ledger:
    """Append-only campaign chronology on ``progress.RunSink``; the disk is the record.

    Crash-resume is recomputation: totals, the next sequence number, the
    consecutive-failure streak, and open reservations are all derived from the
    JSONL rows on every :meth:`open`, never cached across processes. An
    experiment holding a ``reserved`` row without a terminal release counts its
    full worst case in the totals, so a crash mid-experiment can only
    over-reserve, never breach the cap.

    Example:
        >>> ledger = Ledger.open(root / "ledger.jsonl")
        >>> ledger.next_seq()
    """

    sink: RunSink
    _rows: list[LedgerRow]

    @classmethod
    def open(cls, path: Path) -> Ledger:
        try:
            if (
                path.exists()
                and (
                    final_line := next((line for line in reversed(path.read_text().splitlines()) if line.strip()), None)
                )
                is not None
            ):
                loads(final_line)
            sink = RunSink.open(path)
            rows = [LedgerRow.from_record(record) for record in load_journal(path)]
        except (OSError, UnicodeDecodeError, JSONDecodeError, AttributeError, KeyError, TypeError, ValueError) as exc:
            raise PoisonedLedger(f"campaign ledger at {path} is unreadable or malformed") from exc
        for row in rows:
            if not (finite_number(row.usd) and row.usd >= 0 and finite_number(row.wall_s) and row.wall_s >= 0):
                raise PoisonedLedger(f"campaign ledger at {path} carries invalid accounting on seq {row.seq}")
        return cls(sink, rows)

    async def append(self, row: LedgerRow) -> None:
        await self.sink.append(row.to_record())
        self._rows.append(row)

    def rows(self) -> list[LedgerRow]:
        return list(self._rows)

    def next_seq(self) -> int:
        return max((row.seq for row in self._rows), default=0) + 1

    def open_reservations(self) -> dict[int, LedgerRow]:
        released = {row.seq for row in self._rows if row.event in CampaignEvent.TERMINAL}
        return {row.seq: row for row in self._rows if row.event is CampaignEvent.RESERVED and row.seq not in released}

    def total_usd(self) -> float:
        return sum(row.usd for row in self._rows if row.event in CampaignEvent.TERMINAL) + sum(
            row.usd for row in self.open_reservations().values()
        )

    def total_wall_s(self) -> float:
        return sum(row.wall_s for row in self._rows if row.event in CampaignEvent.TERMINAL) + sum(
            row.wall_s for row in self.open_reservations().values()
        )

    def experiments_run(self) -> int:
        return sum(row.event in CampaignEvent.RUNS for row in self._rows)

    def consecutive_failures(self) -> int:
        streak = 0
        for row in self._rows:
            if row.event in CampaignEvent.FAILURES:
                streak += 1
            elif row.event in CampaignEvent.RUNS:
                streak = 0
        return streak

    def failure_reasons(self, *, limit: int = FAILURE_CONTEXT_LIMIT) -> tuple[str, ...]:
        informative = CampaignEvent.FAILURES | {CampaignEvent.ABORTED}
        return tuple(
            f"round {row.seq} {row.event.value}: {row.reason}" for row in self._rows if row.event in informative
        )[-limit:]


class MetaSettings(SectionSettings):
    """Campaign outer-loop settings: the ``[research.meta]`` section of ``~/.athome/config.toml``.

    Attributes:
        root: The campaign state root holding the ledger, retros, pending specs,
            queue, audit dirs, and stop file.
        backend: The spawnllm backend registry name the proposer and retro run on.
        tier: The abstract spawnllm model tier for proposer and retro calls.
    """

    section: ClassVar[tuple[str, ...]] = ("research", "meta")

    root: Path = Path("~/.athome/research/meta")
    backend: str = "codex"
    tier: str = "large"


@dataclass(frozen=True, slots=True)
class CampaignResult:
    """What one ``meta run`` invocation left behind.

    Attributes:
        completed: Completed experiments across the whole ledger.
        total_usd: The reservation-conservative cumulative spend.
        halted: The boundary reason the runner stopped on, or ``None`` when a
            gated round ended awaiting operator review.
    """

    completed: int
    total_usd: float
    halted: str | None


async def request_stop(root: Path, *, reason: str = "operator stop") -> Path:
    """Arm the kill switch: the runner halts at the next experiment boundary.

    The stop file is written atomically (the A3 restart-latch idiom) and
    persists until the operator removes it; the current experiment finishes and
    nothing new launches.

    Args:
        root: The campaign state root.
        reason: Recorded in the stop file and every subsequent ``stopped`` row.

    Returns:
        The stop file path.
    """
    await anyio.Path(root).mkdir(parents=True, exist_ok=True)
    await atomic_write_text(anyio.Path(path := root / STOP_NAME), json.dumps({"reason": reason, "ts": time.time()}))
    return path


async def read_stop(root: Path) -> str | None:
    try:
        payload = loads(await anyio.Path(root / STOP_NAME).read_text())
    except FileNotFoundError:
        return None
    return str(payload["reason"])


def toml_value(value: object) -> str:
    match value:
        case bool() | None:
            raise CampaignError(f"unrepresentable TOML value {value!r}")
        case str() | int() | float():
            return json.dumps(value)
        case tuple():
            return f"[{', '.join(toml_value(item) for item in value)}]"
        case _:
            raise CampaignError(f"unrepresentable TOML value {value!r}")


def spec_toml(spec: ExperimentSpec) -> str:
    scalars = (
        ("name", spec.name),
        ("metric_command", spec.metric_command),
        ("metric_key", spec.metric_key),
        ("direction", spec.direction),
        ("mutable_paths", spec.mutable_paths),
        ("immutable_paths", spec.immutable_paths),
        ("metric_file", spec.metric_file),
        ("hypothesis", spec.hypothesis),
        ("known_good_dir", spec.known_good_dir),
    )
    budget = (
        ("max_units", spec.budget.max_units),
        ("max_wall_s", spec.budget.max_wall_s),
        ("hard_kill_s", spec.budget.hard_kill_s),
        ("max_usd", spec.budget.max_usd),
    )
    lines = [f"{key} = {toml_value(value)}" for key, value in scalars if value is not None]
    lines += ["", "[budget]", *(f"{key} = {toml_value(value)}" for key, value in budget if value is not None)]
    return "\n".join(lines) + "\n"


def render_retro(record: RetroRecord) -> str:
    verdict = record.verdict
    return "\n".join(
        [
            f"{record.experiment}: {verdict.outcome} "
            f"(baseline {record.baseline} -> best {record.best_metric}, uplift {record.uplift})",
            verdict.summary,
            *(f"- evidence: {line}" for line in verdict.evidence),
            *(f"- next: {line}" for line in verdict.next_steps),
        ]
    )


def spec_seq(name: str) -> int:
    return int(name.split("-", 1)[0])


def backend_name(backend: LlmBackend) -> str:
    return str(backend.provider)


def proposed_row(seq: int, round_: ProposalRound) -> LedgerRow:
    return LedgerRow(
        seq=seq,
        event=CampaignEvent.PROPOSED,
        usd=0.0,
        wall_s=0.0,
        reason="",
        extra={"name": round_.spec.name, "template": round_.proposal.template, "attempts": round_.attempts},
    )


def pending_spec(root: Path, seq: int) -> Path:
    match sorted((root / PENDING_DIR).glob(f"{seq:03d}-*.toml")):
        case [path]:
            return path
        case []:
            raise CampaignError(f"no pending spec for seq {seq}")
        case matches:
            raise CampaignError(f"ambiguous pending specs for seq {seq}: {[path.name for path in matches]}")


@dataclass(frozen=True, slots=True)
class Campaign:
    policy: ProposalPolicy
    repo: Path
    root: Path
    backend: LlmBackend
    driver_factory: Callable[[ExperimentSpec], Driver]
    tier: TModel
    mirror_cc_notes: bool
    ledger: Ledger
    retros: RetroJournal

    async def run(self) -> CampaignResult:
        await self.reconcile_retros()
        match self.policy.mode:
            case "auto":
                return await self.run_auto()
            case "gated":
                return await self.run_gated()

    async def reconcile_retros(self) -> None:
        durable = {record.experiment for record in self.retros.records()}
        for row in self.ledger.rows():
            if row.event is CampaignEvent.COMPLETED and (name := str(row.extra["name"])) not in durable:
                await self.write_retro(ExperimentSpec.load(self.root / EXPERIMENTS_DIR / name / SPEC_NAME))

    async def run_auto(self) -> CampaignResult:
        while True:
            if (reason := await self.boundary()) is not None:
                return await self.halt(reason)
            seq = self.ledger.next_seq()
            context = self.context()
            if (reason := await self.stopped()) is not None:
                return await self.halt(reason)
            if (round_ := await self.next_round(seq, context)) is None:
                continue
            await self.materialize(round_, seq=seq, context=context)
            if (refusal := self.reservation_refusal(round_.spec)) is not None:
                return await self.halt(refusal)
            if (reason := await self.stopped()) is not None:
                return await self.halt(reason)
            await self.launch(round_.spec)

    async def run_gated(self) -> CampaignResult:
        for path in sorted((self.root / QUEUE_DIR).glob("*.toml")):
            if (reason := await self.boundary()) is not None:
                return await self.halt(reason)
            spec = self.verified_queue_spec(path)
            if (refusal := self.reservation_refusal(spec)) is not None:
                return await self.halt(refusal)
            if (reason := await self.stopped()) is not None:
                return await self.halt(reason)
            await self.launch(spec)
            await anyio.Path(path).unlink()
        if (reason := await self.boundary()) is not None:
            return await self.halt(reason)
        seq = self.ledger.next_seq()
        context = self.context()
        if (reason := await self.stopped()) is not None:
            return await self.halt(reason)
        if (round_ := await self.next_round(seq, context)) is None:
            return self.result(None)
        text = await self.materialize(round_, seq=seq, context=context)
        pending = anyio.Path(self.root / PENDING_DIR)
        await pending.mkdir(parents=True, exist_ok=True)
        await atomic_write_text(pending / f"{round_.spec.name}.toml", text)
        await self.ledger.append(
            LedgerRow(
                seq=seq,
                event=CampaignEvent.PENDING,
                usd=0.0,
                wall_s=0.0,
                reason="",
                extra={"name": round_.spec.name, "sha256": sha256(text.encode()).hexdigest()},
            )
        )
        return self.result(None)

    async def next_round(self, seq: int, context: ProposerContext) -> ProposalRound | None:
        try:
            round_ = await propose(self.policy, context, backend=self.backend, seq=seq, tier=self.tier)
        except ProposalViolation as violation:
            await self.ledger.append(
                LedgerRow(seq=seq, event=CampaignEvent.REJECTED, usd=0.0, wall_s=0.0, reason=str(violation))
            )
            return None
        await self.ledger.append(proposed_row(seq, round_))
        return round_

    async def stopped(self) -> str | None:
        if (reason := await read_stop(self.root)) is not None:
            return f"stop requested: {reason}"
        return None

    async def boundary(self) -> str | None:
        campaign = self.policy.campaign
        if (reason := await self.stopped()) is not None:
            return reason
        if (total := self.ledger.total_usd()) >= campaign.max_total_usd:
            return (
                f"campaign budget exhausted: recorded ${total:.2f} reached max_total_usd "
                f"${campaign.max_total_usd:.2f}; refusing all new work"
            )
        if self.ledger.experiments_run() >= campaign.max_experiments:
            return f"campaign complete: {campaign.max_experiments} experiments run"
        if (streak := self.ledger.consecutive_failures()) >= campaign.max_consecutive_failures:
            return (
                f"{streak} consecutive failed rounds crossed "
                f"max_consecutive_failures {campaign.max_consecutive_failures}"
            )
        return None

    async def halt(self, reason: str) -> CampaignResult:
        await self.ledger.append(LedgerRow(seq=0, event=CampaignEvent.STOPPED, usd=0.0, wall_s=0.0, reason=reason))
        return self.result(reason)

    def result(self, halted: str | None) -> CampaignResult:
        return CampaignResult(
            completed=sum(row.event is CampaignEvent.COMPLETED for row in self.ledger.rows()),
            total_usd=self.ledger.total_usd(),
            halted=halted,
        )

    def context(self) -> ProposerContext:
        return ProposerContext(
            retros=tuple(render_retro(record) for record in self.retros.records()),
            failures=tuple(sanitize_history(reason) for reason in self.ledger.failure_reasons()),
        )

    def reservation_refusal(self, spec: ExperimentSpec) -> str | None:
        campaign = self.policy.campaign
        if spec.budget.max_usd is None or spec.budget.max_wall_s is None:
            return f"reservation refused: {spec.name} declares no max_usd/max_wall_s worst case to reserve"
        if (reserved := self.ledger.total_usd() + spec.budget.max_usd) > campaign.max_total_usd:
            return (
                f"reservation refused: {spec.name} worst case ${spec.budget.max_usd:.2f} would take the campaign "
                f"to ${reserved:.2f}, over max_total_usd ${campaign.max_total_usd:.2f}"
            )
        if (reserved_wall := self.ledger.total_wall_s() + spec.budget.max_wall_s) > campaign.max_wall_s:
            return (
                f"reservation refused: {spec.name} worst case {spec.budget.max_wall_s:.0f}s would take the campaign "
                f"to {reserved_wall:.0f}s, over max_wall_s {campaign.max_wall_s:.0f}s"
            )
        return None

    def verified_queue_spec(self, path: Path) -> ExperimentSpec:
        seq = spec_seq(path.stem)
        if any(row.seq == seq and row.event in CampaignEvent.LAUNCHED for row in self.ledger.rows()):
            raise CampaignError(f"queued spec {path.name} for seq {seq} was already launched; refusing to run it again")
        approved = next(
            (row for row in reversed(self.ledger.rows()) if row.seq == seq and row.event is CampaignEvent.APPROVED),
            None,
        )
        if approved is None:
            raise CampaignError(f"queued spec {path.name} has no approved ledger row for seq {seq}")
        raw = path.read_bytes()
        if (digest := sha256(raw).hexdigest()) != approved.extra["sha256"]:
            raise CampaignError(
                f"queued spec {path.name} digest {digest} does not match the approved {approved.extra['sha256']}; "
                "refusing to launch a drifted spec"
            )
        return ExperimentSpec.loads(raw.decode(), source=str(path))

    async def materialize(self, round_: ProposalRound, *, seq: int, context: ProposerContext) -> str:
        audit = anyio.Path(self.root / EXPERIMENTS_DIR / round_.spec.name)
        await audit.mkdir(parents=True, exist_ok=True)
        text = spec_toml(round_.spec)
        await atomic_write_text(audit / SPEC_NAME, text)
        if ExperimentSpec.load(Path(audit) / SPEC_NAME) != round_.spec:
            raise CampaignError(f"materialized spec for {round_.spec.name} does not round-trip")
        await atomic_write_text(
            audit / PROPOSAL_NAME,
            json.dumps(
                {
                    "seq": seq,
                    "name": round_.spec.name,
                    "template": round_.proposal.template,
                    "attempts": round_.attempts,
                    "proposal": round_.proposal.model_dump(),
                    "policy_digest": Hasher.digest(asdict(self.policy)),
                    "proposer": {"backend": backend_name(self.backend), "tier": str(self.tier)},
                    "retro_digests": [Hasher.digest(rendered) for rendered in context.retros],
                },
                sort_keys=True,
            ),
        )
        return text

    async def launch(self, spec: ExperimentSpec) -> None:
        seq = spec_seq(spec.name)
        await self.ledger.append(
            LedgerRow(
                seq=seq,
                event=CampaignEvent.RESERVED,
                usd=spec.budget.max_usd,
                wall_s=spec.budget.max_wall_s,
                reason="",
                extra={"name": spec.name},
            )
        )
        await self.ledger.append(
            LedgerRow(seq=seq, event=CampaignEvent.STARTED, usd=0.0, wall_s=0.0, reason="", extra={"name": spec.name})
        )
        started = time.monotonic()
        try:
            result = await run_experiment(
                spec, driver=self.driver_factory(spec), repo=self.repo, mirror_cc_notes=self.mirror_cc_notes
            )
        except PreflightFailure as exc:
            await self.release(spec, seq=seq, event=CampaignEvent.PREFLIGHT_FAILED, reason=str(exc), started=started)
            return
        except BudgetExhausted as exc:
            await self.release(spec, seq=seq, event=CampaignEvent.ABORTED, reason=str(exc), started=started)
            return
        except InfraFailure as exc:
            await self.release(spec, seq=seq, event=CampaignEvent.INFRA_ABORTED, reason=str(exc), started=started)
            return
        await self.release(
            spec,
            seq=seq,
            event=CampaignEvent.COMPLETED,
            reason="",
            started=started,
            extra={"kept": result.kept, "best_metric": result.best.metric if result.best is not None else None},
        )
        await self.write_retro(spec)

    async def release(
        self,
        spec: ExperimentSpec,
        *,
        seq: int,
        event: CampaignEvent,
        reason: str,
        started: float,
        extra: dict[str, object] | None = None,
    ) -> None:
        await self.ledger.append(
            LedgerRow(
                seq=seq,
                event=event,
                usd=await self.actual_usd(spec),
                wall_s=time.monotonic() - started,
                reason=reason,
                extra={"name": spec.name} | (extra or {}),
            )
        )

    async def actual_usd(self, spec: ExperimentSpec) -> float:
        path = await nightly.journal_path(self.repo, spec.name)
        rows = Journal.open(path).rows()
        return sum(row.resources["usd"] for row in rows) + infra_cost(path.with_name(f"{spec.name}.events.jsonl"))

    async def write_retro(self, spec: ExperimentSpec) -> None:
        report = await nightly.report(spec, repo=self.repo)
        verdict = await retro.generate(report.rows, report, backend=self.backend, tier=self.tier)
        await self.retros.append(RetroRecord.from_report(report, verdict))


async def run_campaign(
    policy: ProposalPolicy,
    *,
    repo: Path,
    root: Path,
    backend: LlmBackend | str,
    driver_factory: Callable[[ExperimentSpec], Driver] = ClaudeCodeDriver,
    tier: TModel = "large",
    mirror_cc_notes: bool = False,
) -> CampaignResult:
    """Run campaign rounds under the flock until a boundary halts them.

    On entry, ``completed`` ledger rows missing a durable retro get catch-up
    retros before any proposal round. Each auto-mode round: kill-switch and
    cumulative-cap boundary check (recorded actuals at or over ``max_total_usd``
    latch refusal of all new work), one
    proposer round (a policy violation ledgers ``rejected`` and continues),
    audit-dir materialization, a worst-case reservation refused rather than
    stretched, then the inner loop with every A2/A3 protection unchanged; the
    stop file is re-checked immediately before the proposer call and again
    immediately before reservation/launch. A
    :class:`~athome.research.errors.PreflightFailure` ledgers
    ``preflight_failed``, :class:`~athome.research.spec.BudgetExhausted` ledgers
    ``aborted``, and :class:`~athome.research.failures.InfraFailure` ledgers
    ``infra_aborted`` — none of them stop the campaign until the
    consecutive-failure cap trips. A completed experiment ledgers journal-summed
    actuals and generates its retro (A5), which feeds the next round's proposer
    context; a retro-generation error aborts the campaign loudly, after the
    ``completed`` accounting row is already durable. Gated mode drains the
    approved queue through the identical codepath, then writes one admitted
    proposal to ``pending/`` and returns for operator review.

    Args:
        policy: The frozen operator policy (mode, campaign caps, ceilings, templates).
        repo: The git repository experiments run against.
        root: The campaign state root (ledger, retros, pending, queue, stop file).
        backend: A bound spawnllm backend, or a backend registry name.
        driver_factory: Builds the per-experiment driver; defaults to ``ClaudeCodeDriver``.
        tier: The abstract spawnllm model tier for proposer and retro calls.
        mirror_cc_notes: Whether journals and retros mirror to the installed ``cc-notes``.

    Returns:
        The invocation's :class:`CampaignResult`.

    Raises:
        ConcurrentRun: Another live runner holds the campaign lock.
        PoisonedLedger: The ledger on disk is unreadable, malformed, or torn.
        AccountingIntegrityError: The inner loop latched an accounting abort.
        RetroError: A completed experiment's retrospective could not be generated.
    """
    from spawnllm.backends.registry import BACKENDS_BY_NAME

    await anyio.Path(root).mkdir(parents=True, exist_ok=True)
    async with experiment_lock(root / LOCK_NAME):
        campaign = Campaign(
            policy=policy,
            repo=repo,
            root=root,
            backend=BACKENDS_BY_NAME[backend] if isinstance(backend, str) else backend,
            driver_factory=driver_factory,
            tier=tier,
            mirror_cc_notes=mirror_cc_notes,
            ledger=Ledger.open(root / LEDGER_NAME),
            retros=RetroJournal.open(root / RETROS_NAME, mirror_cc_notes=mirror_cc_notes),
        )
        return await campaign.run()


async def approve(root: Path, seq: int) -> Path:
    """Digest-verify the pending spec for ``seq`` and move it into the run queue.

    The file's SHA-256 must match the ``pending`` ledger row's digest exactly; a
    drifted spec is refused, never repaired. The next gated ``meta run`` drains
    the queue through the identical launch codepath.

    Args:
        root: The campaign state root.
        seq: The pending experiment's ledger sequence number.

    Returns:
        The queued spec path.

    Raises:
        CampaignError: No, or an ambiguous, pending spec for ``seq``; no
            ``pending`` ledger row; or a digest mismatch.
        ConcurrentRun: A live campaign runner holds the campaign lock.
    """
    async with experiment_lock(root / LOCK_NAME):
        path = pending_spec(root, seq)
        ledger = Ledger.open(root / LEDGER_NAME)
        row = next(
            (row for row in reversed(ledger.rows()) if row.seq == seq and row.event is CampaignEvent.PENDING), None
        )
        if row is None:
            raise CampaignError(f"no pending ledger row for seq {seq}")
        text = await anyio.Path(path).read_text()
        if (digest := sha256(text.encode()).hexdigest()) != row.extra["sha256"]:
            raise CampaignError(
                f"pending spec {path.name} digest {digest} does not match the ledgered {row.extra['sha256']}; "
                "refusing to queue a drifted spec"
            )
        queue = anyio.Path(root / QUEUE_DIR)
        await queue.mkdir(parents=True, exist_ok=True)
        await anyio.Path(path).rename(target := Path(queue) / path.name)
        await ledger.append(
            LedgerRow(
                seq=seq,
                event=CampaignEvent.APPROVED,
                usd=0.0,
                wall_s=0.0,
                reason="",
                extra={"name": path.stem, "sha256": digest},
            )
        )
        return target


async def reject(root: Path, seq: int, *, reason: str) -> None:
    """Remove the pending spec for ``seq`` and ledger why the operator refused it.

    Args:
        root: The campaign state root.
        seq: The pending experiment's ledger sequence number.
        reason: The operator's refusal, fed into the next round's proposer context.

    Raises:
        CampaignError: No, or an ambiguous, pending spec for ``seq``.
        ConcurrentRun: A live campaign runner holds the campaign lock.
    """
    async with experiment_lock(root / LOCK_NAME):
        path = pending_spec(root, seq)
        ledger = Ledger.open(root / LEDGER_NAME)
        await anyio.Path(path).unlink()
        await ledger.append(
            LedgerRow(
                seq=seq,
                event=CampaignEvent.REJECTED,
                usd=0.0,
                wall_s=0.0,
                reason=f"operator rejected: {reason}",
                extra={"name": path.stem},
            )
        )


async def campaign_report(root: Path) -> dict[str, object]:
    """Summarize the campaign ledger: totals, event counts, and the kill-switch state.

    Args:
        root: The campaign state root.
    """
    ledger = Ledger.open(root / LEDGER_NAME)
    events = Counter(row.event.value for row in ledger.rows())
    return {
        "next_seq": ledger.next_seq(),
        "completed": events.get(CampaignEvent.COMPLETED.value, 0),
        "experiments_run": ledger.experiments_run(),
        "total_usd": ledger.total_usd(),
        "total_wall_s": ledger.total_wall_s(),
        "consecutive_failures": ledger.consecutive_failures(),
        "open_reservations": sorted(ledger.open_reservations()),
        "events": dict(events),
        "pending": sorted(path.name for path in (root / PENDING_DIR).glob("*.toml")),
        "stop": await read_stop(root),
    }


async def install(policy_path: Path, *, calendar: launchd.Calendar = nightly.NIGHTLY_CALENDAR) -> Path:
    """Install a launchd Calendar agent running the campaign's outer loop.

    The agent runs ``athome research meta run <policy>`` at ``calendar`` in the
    policy's git repository; a re-install replaces the existing agent for the
    same policy file.

    Args:
        policy_path: The campaign policy TOML the agent runs.
        calendar: When to fire; defaults to 02:00 nightly.

    Returns:
        The path of the written launchd plist.
    """
    ProposalPolicy.load(policy_path)
    agent = launchd.AgentSpec(
        label=f"{nightly.RESEARCH_LABEL_PREFIX}meta.{policy_path.stem}",
        command=("athome", "research", "meta", "run", str(policy_path.resolve())),
        schedule=calendar,
        working_dir=await nightly.repo_root(policy_path),
    )
    return await launchd.install(agent)
