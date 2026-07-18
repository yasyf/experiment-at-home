"""Durable registry of detached proposal processes: a harness kill cannot hide spend.

A detached proposal lives in its own session (:func:`athome.detach.launch`), so a
harness-level kill — SIGKILL, or a ``launchctl bootout`` that takes down only the
job's process group — leaves a live BILLING process behind with no Python unwind:
the in-memory recovery state is gone, the per-attempt UUID run name is unrecorded,
and a restart scans only accounting artifacts. The registry closes that gap with an
append-only JSONL file under the campaign root: every proposal appends a ``register``
line *before* the detached process launches, a ``bind`` line the moment the pid
exists, and a ``terminal`` line once its spend enters the accounting stream
(captured, recovered, or carried by a :class:`~athome.research.spec.ProposalTimeout`).
A non-terminal record whose runner is gone is an orphan: a live pid raises the alarm
(spend still accruing with no harness), and a dead pid writes the experiment's
accounting-abort latch — the A3.x exactly-once-or-latch invariant — and marks the
record terminal. :func:`athome.research.meta.run_campaign` refuses to start while an
orphan is live or its latch remains unreconciled, and the A7 watchdog runs the same
scan whenever no live runner holds the campaign lock.

A torn final line (the harness killed mid-append) is skipped on read and healed by
the next append; losing a torn ``terminal`` line only makes the scan conservative —
it latches spend that was in fact accounted, and reconciliation clears it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
from loguru import logger

from athome.cache import atomic_write_text

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

PROCS_NAME = "procs.jsonl"


@dataclass(frozen=True, slots=True)
class ProposalProcess:
    """One registered detached proposal process, folded from its registry lines.

    Attributes:
        run: The detached run name (also names its log, pid, and exit files).
        experiment: The experiment the proposal belongs to.
        seq: The campaign sequence number of the experiment.
        spec_digest: SHA-256 of the materialized experiment TOML.
        started_at: Unix timestamp when the proposal was registered.
        log: The detached run's log file.
        pid_file: The detached run's pid file.
        exit_file: The detached run's exit-code file.
        pid: The bound session-leader pid, or ``None`` if the bind never landed.
        pgid: The bound process-group id, or ``None`` if the bind never landed.
        outcome: The terminal outcome (``accounted``, ``unspawned``, ``orphaned``),
            or ``None`` while the proposal's spend has not entered the accounting
            stream.
    """

    run: str
    experiment: str
    seq: int
    spec_digest: str
    started_at: float
    log: Path
    pid_file: Path
    exit_file: Path
    pid: int | None = None
    pgid: int | None = None
    outcome: str | None = None


@dataclass(frozen=True, slots=True)
class Orphan:
    """A registry record with no live harness accounting for it.

    Attributes:
        record: The orphaned registry record.
        pid: The resolved pid, or ``None`` when no pid was ever durably bound.
        live: Whether the process is still alive (and still billing).
        latch: The experiment's accounting-abort latch path.
    """

    record: ProposalProcess
    pid: int | None
    live: bool
    latch: Path


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def resolve_pid(record: ProposalProcess) -> int | None:
    if record.pid is not None:
        return record.pid
    try:
        text = record.pid_file.read_text()
    except OSError:
        return None
    try:
        return int(text)
    except ValueError:
        return None


async def abort_latch(repo: Path, experiment: str) -> Path:
    from athome.research import nightly

    journal = await nightly.journal_path(repo, experiment)
    return journal.with_name(f"{experiment}.abort.json")


def refusal(orphans: Sequence[Orphan]) -> str:
    described = "; ".join(
        f"{orphan.record.run} ({f'LIVE pid {orphan.pid}, still billing' if orphan.live else f'latched {orphan.latch}'})"
        for orphan in orphans
    )
    return (
        f"detached proposal processes with unreconciled spend: {described}; "
        "check the provider ledger, reconcile the spend, then delete each latch file"
    )


@dataclass(frozen=True, slots=True)
class ProcessRegistry:
    """The append-only proposal-process registry under a campaign root.

    Example:
        >>> registry = ProcessRegistry(root / PROCS_NAME)
        >>> registry.records()
    """

    path: Path

    def experiment(self, name: str, *, seq: int, spec_digest: str) -> ExperimentProcs:
        return ExperimentProcs(self, name, seq, spec_digest)

    def append(self, event: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(event) + "\n").encode()
        fd = os.open(self.path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            if (size := os.fstat(fd).st_size) and os.pread(fd, 1, size - 1) != b"\n":
                os.write(fd, b"\n")  # heal a torn prior line so this record never concatenates onto it
            offset = 0
            while offset < len(payload):
                offset += os.write(fd, payload[offset:])
        finally:
            os.close(fd)

    def finalize(self, run: str, *, outcome: str) -> None:
        self.append({"op": "terminal", "run": run, "outcome": outcome, "ts": time.time()})

    def records(self) -> list[ProposalProcess]:
        procs: dict[str, ProposalProcess] = {}
        for event in self.events():
            match event:
                case {
                    "op": "register",
                    "run": str(run),
                    "experiment": str(experiment),
                    "seq": int(seq),
                    "spec_digest": str(spec_digest),
                    "started_at": int(started_at) | float(started_at),
                    "log": str(log),
                    "pid_file": str(pid_file),
                    "exit_file": str(exit_file),
                }:
                    procs[run] = ProposalProcess(
                        run=run,
                        experiment=experiment,
                        seq=seq,
                        spec_digest=spec_digest,
                        started_at=float(started_at),
                        log=Path(log),
                        pid_file=Path(pid_file),
                        exit_file=Path(exit_file),
                    )
                case {"op": "bind", "run": str(run), "pid": int(pid), "pgid": int(pgid)} if run in procs:
                    procs[run] = replace(procs[run], pid=pid, pgid=pgid)
                case {"op": "terminal", "run": str(run), "outcome": str(outcome)} if run in procs:
                    procs[run] = replace(procs[run], outcome=outcome)
                case _:
                    logger.warning("skipping malformed proposal-process registry line in {}", self.path)
        return list(procs.values())

    def events(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        events = []
        for line in self.path.read_bytes().splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line.decode()))
            except (UnicodeDecodeError, ValueError):
                logger.warning("skipping torn proposal-process registry line in {}", self.path)
        return events


@dataclass(frozen=True, slots=True)
class ExperimentProcs:
    """One experiment's handle on the campaign registry: register, bind, finalize.

    Example:
        >>> procs = registry.experiment("001-round1", seq=1, spec_digest=digest)
        >>> procs.register("athome-research-001-round1-abc", log=log, pid_file=pid, exit_file=exit_)
    """

    registry: ProcessRegistry
    experiment: str
    seq: int
    spec_digest: str

    def register(self, run: str, *, log: Path, pid_file: Path, exit_file: Path) -> None:
        self.registry.append(
            {
                "op": "register",
                "run": run,
                "experiment": self.experiment,
                "seq": self.seq,
                "spec_digest": self.spec_digest,
                "started_at": time.time(),
                "log": str(log),
                "pid_file": str(pid_file),
                "exit_file": str(exit_file),
            }
        )

    def bind(self, run: str, *, pid: int, pgid: int) -> None:
        self.registry.append({"op": "bind", "run": run, "pid": pid, "pgid": pgid})

    def finalize(self, run: str, *, outcome: str) -> None:
        self.registry.finalize(run, outcome=outcome)


async def scan(registry: ProcessRegistry, *, repo: Path, alive: Callable[[int], bool] | None = None) -> list[Orphan]:
    """Sweeps non-terminal registry records into live-orphan alarms or abort latches.

    Only call while no campaign runner can be registering: under the campaign lock on
    start/resume, or from the watchdog after its momentary exclusive probe. A live
    pid is returned as a live orphan and left non-terminal so every later scan keeps
    refusing and alarming until it is reconciled; a dead (or never-bound) pid gets
    the experiment's accounting-abort latch written and its record marked terminal.

    Args:
        registry: The campaign's proposal-process registry.
        repo: The git repository experiments ran against (locates each latch).
        alive: Injectable pid-liveness probe; defaults to ``kill -0``.

    Returns:
        Every orphan found, live ones first as recorded.
    """
    alive = alive or pid_alive
    orphans = []
    for record in registry.records():
        if record.outcome is not None:
            continue
        pid = resolve_pid(record)
        latch = await abort_latch(repo, record.experiment)
        if pid is not None and alive(pid):
            orphans.append(Orphan(record, pid=pid, live=True, latch=latch))
            continue
        await atomic_write_text(
            anyio.Path(latch),
            json.dumps(
                {
                    "reason": f"orphaned detached proposal {record.run} has no accounting trace; "
                    "check the provider ledger, reconcile the spend, then delete the latch file",
                    "run": record.run,
                    "ts": time.time(),
                }
            ),
        )
        registry.finalize(record.run, outcome="orphaned")
        orphans.append(Orphan(record, pid=pid, live=False, latch=latch))
    return orphans


async def unreconciled(registry: ProcessRegistry, *, repo: Path) -> list[Orphan]:
    """Returns previously latched orphans whose abort latch has not been reconciled away."""
    stale = []
    for record in registry.records():
        if record.outcome != "orphaned":
            continue
        latch = await abort_latch(repo, record.experiment)
        if await anyio.Path(latch).exists():
            stale.append(Orphan(record, pid=record.pid, live=False, latch=latch))
    return stale
