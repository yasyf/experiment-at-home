"""Infra-vs-candidate failure classification: machine trouble is never a research result.

A greedy keep/discard night must never journal a flaky network as failed research
directions — one bad night would poison every future contract-memory and retro round.
So a unit failure is split two ways. A *candidate* fault (an immutable-boundary
violation, a metric of the wrong shape, a proposal timeout, git rejecting the
candidate-controlled tree, or a scorer that computed a bad number) is journaled as
CRASH/DISCARD exactly as before and the loop continues. An *infra* fault (an OS or
harness-side git error on the incumbent, a connection reset, or a scorer that writes no
metric while its withheld run log matches an :data:`INFRA_MARKERS` entry) is never
journaled: the same unit index is retried up to :data:`MAX_INFRA_RETRIES` times, and if
it keeps failing the run aborts loudly with :class:`InfraFailure`.

Origin is what separates the two: git operating on candidate-controlled state (staging
or committing the proposal's tree) is a *candidate* fault via :class:`CandidateFault` —
a candidate that swaps a mutable file for a FIFO must journal CRASH and let the loop
continue, not burn every retry and abort the run — while git on the trusted incumbent
(``archive``/``read-tree`` of committed content) stays infra.

Forgeability bound: candidate code inside ``metric_command`` can print an infra marker
to its run log, but a marker only fires *after* the proposal has been committed, so the
retry re-scores that same immutable commit — it never re-proposes — and re-scoring is
deterministic. Every attempt's proposal spend is accounted exactly once (the journaled
outcome carries its own attempt's cost; a retried or aborted attempt records its cost in
the sidecar), and every budget summation sums both, so the marker buys neither free
dollars nor free wall-clock: the retry count is hard-capped, ``max_usd`` still counts the
spend, and the run aborts. It can never manufacture a journaled row — no forged keep,
discard, or contract-memory feedback — so the worst it achieves is burning its own unit's
budget before the run aborts. Worthless.

Before an accounting-integrity abort escapes a unit, the harness atomically writes a
per-run ``<name>.abort.json`` latch with its unit, reason, and timestamp, then best-effort
appends an ``accounting_abort`` sidecar breadcrumb for M3 reporting. A later run refuses
to start while the latch remains: check the provider ledger, reconcile the spend, then
delete the latch file. The sidecar is reporting evidence, not the restart authority. If
the latch write itself fails I/O, the doubly-degraded F3 residue still aborts loudly, as
do unreadable billing evidence or a detached process that cannot be stopped; check and
reconcile the ledger before resuming.

Accepted limitation: if the latch write fails I/O while the sidecar breadcrumb succeeds,
a restart is not blocked because reconciliation never clears append-only sidecar history.
"""

from __future__ import annotations

import json
import os
import time
from math import isfinite
from subprocess import CalledProcessError
from typing import TYPE_CHECKING, Literal

import anyio
from loguru import logger

from athome.cache import atomic_write_text
from athome.research.errors import AccountingIntegrityError, ResearchError

if TYPE_CHECKING:
    from pathlib import Path

INFRA_MARKERS: tuple[str, ...] = (
    "connection reset",
    "connection refused",
    "connection timed out",
    "network is unreachable",
    "no space left on device",
    "rate limit",
    "too many requests",
    "overloaded",
    "at capacity",
    "insufficient capacity",
    "temporarily unavailable",
    "service unavailable",
    "502 bad gateway",
    "503 service",
    "504 gateway timeout",
)
MAX_INFRA_RETRIES = 2

type SidecarKind = Literal["retry", "wall_cancel", "accounting_abort"]


class InfraFailure(ResearchError):
    """Machine trouble, not a research result: a unit failed on infrastructure.

    Raised from the unit evaluation path when a scorer writes no metric file and its
    withheld run log matches an :data:`INFRA_MARKERS` entry, and re-raised by the loop
    once a unit has exhausted its :data:`MAX_INFRA_RETRIES` infra retries. It aborts the
    run loudly and is never journaled, so a restart resumes the same, un-journaled unit.
    """


class CandidateFault(ResearchError):
    """Git rejected the candidate-controlled tree (e.g. a FIFO swapped into a mutable path).

    A candidate fault, not machine trouble: it is journaled as a CRASH and the loop
    continues, never consuming an infra retry or aborting the run.
    """


def infra_log(log: bytes) -> bool:
    text = log.decode("utf-8", "replace").casefold()
    return any(marker in text for marker in INFRA_MARKERS)


def classify(exc: BaseException) -> Literal["accounting", "infra", "candidate"]:
    match exc:
        case AccountingIntegrityError():
            return "accounting"
        case CandidateFault():
            return "candidate"
        case InfraFailure() | OSError() | CalledProcessError():
            return "infra"
        case _:
            return "candidate"


async def record_infra_event(
    path: Path,
    *,
    unit: int,
    attempt: int,
    reason: str,
    cost: float,
    kind: Literal["retry", "wall_cancel"],
) -> None:
    append_event(path, {"unit": unit, "attempt": attempt, "reason": reason, "cost": cost}, kind=kind)


async def record_accounting_abort(latch: Path, events: Path, *, unit: int, reason: str) -> None:
    with anyio.CancelScope(shield=True):
        try:
            await atomic_write_text(
                anyio.Path(latch),
                json.dumps({"unit": unit, "reason": reason, "ts": time.time()}),
            )
        except OSError:
            logger.exception("could not write accounting-abort latch to {}", latch)
        try:
            append_event(events, {"unit": unit, "reason": reason}, kind="accounting_abort")
        except OSError:
            logger.exception("could not write accounting-abort breadcrumb to {}", events)


def append_event(path: Path, event: dict[str, object], *, kind: SidecarKind) -> None:
    payload = (json.dumps(event | {"kind": kind}) + "\n").encode()
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        if (size := os.fstat(fd).st_size) and os.pread(fd, 1, size - 1) != b"\n":
            os.write(fd, b"\n")  # heal a torn prior line so this record never concatenates onto it
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
    finally:
        os.close(fd)


def infra_events(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [record for line in path.read_bytes().splitlines() if (record := parse_event(line, path)) is not None]


def parse_event(line: bytes, path: Path) -> dict[str, object] | None:
    try:
        record = json.loads(line.decode())
    except (UnicodeDecodeError, ValueError):
        logger.warning("skipping malformed infra sidecar line in {}", path)
        return None
    match record:
        case {"kind": "retry" | "wall_cancel" | "accounting_abort"}:
            return record
        case _:
            logger.warning("skipping malformed infra sidecar line in {}", path)
            return None


def infra_retries(path: Path) -> int:
    return sum(event["kind"] == "retry" for event in infra_events(path))


def accounting_aborts(path: Path) -> int:
    return sum(event["kind"] == "accounting_abort" for event in infra_events(path))


def sidecar_cost(value: object) -> float | None:
    match value:
        case bool():
            return None
        case int() | float():
            try:
                cost = float(value)
            except Exception:
                # This parser is total so malformed sidecar rows are skipped independently.
                return None
            return cost if isfinite(cost) and cost >= 0 else None
        case _:
            return None


def infra_cost(path: Path) -> float:
    try:
        events = infra_events(path)
    except OSError as exc:
        raise AccountingIntegrityError(f"could not read infra spend sidecar {path}") from exc
    return sum(cost for event in events if (cost := sidecar_cost(event.get("cost"))) is not None)
