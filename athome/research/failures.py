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

If a wall-cancel cost flush fails I/O, billing evidence cannot be parsed, or the
detached process cannot be stopped, the run aborts loudly without recording that cost;
check and reconcile the ledger before resuming.
"""

from __future__ import annotations

import json
import os
from math import isfinite
from subprocess import CalledProcessError
from typing import TYPE_CHECKING, Literal

from loguru import logger

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
    payload = (
        json.dumps({"unit": unit, "attempt": attempt, "reason": reason, "cost": cost, "kind": kind}) + "\n"
    ).encode()
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
    except (UnicodeDecodeError, json.JSONDecodeError):
        logger.warning("skipping malformed infra sidecar line in {}", path)
        return None
    match record:
        case {"kind": "retry" | "wall_cancel"}:
            return record
        case _:
            logger.warning("skipping malformed infra sidecar line in {}", path)
            return None


def infra_retries(path: Path) -> int:
    return sum(event["kind"] == "retry" for event in infra_events(path))


def infra_cost(path: Path) -> float:
    return sum(
        float(cost)
        for event in infra_events(path)
        if isinstance(cost := event.get("cost"), (int, float))
        and not isinstance(cost, bool)
        and isfinite(cost)
        and cost >= 0
    )
