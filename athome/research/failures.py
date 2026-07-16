"""Infra-vs-candidate failure classification: machine trouble is never a research result.

A greedy keep/discard night must never journal a flaky network as failed research
directions — one bad night would poison every future contract-memory and retro round.
So a unit failure is split two ways. A *candidate* fault (an immutable-boundary
violation, a metric of the wrong shape, a proposal timeout, a scorer that computed a
bad number) is journaled as CRASH/DISCARD exactly as before and the loop continues.
An *infra* fault (an OS or git-subprocess error, a connection reset, or a scorer that
writes no metric while its withheld run log matches an :data:`INFRA_MARKERS` entry)
is never journaled: the same unit index is retried up to :data:`MAX_INFRA_RETRIES`
times, and if it keeps failing the run aborts loudly with :class:`InfraFailure`.

Forgeability bound: candidate code inside ``metric_command`` can print an infra marker
to its run log, but that buys it at most :data:`MAX_INFRA_RETRIES` deterministic
re-scores of the same immutable commit — its mutable edits are frozen by the scoring
boundary and re-scoring is deterministic — with the wall-clock budget still bounding
every attempt and the retry count hard-capped. It can never manufacture a journaled
row, so it cannot forge a keep, a discard, or contract-memory feedback; the worst it
achieves is burning its own unit's budget before the run aborts. Worthless.
"""

from __future__ import annotations

from subprocess import CalledProcessError
from typing import TYPE_CHECKING, Literal

from athome.progress import append_line, load_journal
from athome.research.errors import ResearchError

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


def infra_log(log: bytes) -> bool:
    text = log.decode("utf-8", "replace").casefold()
    return any(marker in text for marker in INFRA_MARKERS)


def classify(exc: BaseException) -> Literal["infra", "candidate"]:
    match exc:
        case InfraFailure() | OSError() | CalledProcessError():
            return "infra"
        case _:
            return "candidate"


async def record_infra_event(path: Path, *, unit: int, attempt: int, reason: str) -> None:
    await append_line(path, {"unit": unit, "attempt": attempt, "reason": reason})


def infra_retries(path: Path) -> int:
    return len(load_journal(path))
