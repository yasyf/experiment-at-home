"""Blocking preflight invariants: validity predicates checked before any LLM spend.

Each check raises a typed :class:`~athome.research.errors.ResearchError` subclass on
its trigger. Born from real post-mortems: a batch of byte-identical drafts burned a
run plus hundreds of judge calls, and a smoke diagnostic journaled a kill-the-model
verdict off an AUC of NaN.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

from athome.research.errors import ResearchError

if TYPE_CHECKING:
    from collections.abc import Sequence


class DraftDiversityViolation(ResearchError):
    """A candidate batch is empty or every draft is byte-identical (constant output)."""


class NanGuardViolation(ResearchError):
    """A metric value is NaN or infinite."""


class ConstantDeciderViolation(ResearchError):
    """A decider returns one label for every row (or has no decisions)."""


def draft_diversity(drafts: Sequence[str]) -> None:
    """Reject a degenerate batch: an empty batch, or N byte-identical drafts.

    Raises:
        DraftDiversityViolation: the batch is empty or every draft is the same string.
    """
    match len(set(draft.strip() for draft in drafts)):
        case 0:
            raise DraftDiversityViolation("no drafts")
        case 1:
            raise DraftDiversityViolation(f"all {len(drafts)} drafts are one string (constant output)")


def nan_guard(metrics: Mapping[str, object]) -> None:
    """Reject metrics containing a NaN or infinite value, recursing into nested mappings.

    Raises:
        NanGuardViolation: one or more metric values are non-finite.
    """
    if bad := sorted(non_finite_keys(metrics, prefix="")):
        raise NanGuardViolation(f"non-finite metrics: {', '.join(bad)}")


def constant_decider(decisions: Sequence[bool]) -> None:
    """Reject a decider whose decisions are empty or all identical.

    Raises:
        ConstantDeciderViolation: the batch is empty or every decision is the same.
    """
    match len(set(decisions)):
        case 0:
            raise ConstantDeciderViolation("no decisions")
        case 1:
            raise ConstantDeciderViolation(
                f"decider {'fires' if decisions[0] else 'abstains'} on all {len(decisions)} rows"
            )


def non_finite_keys(metrics: Mapping[str, object], *, prefix: str) -> list[str]:
    bad: list[str] = []
    for key, value in metrics.items():
        label = f"{prefix}{key}"
        match value:
            case Mapping():
                bad.extend(non_finite_keys(value, prefix=f"{label}."))
            case float() | int() if not isinstance(value, bool) and not math.isfinite(value):
                bad.append(label)
    return bad
