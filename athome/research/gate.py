from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

SYMLINK_MODE = "120000"
AUTOLOADERS = frozenset({"sitecustomize.py", "usercustomize.py", "conftest.py", "__init__.py"})


@dataclass(frozen=True, slots=True)
class TreeChange:
    """One changed path between two commit trees, with its destination file mode.

    Attributes:
        path: The repository-relative path that changed.
        dst_mode: The git mode of the path in the candidate tree (``000000`` for a
            deletion, ``120000`` for a symlink).
    """

    path: str
    dst_mode: str


def parse_diff_tree(raw: str) -> list[TreeChange]:
    parts = [part for part in raw.split("\0") if part]
    return [TreeChange(path=parts[i + 1], dst_mode=parts[i][1:].split(" ")[1]) for i in range(0, len(parts), 2)]


def matches(path: str, patterns: tuple[str, ...]) -> bool:
    return any(PurePosixPath(path).full_match(pattern) for pattern in patterns)


def is_autoloader(path: str) -> bool:
    return PurePosixPath(path).name in AUTOLOADERS or PurePosixPath(path).match("*.pth")


def immutable_violations(
    changes: Sequence[TreeChange], *, mutable: tuple[str, ...], immutable: tuple[str, ...]
) -> list[str]:
    """Returns the changed paths that break the scoring boundary.

    A path violates the boundary when it is a symlink, when it matches an
    ``immutable`` pattern, when it falls outside the ``mutable`` allowlist, or
    when it is a Python auto-loader (``__init__.py``, ``conftest.py``,
    ``sitecustomize.py``, ``usercustomize.py``, a ``*.pth`` file) that would run
    code merely by existing on ``sys.path`` at scoring time — so a rename or
    deletion of an immutable file, an undeclared new file, and an added or
    modified auto-loader are all caught. ``full_match`` gives recursive ``**``
    semantics, so ``eval/**`` matches ``eval/a/b.py``.
    """
    return [
        change.path
        for change in changes
        if change.dst_mode == SYMLINK_MODE
        or matches(change.path, immutable)
        or not matches(change.path, mutable)
        or is_autoloader(change.path)
    ]


def monotone_gate(
    candidate: float, incumbent: float | None, *, direction: Literal["min", "max"], floor: float = 0.0
) -> bool:
    """Keeps a candidate only when it strictly beats the incumbent by more than ``floor``.

    The first candidate (no incumbent) is always kept. Otherwise the candidate
    must improve the monotone metric — larger for ``max``, smaller for ``min`` —
    by a margin exceeding ``floor``.
    """
    if incumbent is None:
        return True
    match direction:
        case "max":
            return candidate - incumbent > floor
        case "min":
            return incumbent - candidate > floor


@dataclass(frozen=True, slots=True)
class PromotionVerdict:
    """The bootstrap-CI gate's decision and its rationale.

    Attributes:
        promote: Whether the candidate should replace the incumbent.
        reason: A human-readable explanation of the decision.
    """

    promote: bool
    reason: str


def bootstrap_ci_gate(
    incumbent: Sequence[float],
    candidate: Sequence[float],
    *,
    direction: Literal["min", "max"],
    n_boot: int = 10_000,
    floor: float = 0.0,
) -> PromotionVerdict:
    """Promotes a candidate only on a clear, floored win; overlapping CIs keep the incumbent.

    Bootstrap-resamples the mean of each sample ``n_boot`` times (seeded, so a
    given pair of samples always yields the same verdict), then promotes only
    when the 95% confidence intervals are disjoint, the candidate lands on the
    winning side, and its mean margin exceeds ``floor``.
    """
    from numpy import asarray, percentile
    from numpy.random import default_rng

    rng = default_rng(0)
    inc, cand = asarray(incumbent, dtype=float), asarray(candidate, dtype=float)
    inc_boot = rng.choice(inc, size=(n_boot, inc.size)).mean(axis=1)
    cand_boot = rng.choice(cand, size=(n_boot, cand.size)).mean(axis=1)
    inc_lo, inc_hi = percentile(inc_boot, [2.5, 97.5])
    cand_lo, cand_hi = percentile(cand_boot, [2.5, 97.5])
    margin = float(cand_boot.mean() - inc_boot.mean()) * (1 if direction == "max" else -1)
    if inc_lo <= cand_hi and cand_lo <= inc_hi:
        return PromotionVerdict(False, "overlapping confidence intervals; incumbent stays")
    if margin > floor:
        return PromotionVerdict(True, f"disjoint CIs, candidate wins by {margin:.4g} (> floor {floor:.4g})")
    return PromotionVerdict(False, f"disjoint CIs but margin {margin:.4g} does not clear floor {floor:.4g}")


def blocking_invariants(rows: Sequence[object], checks: Sequence[Callable]) -> None:
    """Runs each pre-flight check; a failing check raises its typed error before any verdict is journaled."""
    for check in checks:
        check(rows)
