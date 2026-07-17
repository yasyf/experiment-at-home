"""Promotion-gate statistics for trained-model candidates.

Generic, framework-free gate math (sign tests, budgeted thresholds, sentinel AUC,
correction-aware gating) shared by any consumer that compares a candidate model
against an incumbent. Domain strata arrive as caller-supplied masks; nothing here
knows what a "corrective" or "prose" row is. Heavy statistical dependencies load
lazily behind the ``gate`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """The gate's decision over a candidate's scores.

    Attributes:
        promote: Whether the candidate should replace the incumbent.
        reason: Human-readable justification recorded alongside the decision.
        stats: The named statistics the decision was derived from.
    """

    promote: bool
    reason: str
    stats: dict[str, float]


@dataclass(frozen=True, slots=True)
class GateResult:
    """The corrected paired gate over a candidate and incumbent at matched budget.

    Attributes:
        candidate: The candidate's name.
        incumbent: The incumbent's name.
        coverage_wins: Warranted rows where only the candidate fires.
        coverage_losses: Warranted rows where only the incumbent fires.
        coverage_sign_p: Exact sign-test p-value over discordant coverage pairs.
        coverage_sig: Whether coverage significantly favors the candidate.
        budget_held: Whether candidate fires do not exceed incumbent fires.
        cell_auc: The candidate's sentinel AUC.
        incumbent_auc: The incumbent's sentinel AUC.
        auc_not_regressed: Whether candidate AUC is at least incumbent AUC.
        harmful_favors_incumbent: Whether harmful-fire judging favors the incumbent,
            or None while judging is pending.
        promote: The full verdict, or None while harmful-fire judging is pending.
    """

    candidate: str
    incumbent: str
    coverage_wins: int
    coverage_losses: int
    coverage_sign_p: float
    coverage_sig: bool
    budget_held: bool
    cell_auc: float
    incumbent_auc: float
    auc_not_regressed: bool
    harmful_favors_incumbent: bool | None
    promote: bool | None


def sign_test_p(wins: int, losses: int) -> float:
    """Return the exact two-sided sign-test p-value over discordant pairs.

    Args:
        wins: Discordant pairs favoring the candidate.
        losses: Discordant pairs favoring the incumbent.

    Returns:
        The exact two-sided p-value, or 1.0 when there are no discordant pairs.

    Raises:
        ValueError: If either count is negative.
    """
    from scipy.stats import binomtest

    if wins < 0 or losses < 0:
        raise ValueError(f"wins/losses must be >= 0, got {wins}/{losses}")
    if (n := wins + losses) == 0:
        return 1.0
    return float(binomtest(wins, n, 0.5, alternative="two-sided").pvalue)


def _threshold_for_count(scores: np.ndarray, *, budget: int) -> float:
    """Return the lowest threshold admitting at most ``budget`` scores, ties excluded conservatively."""
    import numpy as np

    scores_sorted = np.sort(scores)
    values = np.unique(scores_sorted)
    count_ge = scores.size - np.searchsorted(scores_sorted, values, side="left")
    within = values[count_ge <= budget]
    if within.size:
        return float(within[0])
    return float(np.nextafter(values[-1], np.inf))


def threshold_for_budget(scores: np.ndarray, *, fires_per_100: float, total_turns: int) -> float:
    """Return a threshold whose exceedance count stays within an alert budget.

    The threshold fires where ``score >= threshold``. It is conservative around
    ties: a tied score is excluded when including the whole tie would exceed the
    budget.

    Args:
        scores: Scores whose larger values are more eligible to fire.
        fires_per_100: Maximum fires allowed per 100 total turns.
        total_turns: Turn count used to convert the rate into an integer budget.

    Returns:
        The lowest threshold that admits as many scores as possible within budget.

    Raises:
        ValueError: If scores are empty, total_turns is not positive, or the rate
            is negative.
    """
    import numpy as np

    scores = np.asarray(scores, dtype=np.float64).ravel()
    if scores.size == 0:
        raise ValueError("scores must be non-empty")
    if total_turns <= 0:
        raise ValueError(f"total_turns must be > 0, got {total_turns}")
    if fires_per_100 < 0:
        raise ValueError(f"fires_per_100 must be >= 0, got {fires_per_100}")
    budget = int(np.floor(fires_per_100 * total_turns / 100.0))
    return _threshold_for_count(scores, budget=budget)


def matched_fire_mask(fire_scores: np.ndarray, *, budget_fires: int) -> np.ndarray:
    """Return a higher-is-fire mask matched conservatively to a fire budget.

    Args:
        fire_scores: Explicitly oriented scores where larger values mean a row is
            more likely to fire.
        budget_fires: Maximum number of rows that may fire.

    Returns:
        A boolean mask firing on the highest scores without splitting ties.

    Raises:
        ValueError: If fire_scores is empty or budget_fires is negative.
    """
    import numpy as np

    scores = np.asarray(fire_scores, dtype=np.float64).ravel()
    if scores.size == 0:
        raise ValueError("fire_scores must be non-empty")
    if budget_fires < 0:
        raise ValueError(f"budget_fires must be >= 0, got {budget_fires}")
    threshold = _threshold_for_count(scores, budget=budget_fires)
    return fire_scores >= threshold


def sentinel_auc(labels: np.ndarray, fire_scores: np.ndarray) -> float:
    """Return ROC-AUC for explicitly oriented higher-is-fire scores.

    Args:
        labels: Binary labels where true or 1 identifies rows that should fire.
        fire_scores: Scores where larger values mean a row is more likely to fire.

    Returns:
        The ROC-AUC reported by scikit-learn.
    """
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels.tolist(), fire_scores.tolist()))


def corrected_gate(
    candidate_fire_scores: np.ndarray,
    incumbent_fire_scores: np.ndarray,
    *,
    candidate: str,
    incumbent: str,
    incumbent_fire_threshold: float,
    labels: np.ndarray,
    warranted: np.ndarray,
    harmful_favors_incumbent: bool | None = None,
) -> GateResult:
    """Evaluate the corrected paired gate over common rows at matched budget.

    Every score input has an explicit higher-is-fire contract. Callers starting
    from a no-fire probability must orient it before calling this function. The
    incumbent fires strictly above ``incumbent_fire_threshold``; the candidate is
    then matched conservatively to that fire count. ``warranted`` selects the
    caller-defined stratum in which discordant coverage pairs count.

    Args:
        candidate_fire_scores: Candidate scores where larger values mean fire.
        incumbent_fire_scores: Incumbent scores where larger values mean fire.
        candidate: Candidate name recorded in the result.
        incumbent: Incumbent name recorded in the result.
        incumbent_fire_threshold: Strict lower bound for incumbent fires.
        labels: Binary labels used for both sentinel AUC calculations.
        warranted: Boolean mask selecting rows where coverage wins and losses count.
        harmful_favors_incumbent: Deferred harmful-fire judgment, or None while
            pending.

    Returns:
        Every gate component and the promotion verdict when harmful judging exists.
    """
    incumbent_fires = incumbent_fire_scores > incumbent_fire_threshold
    candidate_fires = matched_fire_mask(candidate_fire_scores, budget_fires=int(incumbent_fires.sum()))
    wins = int((candidate_fires & ~incumbent_fires & warranted).sum())
    losses = int((incumbent_fires & ~candidate_fires & warranted).sum())
    coverage_p = sign_test_p(wins, losses)
    coverage_sig = coverage_p < 0.05 and wins > losses
    budget_held = int(candidate_fires.sum()) <= int(incumbent_fires.sum())
    cell_auc = sentinel_auc(labels, candidate_fire_scores)
    incumbent_auc = sentinel_auc(labels, incumbent_fire_scores)
    auc_not_regressed = cell_auc >= incumbent_auc
    free_pass = coverage_sig and budget_held and auc_not_regressed
    promote = None if harmful_favors_incumbent is None else (free_pass and not harmful_favors_incumbent)
    return GateResult(
        candidate=candidate,
        incumbent=incumbent,
        coverage_wins=wins,
        coverage_losses=losses,
        coverage_sign_p=coverage_p,
        coverage_sig=coverage_sig,
        budget_held=budget_held,
        cell_auc=cell_auc,
        incumbent_auc=incumbent_auc,
        auc_not_regressed=auc_not_regressed,
        harmful_favors_incumbent=harmful_favors_incumbent,
        promote=promote,
    )
