from __future__ import annotations

import math

import numpy as np
import pytest

from athome.train.gate import (
    GateResult,
    corrected_gate,
    matched_fire_mask,
    sentinel_auc,
    sign_test_p,
    threshold_for_budget,
)


def mask(indices: list[int], n: int) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    out[indices] = True
    return out


def golden_frame() -> dict[str, np.ndarray]:
    n = 20
    candidate_no_fire_probs = np.full(n, 0.9)
    candidate_no_fire_probs[:7] = np.linspace(0.01, 0.07, 7)
    candidate_no_fire_probs[7:10] = np.array([0.08, 0.09, 0.10])
    candidate_no_fire_probs[10:] = np.linspace(0.5, 0.99, 10)
    incumbent_no_fire_probs = np.full(n, 0.9)
    incumbent_no_fire_probs[10:17] = 0.1
    corrective = mask(list(range(10)), n)
    prose = np.ones(n, dtype=bool)
    return {
        "candidate_fire_scores": 1.0 - candidate_no_fire_probs,
        "incumbent_fire_scores": 1.0 - incumbent_no_fire_probs,
        "labels": mask(list(range(10)), n),
        "warranted": corrective & prose,
    }


class TestSignTestP:
    def test_no_discordant_pairs_is_one(self) -> None:
        assert sign_test_p(0, 0) == 1.0

    def test_symmetric_split_is_one(self) -> None:
        assert sign_test_p(3, 3) == 1.0

    def test_lopsided_split_matches_golden_value(self) -> None:
        assert sign_test_p(7, 0) == 0.015625

    @pytest.mark.parametrize(("wins", "losses"), [(-1, 0), (0, -1)])
    def test_negative_counts_raise(self, wins: int, losses: int) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            sign_test_p(wins, losses)


class TestThresholdForBudget:
    def test_matches_exceedance_count(self) -> None:
        scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95], dtype=np.float64)
        assert threshold_for_budget(scores, fires_per_100=40, total_turns=5) == 0.9

    def test_zero_budget_pushes_above_max(self) -> None:
        scores = np.array([0.1, 0.5, 0.9], dtype=np.float64)
        assert threshold_for_budget(scores, fires_per_100=0, total_turns=3) > 0.9

    def test_all_ties_are_excluded_when_the_tie_exceeds_budget(self) -> None:
        scores = np.full(3, 0.5)
        threshold = threshold_for_budget(scores, fires_per_100=200 / 3, total_turns=3)
        assert (scores >= threshold).tolist() == [False, False, False]

    def test_empty_scores_raise(self) -> None:
        with pytest.raises(ValueError, match="scores must be non-empty"):
            threshold_for_budget(np.array([]), fires_per_100=1, total_turns=1)


class TestMatchedFireMask:
    def test_fires_highest_explicit_fire_scores_within_budget(self) -> None:
        scores = np.array([0.99, 0.98, 0.97, 0.1, 0.2], dtype=np.float64)
        assert matched_fire_mask(scores, budget_fires=2).tolist() == [True, True, False, False, False]

    @pytest.mark.parametrize(("budget", "expected"), [(0, [False]), (1, [True])])
    def test_tiny_sample(self, budget: int, expected: list[bool]) -> None:
        assert matched_fire_mask(np.array([0.42]), budget_fires=budget).tolist() == expected

    def test_integer_budget_round_trips_exactly(self) -> None:
        scores = np.linspace(0.0, 1.0, 19)
        assert matched_fire_mask(scores, budget_fires=5).sum() == 5


class TestSentinelAuc:
    def test_higher_scores_explicitly_mean_fire(self) -> None:
        labels = np.array([True, True, False, False])
        fire_scores = np.array([0.9, 0.8, 0.2, 0.1])
        assert sentinel_auc(labels, fire_scores) == 1.0

    def test_degenerate_labels_return_nan(self) -> None:
        with pytest.warns(UserWarning, match="Only one class is present"):
            auc = sentinel_auc(np.array([False, False]), np.array([0.1, 0.2]))
        assert math.isnan(auc)


class TestCorrectedGate:
    def test_cc_steer_golden_replay_matches_every_field(self) -> None:
        frame = golden_frame()
        result = corrected_gate(
            frame["candidate_fire_scores"],
            frame["incumbent_fire_scores"],
            candidate="cand",
            incumbent="inc",
            incumbent_fire_threshold=1.0 - 0.5,
            labels=frame["labels"],
            warranted=frame["warranted"],
            harmful_favors_incumbent=False,
        )
        assert result == GateResult(
            candidate="cand",
            incumbent="inc",
            coverage_wins=7,
            coverage_losses=0,
            coverage_sign_p=0.015625,
            coverage_sig=True,
            budget_held=True,
            cell_auc=1.0,
            incumbent_auc=0.15000000000000002,
            auc_not_regressed=True,
            harmful_favors_incumbent=False,
            promote=True,
        )

    def test_empty_warranted_stratum_cannot_promote(self) -> None:
        frame = golden_frame()
        result = corrected_gate(
            frame["candidate_fire_scores"],
            frame["incumbent_fire_scores"],
            candidate="cand",
            incumbent="inc",
            incumbent_fire_threshold=0.5,
            labels=frame["labels"],
            warranted=np.zeros(20, dtype=bool),
            harmful_favors_incumbent=False,
        )
        assert (result.coverage_wins, result.coverage_losses, result.coverage_sign_p) == (0, 0, 1.0)
        assert result.coverage_sig is False
        assert result.promote is False

    def test_harmful_judging_pending_leaves_promotion_pending(self) -> None:
        frame = golden_frame()
        result = corrected_gate(
            frame["candidate_fire_scores"],
            frame["incumbent_fire_scores"],
            candidate="cand",
            incumbent="inc",
            incumbent_fire_threshold=0.5,
            labels=frame["labels"],
            warranted=frame["warranted"],
        )
        assert result.harmful_favors_incumbent is None
        assert result.promote is None

    def test_degenerate_auc_fails_closed(self) -> None:
        scores = np.array([0.9, 0.8, 0.2])
        with pytest.warns(UserWarning, match="Only one class is present"):
            result = corrected_gate(
                scores,
                scores,
                candidate="cand",
                incumbent="inc",
                incumbent_fire_threshold=0.5,
                labels=np.ones(3, dtype=bool),
                warranted=np.ones(3, dtype=bool),
                harmful_favors_incumbent=False,
            )
        assert math.isnan(result.cell_auc)
        assert math.isnan(result.incumbent_auc)
        assert result.auc_not_regressed is False
        assert result.promote is False
