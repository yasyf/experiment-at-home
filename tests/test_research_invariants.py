from __future__ import annotations

import math

import pytest

from athome.research.errors import ResearchError
from athome.research.invariants import (
    ConstantDeciderViolation,
    DraftDiversityViolation,
    NanGuardViolation,
    constant_decider,
    draft_diversity,
    nan_guard,
)


def test_draft_diversity_rejects_degenerate_batch() -> None:
    with pytest.raises(DraftDiversityViolation, match="constant output"):
        draft_diversity(["same"] * 5)
    with pytest.raises(DraftDiversityViolation, match="no drafts"):
        draft_diversity([])
    draft_diversity(["a", "b", "a"])  # diverse -> no raise


def test_nan_guard_rejects_non_finite_including_nested() -> None:
    with pytest.raises(NanGuardViolation, match="auc"):
        nan_guard({"auc": math.nan})
    with pytest.raises(NanGuardViolation, match="nested.loss"):
        nan_guard({"nested": {"loss": math.inf}})
    nan_guard({"auc": 0.9, "flag": True, "note": "ok", "nested": {"loss": 0.1}})  # finite -> no raise


def test_constant_decider_rejects_single_label() -> None:
    with pytest.raises(ConstantDeciderViolation, match="fires"):
        constant_decider([True] * 4)
    with pytest.raises(ConstantDeciderViolation, match="abstains"):
        constant_decider([False, False])
    with pytest.raises(ConstantDeciderViolation, match="no decisions"):
        constant_decider([])
    constant_decider([True, False, True])  # mixed -> no raise


def test_violations_are_research_errors() -> None:
    assert issubclass(DraftDiversityViolation, ResearchError)
    assert issubclass(NanGuardViolation, ResearchError)
    assert issubclass(ConstantDeciderViolation, ResearchError)
