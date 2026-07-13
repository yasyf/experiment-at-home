from __future__ import annotations

import math

import pytest

from athome.research.calibrate import (
    Calibration,
    MetricCalibration,
    SaturationRefusal,
    calibrate,
    calibrate_metric,
)


def test_calibrate_metric_computes_mean_bounds() -> None:
    cal = calibrate_metric("cosine", [0.8, 0.9, 1.0], [0.1, 0.2, 0.0])
    assert cal.ceiling == pytest.approx(0.9)
    assert cal.floor == pytest.approx(0.1)


@pytest.mark.parametrize(
    "score, expected",
    [
        pytest.param(0.1, 0.0, id="at-floor"),
        pytest.param(0.9, 1.0, id="at-ceiling"),
        pytest.param(0.5, 0.5, id="midpoint"),
        pytest.param(0.0, 0.0, id="below-floor-clamps"),
        pytest.param(1.5, 1.0, id="above-ceiling-clamps"),
    ],
)
def test_normalize_maps_between_floor_and_ceiling(score: float, expected: float) -> None:
    assert MetricCalibration(ceiling=0.9, floor=0.1).normalize(score) == pytest.approx(expected)


def test_saturated_metric_pinned_at_ceiling_refuses() -> None:
    with pytest.raises(SaturationRefusal, match="pinned at its ceiling"):
        calibrate_metric("saturated", [1.0, 1.0, 1.0], [0.9, 0.95, 0.9], saturation_at=1.0)


def test_collapsed_spread_refuses_rather_than_faking_it() -> None:
    with pytest.raises(SaturationRefusal, match="cannot discriminate"):
        calibrate_metric("flat", [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


def test_inverted_bounds_refuse() -> None:
    with pytest.raises(SaturationRefusal, match="cannot discriminate"):
        calibrate_metric("inverted", [0.2, 0.2], [0.8, 0.8])


def test_non_finite_bounds_refuse() -> None:
    with pytest.raises(SaturationRefusal, match="non-finite"):
        calibrate_metric("nan", [math.nan], [0.1])


def test_calibrate_batches_and_normalizes_by_name() -> None:
    calibration = calibrate({"a": ([0.9], [0.1]), "b": ([0.6], [0.2])})
    assert calibration.normalize("a", 0.5) == pytest.approx(0.5)
    assert calibration.normalize("b", 0.4) == pytest.approx(0.5)


def test_calibrate_surfaces_a_saturated_metric_in_the_batch() -> None:
    with pytest.raises(SaturationRefusal):
        calibrate({"good": ([0.9], [0.1]), "bad": ([1.0], [1.0])}, saturation_at=1.0)


def test_calibration_round_trips_through_a_dict() -> None:
    calibration = calibrate({"a": ([0.9], [0.1])})
    restored = Calibration.from_dict(calibration.as_dict())
    assert restored.as_dict() == calibration.as_dict()
    assert restored.normalize("a", 0.5) == pytest.approx(0.5)
