"""Ceiling/floor score normalization with saturation refusals.

A metric is calibrated by an author-side ceiling (its score sample when it should
score high) and a foreign floor (its sample when it should score low); raw scores
then normalize into ``[0, 1]`` between the two. The headline guard is a saturation
refusal: a metric pinned at its ceiling — one whose ceiling and floor collapse onto
the same value, or that sits at a known maximum — cannot discriminate, so it
refuses to calibrate rather than manufacture a spread out of noise. Donor:
write-like-me ``evals/calibrate.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import TYPE_CHECKING

from athome.research.errors import ResearchError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

MIN_SPREAD = 1e-9
SATURATION_EPSILON = 1e-9


class CalibrationError(ResearchError):
    """The root of a calibration failure."""


class SaturationRefusal(CalibrationError):
    """A metric is pinned at its ceiling and cannot discriminate; calibration refuses to report a fake spread."""


@dataclass(frozen=True, slots=True)
class MetricCalibration:
    """One metric's calibrated bounds, mapping raw scores onto ``[0, 1]``.

    Attributes:
        ceiling: The score the metric reaches when it should score high.
        floor: The score the metric reaches when it should score low.

    Example:
        >>> MetricCalibration(ceiling=0.9, floor=0.1).normalize(0.5)
        0.5
    """

    ceiling: float
    floor: float

    def normalize(self, score: float) -> float:
        """Map ``score`` onto ``[0, 1]`` between the floor and ceiling, clamped at both ends."""
        return min(1.0, max(0.0, (score - self.floor) / (self.ceiling - self.floor)))


@dataclass(frozen=True, slots=True)
class Calibration:
    """A calibrated bound per metric name.

    Example:
        >>> Calibration.from_dict(payload).normalize("cosine", 0.5)
    """

    metrics: dict[str, MetricCalibration]

    def normalize(self, name: str, score: float) -> float:
        return self.metrics[name].normalize(score)

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {name: {"ceiling": cal.ceiling, "floor": cal.floor} for name, cal in self.metrics.items()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Mapping[str, float]]) -> Calibration:
        return cls({name: MetricCalibration(bounds["ceiling"], bounds["floor"]) for name, bounds in payload.items()})


def calibrate_metric(
    name: str,
    ceiling_scores: Sequence[float],
    floor_scores: Sequence[float],
    *,
    min_spread: float = MIN_SPREAD,
    saturation_at: float | None = None,
) -> MetricCalibration:
    """Calibrate one metric from its ceiling and floor score samples, refusing a saturated metric.

    Args:
        ceiling_scores: The metric's scores where it should score high.
        floor_scores: The metric's scores where it should score low.
        min_spread: The smallest ceiling-minus-floor gap that still discriminates.
        saturation_at: A known maximum the metric cannot exceed; a ceiling that reaches it is saturated.

    Raises:
        SaturationRefusal: the bounds are non-finite, the ceiling sits at ``saturation_at``,
            or the ceiling-floor spread does not clear ``min_spread``.
    """
    ceiling, floor = fmean(ceiling_scores), fmean(floor_scores)
    if not (math.isfinite(ceiling) and math.isfinite(floor)):
        raise SaturationRefusal(f"metric {name!r} produced a non-finite ceiling/floor ({ceiling}, {floor})")
    if saturation_at is not None and ceiling >= saturation_at - SATURATION_EPSILON:
        raise SaturationRefusal(
            f"metric {name!r} is pinned at its ceiling {saturation_at} (ceiling {ceiling:.4g}); "
            "refusing to report a fake spread"
        )
    if ceiling - floor <= min_spread:
        raise SaturationRefusal(
            f"metric {name!r} cannot discriminate: ceiling {ceiling:.4g} - floor {floor:.4g} <= {min_spread}"
        )
    return MetricCalibration(ceiling=ceiling, floor=floor)


def calibrate(
    samples: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    *,
    min_spread: float = MIN_SPREAD,
    saturation_at: float | None = None,
) -> Calibration:
    """Calibrate every metric from its ``(ceiling_scores, floor_scores)`` pair; a saturated metric raises."""
    return Calibration(
        {
            name: calibrate_metric(name, ceiling, floor, min_spread=min_spread, saturation_at=saturation_at)
            for name, (ceiling, floor) in samples.items()
        }
    )
