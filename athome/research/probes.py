"""Topic-leakage probes: detect a judge that keys on topic rather than quality.

Two style-preserving topic swaps expose a topic-leaky scorer. The style-swap arm
rewrites on-topic author text onto a foreign topic (style held, topic made
foreign); the topic-match arm rewrites foreign text onto the author's topic (style
foreign, topic made the author's). A topic-pure judge scores by quality and barely
moves; a topic-leaky judge drops the author's own text once its topic changes and
inflates foreign text once it wears the author's topic. Both deltas are expressed
as fractions of the judge's own author-vs-foreign spread, so magnitudes compare
across scorers in different native units. Donor: write-like-me ``lab/probes.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import TYPE_CHECKING

from athome.research.errors import ResearchError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

LEAKAGE_TOLERANCE = 0.5


class ProbeError(ResearchError):
    """The probe cannot normalize: the judge's author-vs-foreign spread is zero or non-finite."""


class TopicLeakageViolation(ResearchError):
    """A judge keys on topic rather than quality: a leakage fraction exceeded the tolerance."""


@dataclass(frozen=True, slots=True)
class LeakageReport:
    """One judge's topic-leakage profile under style-preserving topic swaps.

    ``style_swap_drop`` and ``topic_match_inflation`` are fractions of the judge's
    own ``author_baseline - foreign_baseline`` spread. A topic-pure judge leaves both
    near zero; a topic-leaky judge shows a large positive ``style_swap_drop`` (author
    text loses a spread's worth of quality when only its topic changes) and a large
    positive ``topic_match_inflation`` (foreign text gains a spread's worth when only
    its topic becomes the author's).

    Attributes:
        author_baseline: Mean score of on-topic author text.
        foreign_baseline: Mean score of off-topic foreign text.
        style_swap_score: Mean score of author text rewritten onto the foreign topic.
        topic_match_score: Mean score of foreign text rewritten onto the author's topic.
        style_swap_drop: The author-text quality lost to the topic swap, as a fraction of spread.
        topic_match_inflation: The foreign-text quality gained by the topic swap, as a fraction of spread.
    """

    author_baseline: float
    foreign_baseline: float
    style_swap_score: float
    topic_match_score: float
    style_swap_drop: float
    topic_match_inflation: float

    @property
    def leakage(self) -> float:
        return max(self.style_swap_drop, self.topic_match_inflation)

    def check(self, *, tolerance: float = LEAKAGE_TOLERANCE) -> None:
        """Flag a topic-leaky judge.

        Raises:
            TopicLeakageViolation: a leakage fraction exceeded ``tolerance``.
        """
        if self.leakage > tolerance:
            raise TopicLeakageViolation(
                f"topic leakage {self.leakage:.3f} > {tolerance} "
                f"(style-swap drop {self.style_swap_drop:.3f}, topic-match inflation {self.topic_match_inflation:.3f})"
            )


def topic_leakage(
    score: Callable[[str], float],
    *,
    author_texts: Sequence[str],
    foreign_texts: Sequence[str],
    style_swapped: Sequence[str],
    topic_matched: Sequence[str],
) -> LeakageReport:
    """Score the baselines and the two topic-swapped arms, normalizing the leakage deltas by spread.

    Args:
        score: The judge's scalar quality score for a passage.
        author_texts: On-topic author passages (the high baseline).
        foreign_texts: Off-topic foreign passages (the low baseline).
        style_swapped: Author passages rewritten onto the foreign topic (style held).
        topic_matched: Foreign passages rewritten onto the author's topic (style foreign).

    Raises:
        ProbeError: the author-vs-foreign baseline spread is zero or non-finite, so deltas cannot normalize.
    """
    author = fmean(score(text) for text in author_texts)
    foreign = fmean(score(text) for text in foreign_texts)
    style_swap = fmean(score(text) for text in style_swapped)
    topic_match = fmean(score(text) for text in topic_matched)
    spread = author - foreign
    if not math.isfinite(spread) or spread == 0.0:
        raise ProbeError(f"author-vs-foreign spread is {spread}; cannot normalize topic-leakage deltas")
    return LeakageReport(
        author_baseline=author,
        foreign_baseline=foreign,
        style_swap_score=style_swap,
        topic_match_score=topic_match,
        style_swap_drop=(author - style_swap) / spread,
        topic_match_inflation=(topic_match - foreign) / spread,
    )
