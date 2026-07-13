from __future__ import annotations

import pytest

from athome.research.probes import LeakageReport, ProbeError, TopicLeakageViolation, topic_leakage

AUTHOR = ["author on topic one", "author on topic two"]
FOREIGN = ["foreign off topic one", "foreign off topic two"]


def quality_score(text: str) -> float:
    """A topic-pure judge: scores by the 'author' style marker, indifferent to topic."""
    return 1.0 if text.startswith("author") else 0.0


def topic_score(text: str) -> float:
    """A topic-leaky judge: scores by whether the text mentions the author's pet topic."""
    return 1.0 if "pet-topic" in text else 0.0


def test_topic_pure_judge_shows_no_leakage() -> None:
    report = topic_leakage(
        quality_score,
        author_texts=AUTHOR,
        foreign_texts=FOREIGN,
        style_swapped=["author now off topic", "author still off topic"],
        topic_matched=["foreign wearing author topic", "foreign topic swap"],
    )
    assert report.style_swap_drop == pytest.approx(0.0)
    assert report.topic_match_inflation == pytest.approx(0.0)
    report.check()


def test_topic_leaky_judge_is_flagged() -> None:
    report = topic_leakage(
        topic_score,
        author_texts=["author pet-topic one", "author pet-topic two"],
        foreign_texts=["foreign other one", "foreign other two"],
        style_swapped=["author other one", "author other two"],
        topic_matched=["foreign pet-topic one", "foreign pet-topic two"],
    )
    assert report.style_swap_drop == pytest.approx(1.0)
    assert report.topic_match_inflation == pytest.approx(1.0)
    with pytest.raises(TopicLeakageViolation, match="topic leakage"):
        report.check()


def test_leakage_property_takes_the_worse_arm() -> None:
    report = LeakageReport(
        author_baseline=1.0,
        foreign_baseline=0.0,
        style_swap_score=0.9,
        topic_match_score=0.7,
        style_swap_drop=0.1,
        topic_match_inflation=0.7,
    )
    assert report.leakage == pytest.approx(0.7)
    with pytest.raises(TopicLeakageViolation):
        report.check(tolerance=0.5)
    report.check(tolerance=0.8)


def test_zero_spread_cannot_normalize() -> None:
    with pytest.raises(ProbeError, match="cannot normalize"):
        topic_leakage(
            lambda text: 0.5,
            author_texts=AUTHOR,
            foreign_texts=FOREIGN,
            style_swapped=AUTHOR,
            topic_matched=FOREIGN,
        )
