from __future__ import annotations

import pytest

from athome.research.common import (
    ConfusionMatrix,
    Hasher,
    StratifiedSplitter,
    canonical_json,
    dataset_digest,
)


def test_canonical_json_sorts_keys_and_returns_bytes() -> None:
    assert canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_canonical_json_stringifies_non_json_values() -> None:
    assert canonical_json({"p": object.__name__}) == b'{"p":"object"}'


def test_hasher_is_an_associative_accumulator() -> None:
    assert Hasher().update(b"ab").hexdigest() == Hasher().update(b"a").update(b"b").hexdigest()


def test_hasher_digest_is_key_order_invariant() -> None:
    assert Hasher.digest({"a": 1, "b": 2}) == Hasher.digest({"b": 2, "a": 1})


def test_dataset_digest_is_order_invariant() -> None:
    items = [{"x": 1}, {"y": 2}, {"z": 3}]
    assert dataset_digest(items) == dataset_digest(list(reversed(items)))


def test_dataset_digest_changes_with_content() -> None:
    assert dataset_digest([{"x": 1}]) != dataset_digest([{"x": 2}])


def test_dataset_digest_is_16_hex_chars() -> None:
    digest = dataset_digest([{"x": 1}, {"x": 2}])
    assert len(digest) == 16
    assert all(char in "0123456789abcdef" for char in digest)


def test_stratified_split_is_disjoint_and_covers_the_deduped_pool() -> None:
    rows = [{"id": i, "label": i % 2} for i in range(40)]
    split = StratifiedSplitter().split(rows, by=["label"])
    keys = [{Hasher.digest(row) for row in group} for group in split]
    train, val, test = keys
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == {Hasher.digest(row) for row in rows}
    assert len(train) + len(val) + len(test) == 40


def test_stratified_split_dedupes_identical_rows() -> None:
    split = StratifiedSplitter().split([{"label": "a"}] * 5, by=["label"])
    assert len(split.train) + len(split.val) + len(split.test) == 1


def test_stratified_split_is_deterministic_under_the_seed() -> None:
    rows = [{"id": i, "label": i % 3} for i in range(30)]
    assert StratifiedSplitter().split(rows, by=["label"]) == StratifiedSplitter().split(rows, by=["label"])


def test_stratified_split_stratifies_each_class_proportionally() -> None:
    rows = [{"id": i, "label": i % 2} for i in range(40)]
    train_labels = {row["label"] for row in StratifiedSplitter().split(rows, by=["label"]).train}
    assert train_labels == {0, 1}


def test_stratified_splitter_rejects_ratios_that_miss_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        StratifiedSplitter(ratios=(0.5, 0.2, 0.2))


def test_confusion_matrix_counts_each_quadrant() -> None:
    matrix = ConfusionMatrix.from_pairs(
        [(True, True), (True, True), (True, False), (False, True), (False, False), (False, False), (False, False)]
    )
    assert (matrix.tp, matrix.fn, matrix.fp, matrix.tn) == (2, 1, 1, 3)
    assert matrix.total() == 7


def test_confusion_matrix_metric_values() -> None:
    matrix = ConfusionMatrix(tp=2, fp=1, fn=1, tn=3)
    assert matrix.accuracy() == pytest.approx(5 / 7)
    assert matrix.precision() == pytest.approx(2 / 3)
    assert matrix.recall() == pytest.approx(2 / 3)
    assert matrix.f1() == pytest.approx(2 / 3)
    assert matrix.as_dict() == {
        "accuracy": pytest.approx(5 / 7),
        "precision": pytest.approx(2 / 3),
        "recall": pytest.approx(2 / 3),
        "f1": pytest.approx(2 / 3),
    }


def test_confusion_matrix_empty_denominators_are_zero() -> None:
    empty = ConfusionMatrix(tp=0, fp=0, fn=0, tn=0)
    assert (empty.accuracy(), empty.precision(), empty.recall(), empty.f1()) == (0.0, 0.0, 0.0, 0.0)
