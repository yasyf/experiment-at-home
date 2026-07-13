"""Shared research-harness primitives: canonical JSON, digests, splits, metrics."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, NewType

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

Row = Mapping[str, object]
DatasetDigest = NewType("DatasetDigest", str)


def canonical_json(payload: object) -> bytes:
    """Deterministic JSON bytes: sorted keys, no whitespace, non-JSON values stringified."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()


@dataclass(slots=True)
class Hasher:
    """Incremental blake2b digest accumulator over byte chunks.

    Example:
        >>> Hasher().update(b"a").update(b"b").hexdigest()
    """

    _acc: hashlib.blake2b = field(init=False, default_factory=hashlib.blake2b)

    def update(self, data: bytes) -> Hasher:
        self._acc.update(data)
        return self

    def hexdigest(self) -> str:
        return self._acc.hexdigest()

    @classmethod
    def digest(cls, payload: object) -> str:
        return cls().update(canonical_json(payload)).hexdigest()


def dataset_digest(items: Iterable[object]) -> DatasetDigest:
    """Order-invariant content digest over items (sorted per-item blake2b hashes)."""
    combined = Hasher()
    for digest in sorted(Hasher.digest(item) for item in items):
        combined.update(digest.encode())
    return DatasetDigest(combined.hexdigest()[:16])


class Split(NamedTuple):
    train: tuple[Row, ...]
    val: tuple[Row, ...]
    test: tuple[Row, ...]


@dataclass(frozen=True, slots=True)
class StratifiedSplitter:
    """Deterministic stratified train/val/test split with dedupe and disjointness asserts.

    Rows are deduped by content digest, bucketed by the ``by`` columns, and each
    bucket is shuffled under a fixed seed before being sliced by ``ratios``.

    Example:
        >>> StratifiedSplitter().split(rows, by=["label"])
    """

    seed: int = 1729
    ratios: tuple[float, float, float] = (0.75, 0.10, 0.15)

    def __post_init__(self) -> None:
        if abs(sum(self.ratios) - 1.0) > 1e-9:
            raise ValueError(f"ratios must sum to 1, got {self.ratios}")

    def split(self, pool: Iterable[Row], *, by: Sequence[str]) -> Split:
        rng = random.Random(self.seed)
        buckets = stratify(dedupe(pool), by=by)
        slices = [slice_bucket(buckets[key], rng=rng, ratios=self.ratios) for key in sorted(buckets)]
        split = Split(
            tuple(row for s in slices for row in s.train),
            tuple(row for s in slices for row in s.val),
            tuple(row for s in slices for row in s.test),
        )
        assert_disjoint(split)
        return split


def dedupe(pool: Iterable[Row]) -> tuple[Row, ...]:
    return tuple({Hasher.digest(row): row for row in pool}.values())


def stratify(rows: Iterable[Row], *, by: Sequence[str]) -> dict[tuple[str, ...], list[Row]]:
    buckets: dict[tuple[str, ...], list[Row]] = defaultdict(list)
    for row in rows:
        buckets[tuple(str(row.get(col, "")) for col in by)].append(row)
    return buckets


def slice_bucket(rows: Sequence[Row], *, rng: random.Random, ratios: tuple[float, float, float]) -> Split:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_train = round(len(shuffled) * ratios[0])
    n_val = round(len(shuffled) * ratios[1])
    return Split(
        tuple(shuffled[:n_train]),
        tuple(shuffled[n_train : n_train + n_val]),
        tuple(shuffled[n_train + n_val :]),
    )


def assert_disjoint(split: Split) -> None:
    train, val, test = ({Hasher.digest(row) for row in group} for group in split)
    assert not train & test, "train/test contamination"
    assert not val & test, "val/test contamination"
    assert not train & val, "train/val contamination"


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """Binary confusion matrix over ``(truth, prediction)`` pairs.

    Example:
        >>> ConfusionMatrix.from_pairs([(True, True), (False, True)]).precision()
    """

    tp: int
    fp: int
    fn: int
    tn: int

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[bool, bool]]) -> ConfusionMatrix:
        counts = Counter(pairs)
        return cls(
            tp=counts[True, True],
            fp=counts[False, True],
            fn=counts[True, False],
            tn=counts[False, False],
        )

    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total() if self.total() else 0.0

    def precision(self) -> float:
        return self.tp / predicted if (predicted := self.tp + self.fp) else 0.0

    def recall(self) -> float:
        return self.tp / actual if (actual := self.tp + self.fn) else 0.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict[str, float]:
        return {"accuracy": self.accuracy(), "precision": self.precision(), "recall": self.recall(), "f1": self.f1()}
