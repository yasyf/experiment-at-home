"""spawnllm-backed structured judges with position debiasing, vote caching, and health controls.

A :class:`Judge` wraps ``spawnllm.extract`` over a pydantic verdict model; the
heavy backend is imported lazily so the research core stays import-clean without
the ``llm`` extra. Pairwise judging is position-debiased by a seeded coin so slot
bias washes out, and every vote is cached by a sha256 key so re-runs and
overlapping batches never re-buy a vote. Two guards keep a judge honest: embedded
paraphrase/garbage control pairs that fail the batch when the judge flunks them
(donor: cc-steer-lab judge kit), and a cross-family refusal so a model never grades
its own family's output (donor: write-like-me cross-family rule).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, cast

import anyio
from loguru import logger
from pydantic import BaseModel

from athome.research.errors import ResearchError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from pathlib import Path

    from spawnllm import TModel

MAX_ATTEMPTS = 4
BACKOFF_BASE_S = 5.0
DIGEST_CHARS = 16
DEFAULT_CONCURRENCY = 8
MIN_PARAPHRASE_TIE = 0.5
MIN_GARBAGE_LOSS = 0.9


class JudgeError(ResearchError):
    """A judge call could not be completed after exhausting its retries."""


class CrossFamilyViolation(ResearchError):
    """A judge shares the generator's model family and cannot grade it impartially."""


class JudgeControlsViolation(ResearchError):
    """A judging batch's embedded control pairs failed: the judge is untrustworthy, so the cell is invalid."""


class Vote(StrEnum):
    WIN = "win"
    TIE = "tie"
    LOSS = "loss"


class Pairwise(BaseModel):
    """The A/B/tie verdict a position-debiased pairwise judge returns."""

    winner: Literal["A", "B", "tie"]


async def run_verdict[T: BaseModel](prompt: str, verdict_model: type[T], *, tier: TModel, timeout: int) -> T:
    from spawnllm import extract

    return await extract(prompt, verdict_model, model=tier, timeout=timeout)


async def with_backoff[T](call: Callable[[], Awaitable[T]], *, label: str, attempts: int = MAX_ATTEMPTS) -> T:
    """Retry a judge call on transient backend errors with exponential backoff, then raise :class:`JudgeError`."""
    from spawnllm import BackendCallError

    last: BackendCallError | None = None
    for attempt in range(attempts):
        try:
            return await call()
        except BackendCallError as error:
            last = error
            logger.warning("judge {} attempt {}/{} failed: {}", label, attempt + 1, attempts, error)
        if attempt + 1 < attempts:
            await anyio.sleep(BACKOFF_BASE_S * 2**attempt)
    raise JudgeError(f"judge call {label!r} failed after {attempts} attempts") from last


async def gather_bounded[T](
    tasks: Sequence[Callable[[], Awaitable[T]]], *, concurrency: int = DEFAULT_CONCURRENCY
) -> list[T]:
    """Run the task factories under a capacity limiter, preserving input order in the results."""
    results: list[T | None] = [None] * len(tasks)
    limiter = anyio.CapacityLimiter(concurrency)

    async def one(index: int) -> None:
        async with limiter:
            results[index] = await tasks[index]()

    async with anyio.create_task_group() as group:
        for index in range(len(tasks)):
            group.start_soon(one, index)
    return cast("list[T]", results)


def ensure_cross_family(judge_family: str, generator_family: str) -> None:
    """Refuse a same-family judge: a model must not grade its own family's output (self-preference bias).

    Raises:
        CrossFamilyViolation: the judge and generator share a model family.
    """
    if judge_family == generator_family:
        raise CrossFamilyViolation(
            f"judge family {judge_family!r} must differ from generator family {generator_family!r}: "
            "a model cannot impartially grade its own family's output"
        )


def coin(row_id: str, seed: int) -> bool:
    """A deterministic per-``(row_id, seed)`` coin: ``True`` places the candidate in slot A."""
    return int(hashlib.sha256(f"{row_id}|{seed}".encode()).hexdigest(), 16) % 2 == 0


def vote_of(winner: Literal["A", "B", "tie"], *, candidate_is_a: bool) -> Vote:
    """Map the judge's A/B/tie onto the candidate's win/tie/loss given the slot it took."""
    match winner:
        case "tie":
            return Vote.TIE
        case "A":
            return Vote.WIN if candidate_is_a else Vote.LOSS
        case "B":
            return Vote.LOSS if candidate_is_a else Vote.WIN


@dataclass(frozen=True, slots=True)
class VoteCache:
    """A disk-backed judge-vote cache keyed by a sha256 over ``(row, candidate, seed)``.

    Loaded once into memory on :meth:`open`; :meth:`get` serves from memory and
    :meth:`put` serializes the full snapshot under a lock, so overlapping batches
    never re-buy a vote and concurrent writes never tear the file.

    Example:
        >>> cache = VoteCache.open(Path("~/.athome/research/votes.json"))
    """

    path: Path
    _votes: dict[str, str]
    _lock: anyio.Lock = field(default_factory=anyio.Lock)

    @classmethod
    def open(cls, path: Path) -> VoteCache:
        return cls(path, json.loads(path.read_text()) if path.exists() else {})

    @staticmethod
    def key(row_id: str, candidate: str, seed: int) -> str:
        return f"{row_id}|{hashlib.sha256(candidate.encode()).hexdigest()[:DIGEST_CHARS]}|{seed}"

    def get(self, row_id: str, candidate: str, seed: int) -> Vote | None:
        stored = self._votes.get(self.key(row_id, candidate, seed))
        return Vote(stored) if stored is not None else None

    async def put(self, row_id: str, candidate: str, seed: int, vote: Vote) -> None:
        self._votes[self.key(row_id, candidate, seed)] = vote.value
        async with self._lock:
            await anyio.Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            await anyio.Path(self.path).write_text(json.dumps(self._votes, indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True, slots=True)
class Judge[T: BaseModel]:
    """A spawnllm-backed structured judge over a pydantic verdict model.

    Attributes:
        verdict_model: The pydantic model the judge's structured output validates against.
        family: The judge model's own family, enforced against the generator's by cross-family checks.
        tier: The abstract spawnllm model tier the judge runs at.
        timeout: Seconds to wait before the backend process is killed.

    Example:
        >>> await Judge(Pairwise, family="anthropic").verdict(prompt)
    """

    verdict_model: type[T]
    family: str
    tier: TModel = "large"
    timeout: int = 240

    async def verdict(self, prompt: str, *, label: str = "judge") -> T:
        """Run one structured judging call over ``prompt``, retrying transient backend errors."""
        return await with_backoff(
            lambda: run_verdict(prompt, self.verdict_model, tier=self.tier, timeout=self.timeout), label=label
        )


async def pairwise_vote(
    judge: Judge[Pairwise],
    *,
    row_id: str,
    candidate: str,
    reference: str,
    build_prompt: Callable[[str, str], str],
    seed: int,
    cache: VoteCache | None = None,
) -> Vote:
    """One position-debiased pairwise vote of ``candidate`` against ``reference``.

    The candidate's A/B slot is a deterministic coin on ``(row_id, seed)`` so
    position bias washes out across a subset. ``build_prompt(a, b)`` fills the
    domain prompt from the two slots. A cache hit returns without buying a vote.
    """
    if cache is not None and (hit := cache.get(row_id, candidate, seed)) is not None:
        return hit
    candidate_is_a = coin(row_id, seed)
    a, b = (candidate, reference) if candidate_is_a else (reference, candidate)
    verdict = await judge.verdict(build_prompt(a, b), label=f"pairwise:{row_id}")
    vote = vote_of(verdict.winner, candidate_is_a=candidate_is_a)
    if cache is not None:
        await cache.put(row_id, candidate, seed, vote)
    return vote


@dataclass(frozen=True, slots=True)
class ControlPair:
    """One embedded control candidate judged against its row's reference.

    A ``"paraphrase"`` restates the reference (a healthy judge answers tie); a
    ``"garbage"`` is unrelated text (a healthy judge picks the reference).
    """

    row_id: str
    kind: Literal["paraphrase", "garbage"]
    candidate: str
    reference: str


@dataclass(frozen=True, slots=True)
class ControlsReport:
    """The health of one judging batch's embedded control pairs.

    Attributes:
        n_paraphrase: Paraphrase controls judged.
        n_garbage: Garbage controls judged.
        paraphrase_tie_rate: Share of paraphrase controls the judge tied (should be high).
        garbage_loss_rate: Share of garbage controls the judge rejected (should be near one).
    """

    n_paraphrase: int
    n_garbage: int
    paraphrase_tie_rate: float
    garbage_loss_rate: float

    def check(
        self, *, min_paraphrase_tie: float = MIN_PARAPHRASE_TIE, min_garbage_loss: float = MIN_GARBAGE_LOSS
    ) -> None:
        """Fail the batch when the judge flunks its controls.

        Raises:
            JudgeControlsViolation: no controls ran, garbage beat the reference too
                often, or the paraphrase tie rate fell below the floor.
        """
        match self:
            case ControlsReport(n_paraphrase=0) | ControlsReport(n_garbage=0):
                raise JudgeControlsViolation("no control pairs ran; the batch cannot be validated")
            case ControlsReport(garbage_loss_rate=rate) if rate < min_garbage_loss:
                raise JudgeControlsViolation(
                    f"garbage beat or tied the reference too often (loss rate {rate:.3f} < {min_garbage_loss}) "
                    f"over {self.n_garbage} pairs"
                )
            case ControlsReport(paraphrase_tie_rate=rate) if rate < min_paraphrase_tie:
                raise JudgeControlsViolation(
                    f"paraphrase tie rate {rate:.3f} < {min_paraphrase_tie} over {self.n_paraphrase} pairs"
                )


async def run_controls(
    judge: Judge[Pairwise],
    pairs: Sequence[ControlPair],
    *,
    build_prompt: Callable[[str, str], str],
    seed: int,
    cache: VoteCache | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> ControlsReport:
    """Judge every embedded control pair and tally the paraphrase-tie and garbage-loss rates."""
    votes = await gather_bounded(
        [
            lambda pair=pair: pairwise_vote(
                judge,
                row_id=f"{pair.row_id}|control-{pair.kind}",
                candidate=pair.candidate,
                reference=pair.reference,
                build_prompt=build_prompt,
                seed=seed,
                cache=cache,
            )
            for pair in pairs
        ],
        concurrency=concurrency,
    )
    paraphrase = [vote for pair, vote in zip(pairs, votes, strict=True) if pair.kind == "paraphrase"]
    garbage = [vote for pair, vote in zip(pairs, votes, strict=True) if pair.kind == "garbage"]
    return ControlsReport(
        n_paraphrase=len(paraphrase),
        n_garbage=len(garbage),
        paraphrase_tie_rate=rate_of(paraphrase, Vote.TIE),
        garbage_loss_rate=rate_of(garbage, Vote.LOSS),
    )


def rate_of(votes: Sequence[Vote], expected: Vote) -> float:
    return sum(1 for vote in votes if vote is expected) / len(votes) if votes else 0.0
