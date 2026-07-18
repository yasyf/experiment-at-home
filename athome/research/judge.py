"""spawnllm-backed structured judges with position debiasing, vote caching, and health controls.

A :class:`Judge` wraps ``spawnllm.extract`` over a pydantic verdict model. It holds
a concrete :class:`spawnllm.backends.base.LlmBackend`, so its family and model are
**derived** from that binding, never declared — a judge on a provider that maps to no
single family is unrepresentable. The heavy backend is imported lazily so the research
core stays import-clean without the ``llm`` extra. Pairwise judging is position-debiased
by a seeded coin so slot bias washes out, and every vote is cached by a canonical-JSON
sha256 key so re-runs and overlapping batches never re-buy a vote. :func:`run_verdict`
is the single buy boundary: every backend call passes through it after a cross-family
check and a proof-carrying spend grant, so no caller can spend without clearing the
golden gate and the embedded health controls (paraphrase/garbage control pairs, donor:
cc-steer-lab judge kit; cross-family refusal, donor: write-like-me cross-family rule).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, NewType
from uuid import uuid4

import anyio
from loguru import logger
from pydantic import BaseModel

from athome.concurrency import gather_bounded
from athome.research.common import canonical_json
from athome.research.errors import ResearchError
from athome.research.golden import GoldenProof, VerifiedManifest

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from pathlib import Path

    from spawnllm import TModel
    from spawnllm.backends.base import LlmBackend

MAX_ATTEMPTS = 4
BACKOFF_BASE_S = 5.0
DEFAULT_CONCURRENCY = 8
MIN_PARAPHRASE_TIE = 0.5
MIN_GARBAGE_LOSS = 0.9
VOTE_KEY_SCHEMA = 2
FAMILY_ALIASES: dict[str, str] = {
    "claude": "anthropic",
    "anthropic": "anthropic",
    "opus": "anthropic",
    "sonnet": "anthropic",
    "haiku": "anthropic",
    "fable": "anthropic",
    "gpt": "openai",
    "openai": "openai",
    "chatgpt": "openai",
    "codex": "openai",
    "o1": "openai",
    "o3": "openai",
    "gemini": "google",
    "google": "google",
    "bard": "google",
    "palm": "google",
    "llama": "meta",
    "meta": "meta",
    "mistral": "mistral",
    "mixtral": "mistral",
    "qwen": "qwen",
    "qwq": "qwen",
    "qvq": "qwen",
    "alibaba": "qwen",
}
PROVIDER_FAMILIES: dict[str, str] = {
    "claude": "anthropic",
    "codex": "openai",
    "gemini": "google",
    "antigravity": "google",
}

HealthEpoch = NewType("HealthEpoch", str)


class JudgeError(ResearchError):
    """A judge call could not be completed after exhausting its retries."""


class CrossFamilyViolation(ResearchError):
    """A judge shares the generator's model family and cannot grade it impartially."""


class JudgeControlsViolation(ResearchError):
    """A judging batch's embedded control pairs failed: the judge is untrustworthy, so the cell is invalid."""


class UnknownFamilyError(ResearchError):
    """A model/provider string maps to no known family; cross-family safety cannot be established."""


class Vote(StrEnum):
    WIN = "win"
    TIE = "tie"
    LOSS = "loss"


class Pairwise(BaseModel):
    """The A/B/tie verdict a position-debiased pairwise judge returns."""

    winner: Literal["A", "B", "tie"]


def family_of(value: str) -> str:
    """Map a model or family string to its canonical family, failing closed on the unknown.

    Tokenizes on non-alphabetic runs so model names resolve (``"gemini-2.5-pro"`` ->
    google, ``"claude-3-7-sonnet"`` -> anthropic, ``"azure-openai"`` -> openai). Zero
    matched tokens, or tokens matching two different families, raise.

    Raises:
        UnknownFamilyError: no token resolves, or tokens span more than one family.
    """
    if len(families := {FAMILY_ALIASES[t] for t in re.split(r"[^a-z]+", value.lower()) if t in FAMILY_ALIASES}) == 1:
        return families.pop()
    raise UnknownFamilyError(f"cannot map {value!r} to one model family (matched {sorted(families)!r})")


def ensure_cross_family(judge_family: str, generator_family: str) -> None:
    """Refuse a same-family judge: a model must not grade its own family's output (self-preference bias).

    ``judge_family`` is already canonical (derived via :data:`PROVIDER_FAMILIES`); the
    generator's is resolved through :func:`family_of`, failing closed on an unknown, so
    ``"claude-3-7-sonnet"`` cannot slip past a judge whose family is ``"anthropic"``.

    Raises:
        CrossFamilyViolation: the judge and generator resolve to the same model family.
        UnknownFamilyError: the generator family maps to no known family.
    """
    if judge_family == family_of(generator_family):
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


def rate_of(votes: Sequence[Vote], expected: Vote) -> float:
    return sum(1 for vote in votes if vote is expected) / len(votes) if votes else 0.0


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


@dataclass(frozen=True, slots=True)
class JudgeIdentity:
    """The concrete identity a vote is keyed on: requested provider, model, and verdict schema.

    The model is the *requested* concrete provider model (spawnllm does not echo the
    provider-served model id), so the identity binds what was asked for, not what a
    provider silently substituted.

    Attributes:
        provider: The backend's spawnllm provider name (e.g. ``"codex"``).
        model: The concrete model the judge's tier requests on that backend.
        verdict_schema_sha256: A sha256 over the verdict model's canonical JSON schema.
    """

    provider: str
    model: str
    verdict_schema_sha256: str


@dataclass(frozen=True, slots=True)
class Judge[T: BaseModel]:
    """A spawnllm-backed structured judge bound to a concrete backend.

    The judge holds an :class:`~spawnllm.backends.base.LlmBackend`; its family and model
    are derived from that binding, never declared. Construction fails closed when the
    backend's provider maps to no single family, so a same-family check can never be blind.

    Attributes:
        verdict_model: The pydantic model the judge's structured output validates against.
        backend: The concrete spawnllm backend every call runs on (kills auto-selection).
        tier: The abstract spawnllm model tier the judge runs at.
        timeout: Seconds to wait before the backend process is killed.

    Example:
        >>> await Judge.bind(Pairwise, backend="codex").verdict(prompt, generator_family="anthropic", grant=clearance)
    """

    verdict_model: type[T]
    backend: LlmBackend
    tier: TModel = "large"
    timeout: int = 240

    def __post_init__(self) -> None:
        if (provider := self.backend.provider) not in PROVIDER_FAMILIES:
            raise UnknownFamilyError(
                f"backend provider {provider!r} maps to no model family; cross-family checks would be blind"
            )

    @classmethod
    def bind(
        cls, verdict_model: type[T], *, backend: LlmBackend | str, tier: TModel = "large", timeout: int = 240
    ) -> Judge[T]:
        """Bind a judge to a backend instance, or to a registry backend by name.

        Args:
            verdict_model: The pydantic model the judge validates against.
            backend: An ``LlmBackend`` instance, or a ``BACKENDS_BY_NAME`` key.
            tier: The abstract model tier the judge runs at.
            timeout: Seconds to wait before the backend process is killed.
        """
        from spawnllm.backends.registry import BACKENDS_BY_NAME

        return cls(
            verdict_model=verdict_model,
            backend=BACKENDS_BY_NAME[backend] if isinstance(backend, str) else backend,
            tier=tier,
            timeout=timeout,
        )

    @property
    def family(self) -> str:
        """The judge's canonical model family, derived from its backend's provider."""
        return PROVIDER_FAMILIES[self.backend.provider]

    @property
    def model_name(self) -> str:
        """The concrete model the judge's tier resolves to on its backend."""
        return self.backend.models[self.tier]

    @property
    def identity(self) -> JudgeIdentity:
        """The concrete, vote-cache-keying identity of this judge."""
        return JudgeIdentity(
            provider=self.backend.provider,
            model=self.model_name,
            verdict_schema_sha256=hashlib.sha256(canonical_json(self.verdict_model.model_json_schema())).hexdigest(),
        )

    async def verdict(self, prompt: str, *, generator_family: str, grant: SpendGrant, label: str = "judge") -> T:
        """Run one structured judging call through the single buy boundary.

        Raises:
            CrossFamilyViolation: the judge shares the generator's model family.
            UnknownFamilyError: the generator family maps to no known family.
            GoldenGateViolation: the grant's golden gate is red.
            JudgeControlsViolation: the grant's health controls are red.
            JudgeError: the backend call failed after its retries.
        """
        return await run_verdict(self, prompt, generator_family=generator_family, grant=grant, label=label)


@dataclass(frozen=True, slots=True)
class PanelGrant:
    """Bootstrap lane: buys only panel votes over a verified packet's rows.

    Constructible only from a :class:`~athome.research.golden.VerifiedManifest`, so panel
    spend presupposes packet integrity; its economic teeth are downstream — panel votes
    cannot unlock candidate spend without ``prove_gate`` agreeing with the human labels.

    Attributes:
        manifest: The verified packet whose rows the panel votes.
    """

    manifest: VerifiedManifest


@dataclass(frozen=True, slots=True)
class ControlsGrant:
    """Controls lane: buys only this epoch's control-pair votes, behind a green golden gate.

    Attributes:
        golden: The passed golden gate this epoch's controls run behind.
        epoch: The health epoch these control votes belong to.
    """

    golden: GoldenProof
    epoch: HealthEpoch

    def check(self) -> None:
        """Re-verify the golden gate before control spend.

        Raises:
            GoldenGateViolation: the golden gate is red.
        """
        self.golden.check()


@dataclass(frozen=True, slots=True)
class SpendClearance:
    """Both spend preconditions bound together, re-verified at every buy.

    Attributes:
        golden: The passed golden gate.
        controls: The health-controls report the candidate votes clear.
        min_paraphrase_tie: The paraphrase-tie floor the controls must clear.
        min_garbage_loss: The garbage-loss floor the controls must clear.
    """

    golden: GoldenProof
    controls: ControlsReport
    min_paraphrase_tie: float = MIN_PARAPHRASE_TIE
    min_garbage_loss: float = MIN_GARBAGE_LOSS

    def check(self) -> None:
        """Re-verify both spend preconditions at the trust boundary.

        Raises:
            GoldenGateViolation: the golden gate is red.
            JudgeControlsViolation: the health controls fell below a floor.
        """
        self.golden.check()
        self.controls.check(min_paraphrase_tie=self.min_paraphrase_tie, min_garbage_loss=self.min_garbage_loss)


type SpendGrant = PanelGrant | ControlsGrant | SpendClearance


async def run_verdict[T: BaseModel](
    judge: Judge[T], prompt: str, *, generator_family: str, grant: SpendGrant, label: str
) -> T:
    """The single buy boundary: every backend call passes here, after every guard.

    Refuses a same-family generator, re-checks the grant's spend preconditions, then runs
    the bound backend directly — ``extract(..., backend=judge.backend)`` never reaches
    spawnllm's auto-select branch. Retries wrap only the backend call, so the guards run once.

    Raises:
        CrossFamilyViolation: the judge shares the generator's model family.
        UnknownFamilyError: the generator family maps to no known family.
        GoldenGateViolation: the grant's golden gate is red.
        JudgeControlsViolation: the grant's health controls are red.
        JudgeError: the backend call failed after its retries.
    """
    from spawnllm import extract

    ensure_cross_family(judge.family, generator_family)
    match grant:
        case SpendClearance() | ControlsGrant():
            grant.check()
        case PanelGrant():
            pass
    return await with_backoff(
        lambda: extract(prompt, judge.verdict_model, backend=judge.backend, model=judge.tier, timeout=judge.timeout),
        label=label,
    )


@dataclass(frozen=True, slots=True)
class VoteContext:
    """The batch-level evaluation context every cached vote is bound to.

    Pairs the judging prompt's version with the dataset/config comparability digest, so a
    cached vote is reused only while both still hold: bump the prompt or change the dataset
    and the vote misses the cache and is re-bought, rather than a stale verdict being replayed.

    Attributes:
        prompt_version: The judging prompt's version tag; a bump invalidates cached votes.
        digest: The dataset/config comparability digest the votes were cast under.

    Example:
        >>> VoteContext(prompt_version="pairwise-v3", digest=comparability.dataset_digest)
    """

    prompt_version: str
    digest: str


@dataclass(frozen=True, slots=True)
class VoteCache:
    """A disk-backed judge-vote cache keyed by the full evaluation context of a vote.

    The key is a canonical-JSON, schema-versioned sha256 (see :data:`VOTE_KEY_SCHEMA`)
    binding the judge identity, prompt version, and dataset/config digest to the row,
    candidate, reference, and slot seed, so a cached vote is served only for an identical
    judging context: a WIN against one reference is never replayed against a new reference,
    and votes from a healthy judge never mask a now-poisoned one. JSON escaping makes the
    key delimiter-injection-proof, and a ``VOTE_KEY_SCHEMA`` bump invalidates a whole
    generation at once (stale v1 entries simply never match again).

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
    def key(
        identity: JudgeIdentity, context: VoteContext, *, row_id: str, candidate: str, reference: str, seed: int
    ) -> str:
        return hashlib.sha256(
            canonical_json(
                {
                    "v": VOTE_KEY_SCHEMA,
                    "judge": {
                        "provider": identity.provider,
                        "model": identity.model,
                        "verdict_schema": identity.verdict_schema_sha256,
                    },
                    "prompt_version": context.prompt_version,
                    "digest": context.digest,
                    "row_id": row_id,
                    "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
                    "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
                    "seed": seed,
                }
            )
        ).hexdigest()

    def get(
        self, identity: JudgeIdentity, context: VoteContext, *, row_id: str, candidate: str, reference: str, seed: int
    ) -> Vote | None:
        stored = self._votes.get(
            self.key(identity, context, row_id=row_id, candidate=candidate, reference=reference, seed=seed)
        )
        return Vote(stored) if stored is not None else None

    async def put(
        self,
        identity: JudgeIdentity,
        context: VoteContext,
        *,
        row_id: str,
        candidate: str,
        reference: str,
        seed: int,
        vote: Vote,
    ) -> None:
        self._votes[self.key(identity, context, row_id=row_id, candidate=candidate, reference=reference, seed=seed)] = (
            vote.value
        )
        async with self._lock:
            await anyio.Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            await anyio.Path(self.path).write_text(json.dumps(self._votes, indent=2, sort_keys=True) + "\n")


async def pairwise_vote(
    judge: Judge[Pairwise],
    *,
    generator_family: str,
    grant: SpendGrant,
    context: VoteContext,
    row_id: str,
    candidate: str,
    reference: str,
    build_prompt: Callable[[str, str], str],
    seed: int,
    cache: VoteCache | None = None,
) -> Vote:
    """One position-debiased pairwise vote of ``candidate`` against ``reference``.

    Refuses a same-family generator before the cache lookup — failing closed on an unknown
    family — so neither a cache hit nor a buy can return a same-family verdict. The
    candidate's A/B slot is a deterministic coin on ``(row_id, seed)`` so position bias
    washes out across a subset. ``build_prompt(a, b)`` fills the domain prompt from the two
    slots. A cache hit returns without buying a vote, but only when the judge identity,
    prompt version, dataset/config digest, reference, candidate, and seed all match; a miss
    buys through :meth:`Judge.verdict`, the single buy boundary.

    Raises:
        CrossFamilyViolation: the judge shares the generator's model family.
        UnknownFamilyError: the generator family maps to no known family.
    """
    ensure_cross_family(judge.family, generator_family)
    if cache is not None:
        hit = cache.get(judge.identity, context, row_id=row_id, candidate=candidate, reference=reference, seed=seed)
        if hit is not None:
            return hit
    candidate_is_a = coin(row_id, seed)
    a, b = (candidate, reference) if candidate_is_a else (reference, candidate)
    verdict = await judge.verdict(
        build_prompt(a, b), generator_family=generator_family, grant=grant, label=f"pairwise:{row_id}"
    )
    vote = vote_of(verdict.winner, candidate_is_a=candidate_is_a)
    if cache is not None:
        await cache.put(
            judge.identity, context, row_id=row_id, candidate=candidate, reference=reference, seed=seed, vote=vote
        )
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
class JudgeRow:
    """One real candidate to judge against its reference — the spend the health controls gate.

    Attributes:
        row_id: The stable row identifier the position-debias coin and the vote cache key on.
        candidate: The candidate output under evaluation.
        reference: The reference the candidate is judged against.
    """

    row_id: str
    candidate: str
    reference: str


@dataclass(frozen=True, slots=True)
class ControlsReport:
    """The health of one judging batch's embedded control pairs.

    Attributes:
        epoch: The health epoch these controls were bought in; a fresh uuid per
            :func:`run_controls` call, so cross-batch reuse of a report is auditable.
        n_paraphrase: Paraphrase controls judged.
        n_garbage: Garbage controls judged.
        paraphrase_tie_rate: Share of paraphrase controls the judge tied (should be high).
        garbage_loss_rate: Share of garbage controls the judge rejected (should be near one).
    """

    epoch: HealthEpoch
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
    generator_family: str,
    golden: GoldenProof,
    context: VoteContext,
    build_prompt: Callable[[str, str], str],
    seed: int,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> ControlsReport:
    """Judge every embedded control pair in a fresh health epoch and tally the control rates.

    Controls are never cached: each call mints a new :class:`HealthEpoch` and buys every
    control vote afresh with ``cache=None``, so a warm cache can never mask a now-poisoned
    judge. The composite control ``row_id`` (``f"{pair.row_id}|control-{pair.kind}"``) feeds
    only the debias coin. Fails fast on a red golden gate before minting the epoch, so the
    gate surfaces bare rather than wrapped in the control fan-out's task group.

    Raises:
        CrossFamilyViolation: the judge shares the generator's model family.
        GoldenGateViolation: the golden gate is red.
    """
    golden.check()
    grant = ControlsGrant(golden=golden, epoch=(epoch := HealthEpoch(uuid4().hex)))
    votes = await gather_bounded(
        [
            lambda pair=pair: pairwise_vote(
                judge,
                generator_family=generator_family,
                grant=grant,
                context=context,
                row_id=f"{pair.row_id}|control-{pair.kind}",
                candidate=pair.candidate,
                reference=pair.reference,
                build_prompt=build_prompt,
                seed=seed,
            )
            for pair in pairs
        ],
        concurrency=concurrency,
    )
    paraphrase = [vote for pair, vote in zip(pairs, votes, strict=True) if pair.kind == "paraphrase"]
    garbage = [vote for pair, vote in zip(pairs, votes, strict=True) if pair.kind == "garbage"]
    return ControlsReport(
        epoch=epoch,
        n_paraphrase=len(paraphrase),
        n_garbage=len(garbage),
        paraphrase_tie_rate=rate_of(paraphrase, Vote.TIE),
        garbage_loss_rate=rate_of(garbage, Vote.LOSS),
    )


async def judge_candidates(
    judge: Judge[Pairwise],
    rows: Sequence[JudgeRow],
    *,
    generator_family: str,
    controls: Sequence[ControlPair],
    golden: GoldenProof,
    context: VoteContext,
    build_prompt: Callable[[str, str], str],
    seed: int,
    cache: VoteCache | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[Vote]:
    """Cast a batch of pairwise votes only after cross-family, golden gate, and controls all pass.

    The enforced spend path: refuse a same-family judge, re-run the embedded controls in a
    fresh epoch behind the golden gate, bind both into a :class:`SpendClearance`, and check
    it before the candidate fan-out. There is no report for the caller to forget — a red
    golden gate or poisoned judge raises before any candidate vote, and every candidate buy
    re-checks the clearance at the boundary.

    Raises:
        CrossFamilyViolation: the judge shares the generator's model family.
        GoldenGateViolation: the golden gate is red.
        JudgeControlsViolation: the judge failed its embedded health controls.
    """
    ensure_cross_family(judge.family, generator_family)
    clearance = SpendClearance(
        golden=golden,
        controls=await run_controls(
            judge,
            controls,
            generator_family=generator_family,
            golden=golden,
            context=context,
            build_prompt=build_prompt,
            seed=seed,
            concurrency=concurrency,
        ),
    )
    clearance.check()
    return await gather_bounded(
        [
            lambda row=row: pairwise_vote(
                judge,
                generator_family=generator_family,
                grant=clearance,
                context=context,
                row_id=row.row_id,
                candidate=row.candidate,
                reference=row.reference,
                build_prompt=build_prompt,
                seed=seed,
                cache=cache,
            )
            for row in rows
        ],
        concurrency=concurrency,
    )
