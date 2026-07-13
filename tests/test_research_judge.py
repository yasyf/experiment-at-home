from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import pytest

from athome.research import judge as judge_mod
from athome.research.judge import (
    ControlPair,
    ControlsReport,
    CrossFamilyViolation,
    Judge,
    JudgeControlsViolation,
    JudgeError,
    Pairwise,
    Vote,
    VoteCache,
    coin,
    ensure_cross_family,
    gather_bounded,
    pairwise_vote,
    run_controls,
    vote_of,
    with_backoff,
)

# spawnllm is the `llm` extra; skip this module cleanly on the dev-only free-threaded CI job.
spawnllm = pytest.importorskip("spawnllm")
BackendCallError = spawnllm.BackendCallError

if TYPE_CHECKING:
    from pathlib import Path


def build(a: str, b: str) -> str:
    return f"A={a}|B={b}"


def slots(prompt: str) -> tuple[str, str]:
    a, _, b = prompt.removeprefix("A=").partition("|B=")
    return a, b


@pytest.mark.parametrize(
    "winner, candidate_is_a, expected",
    [
        pytest.param("tie", True, Vote.TIE, id="tie-as-a"),
        pytest.param("tie", False, Vote.TIE, id="tie-as-b"),
        pytest.param("A", True, Vote.WIN, id="a-wins-candidate-is-a"),
        pytest.param("A", False, Vote.LOSS, id="a-wins-candidate-is-b"),
        pytest.param("B", True, Vote.LOSS, id="b-wins-candidate-is-a"),
        pytest.param("B", False, Vote.WIN, id="b-wins-candidate-is-b"),
    ],
)
def test_vote_of_maps_slot_to_candidate_outcome(winner: str, candidate_is_a: bool, expected: Vote) -> None:
    assert vote_of(winner, candidate_is_a=candidate_is_a) is expected


def test_coin_is_deterministic_and_spans_both_slots() -> None:
    assert coin("row", 3) == coin("row", 3)
    outcomes = {coin("row", seed) for seed in range(16)}
    assert outcomes == {True, False}


def test_ensure_cross_family_refuses_same_family() -> None:
    with pytest.raises(CrossFamilyViolation, match="own family"):
        ensure_cross_family("anthropic", "anthropic")


def test_ensure_cross_family_allows_a_different_family() -> None:
    ensure_cross_family("anthropic", "openai")


async def test_vote_cache_roundtrips_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "votes.json"
    cache = VoteCache.open(path)
    assert cache.get("r", "cand", 1) is None
    await cache.put("r", "cand", 1, Vote.WIN)
    assert cache.get("r", "cand", 1) is Vote.WIN
    assert VoteCache.open(path).get("r", "cand", 1) is Vote.WIN
    assert VoteCache.open(path).get("r", "cand", 2) is None


async def test_pairwise_vote_hits_cache_and_never_rebuys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake(prompt: str, verdict_model: type[Pairwise], *, tier: str, timeout: int) -> Pairwise:
        calls.append(prompt)
        return Pairwise(winner="A")

    monkeypatch.setattr(judge_mod, "run_verdict", fake)
    cache = VoteCache.open(tmp_path / "votes.json")
    j: Judge[Pairwise] = Judge(Pairwise, family="x")
    first = await pairwise_vote(
        j, row_id="r", candidate="cand", reference="ref", build_prompt=build, seed=1, cache=cache
    )
    second = await pairwise_vote(
        j, row_id="r", candidate="cand", reference="ref", build_prompt=build, seed=1, cache=cache
    )
    assert first is second
    assert len(calls) == 1


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
async def test_pairwise_is_position_debiased(seed: int, monkeypatch: pytest.MonkeyPatch) -> None:
    async def perfect(prompt: str, verdict_model: type[Pairwise], *, tier: str, timeout: int) -> Pairwise:
        a, _ = slots(prompt)
        return Pairwise(winner="A" if a == "cand" else "B")

    monkeypatch.setattr(judge_mod, "run_verdict", perfect)
    j: Judge[Pairwise] = Judge(Pairwise, family="x")
    vote = await pairwise_vote(j, row_id="row", candidate="cand", reference="ref", build_prompt=build, seed=seed)
    assert vote is Vote.WIN


def make_control_pairs() -> list[ControlPair]:
    return [
        ControlPair(row_id="r0", kind="paraphrase", candidate="PARA:r0", reference="REF:r0"),
        ControlPair(row_id="r0", kind="garbage", candidate="GARB:r0", reference="REF:r0"),
        ControlPair(row_id="r1", kind="paraphrase", candidate="PARA:r1", reference="REF:r1"),
        ControlPair(row_id="r1", kind="garbage", candidate="GARB:r1", reference="REF:r1"),
    ]


async def healthy_judge(prompt: str, verdict_model: type[Pairwise], *, tier: str, timeout: int) -> Pairwise:
    a, b = slots(prompt)
    reference_slot, candidate = ("A", b) if a.startswith("REF") else ("B", a)
    return Pairwise(winner="tie") if candidate.startswith("PARA") else Pairwise(winner=reference_slot)


async def poisoned_judge(prompt: str, verdict_model: type[Pairwise], *, tier: str, timeout: int) -> Pairwise:
    a, _ = slots(prompt)
    return Pairwise(winner="A" if not a.startswith("REF") else "B")


async def test_run_controls_passes_a_healthy_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(judge_mod, "run_verdict", healthy_judge)
    report = await run_controls(Judge(Pairwise, family="x"), make_control_pairs(), build_prompt=build, seed=7)
    assert report.n_paraphrase == report.n_garbage == 2
    assert report.paraphrase_tie_rate == 1.0
    assert report.garbage_loss_rate == 1.0
    report.check()


async def test_run_controls_fails_a_poisoned_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(judge_mod, "run_verdict", poisoned_judge)
    report = await run_controls(Judge(Pairwise, family="x"), make_control_pairs(), build_prompt=build, seed=7)
    assert report.garbage_loss_rate == 0.0
    with pytest.raises(JudgeControlsViolation, match="garbage"):
        report.check()


def test_controls_check_requires_both_control_kinds() -> None:
    with pytest.raises(JudgeControlsViolation, match="no control pairs"):
        ControlsReport(n_paraphrase=0, n_garbage=3, paraphrase_tie_rate=0.0, garbage_loss_rate=1.0).check()


def test_controls_check_flags_a_low_paraphrase_tie_rate() -> None:
    with pytest.raises(JudgeControlsViolation, match="paraphrase tie rate"):
        ControlsReport(n_paraphrase=10, n_garbage=10, paraphrase_tie_rate=0.2, garbage_loss_rate=1.0).check()


async def test_with_backoff_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(judge_mod, "BACKOFF_BASE_S", 0.0)
    attempts = {"n": 0}

    async def call() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise BackendCallError("transient")
        return "ok"

    assert await with_backoff(call, label="t") == "ok"
    assert attempts["n"] == 3


async def test_with_backoff_exhausts_and_raises_judge_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(judge_mod, "BACKOFF_BASE_S", 0.0)

    async def call() -> str:
        raise BackendCallError("always")

    with pytest.raises(JudgeError, match="after 2 attempts"):
        await with_backoff(call, label="t", attempts=2)


async def test_gather_bounded_preserves_input_order() -> None:
    async def make(index: int) -> int:
        await anyio.sleep(0)
        return index

    assert await gather_bounded([lambda i=i: make(i) for i in range(10)], concurrency=3) == list(range(10))


@pytest.mark.live
async def test_live_pairwise_judge_prefers_the_relevant_steer() -> None:
    def prompt(a: str, b: str) -> str:
        return (
            "Two candidate steering messages for a coding agent whose test suite is failing. "
            "The user wants the failing test fixed. Which candidate better matches that intent? "
            'Respond {"winner":"A"}, {"winner":"B"}, or {"winner":"tie"}.\n\n'
            f"=== CANDIDATE A ===\n{a}\n\n=== CANDIDATE B ===\n{b}"
        )

    vote = await pairwise_vote(
        Judge(Pairwise, family="anthropic"),
        row_id="live",
        candidate="Fix the failing test in test_math.py before adding features.",
        reference="Delete the whole repository and start over.",
        build_prompt=prompt,
        seed=1,
    )
    assert vote is Vote.WIN
