from __future__ import annotations

import hashlib
import inspect
from typing import TYPE_CHECKING

import anyio
import pytest

from athome.research import judge as judge_mod
from athome.research.common import canonical_json
from athome.research.golden import (
    AgreementReport,
    GoldenGate,
    GoldenGateViolation,
    GoldenProof,
    VerifiedManifest,
    prove_gate,
    verify_packet,
)
from athome.research.judge import (
    ControlPair,
    ControlsReport,
    CrossFamilyViolation,
    HealthEpoch,
    Judge,
    JudgeControlsViolation,
    JudgeError,
    JudgeIdentity,
    JudgeRow,
    Pairwise,
    PanelGrant,
    SpendClearance,
    UnknownFamilyError,
    Vote,
    VoteCache,
    VoteContext,
    coin,
    ensure_cross_family,
    family_of,
    gather_bounded,
    judge_candidates,
    pairwise_vote,
    run_controls,
    vote_of,
    with_backoff,
)

# spawnllm is the `llm` extra; skip this module cleanly on the dev-only free-threaded CI job.
spawnllm = pytest.importorskip("spawnllm")
BackendCallError = spawnllm.BackendCallError
LlmBackend = spawnllm.LlmBackend
Output = spawnllm.Output
Response = spawnllm.Response
Result = spawnllm.Result

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from spawnllm import RunSpec

CTX = VoteContext(prompt_version="v1", digest="d1")


class StubBackend(LlmBackend):
    """A deterministic in-process backend: records every RunSpec and returns a scripted Pairwise."""

    def __init__(
        self,
        *,
        provider: str = "codex",
        models: dict[str, str] | None = None,
        decide: Callable[[str], str] | None = None,
    ) -> None:
        self.provider = provider  # type: ignore[misc]
        self.models = models if models is not None else {"large": "gpt-5.2"}  # type: ignore[misc]
        self.calls: list[RunSpec] = []
        self._decide = decide or (lambda _prompt: "A")

    async def aexecute(self, spec: RunSpec) -> Response:
        self.calls.append(spec)
        parsed = Pairwise(winner=self._decide(spec.prompt))
        return Response(spec=spec, output=Output(raw=""), result=Result(raw="", parsed=parsed))

    def execute(self, spec: RunSpec) -> Response:
        raise NotImplementedError

    def env(self, spec: RunSpec) -> dict[str, str]:
        return {}

    def is_authenticated(self, *, timeout: int) -> bool:
        return True

    def check_status(self, *, timeout: int = 10) -> object:
        raise NotImplementedError


def build(a: str, b: str) -> str:
    return f"A={a}|B={b}"


def slots(prompt: str) -> tuple[str, str]:
    a, _, b = prompt.removeprefix("A=").partition("|B=")
    return a, b


def perfect_decide(prompt: str) -> str:
    a, _ = slots(prompt)
    return "A" if a == "cand" else "B"


def healthy_decide(prompt: str) -> str:
    a, b = slots(prompt)
    reference_slot, candidate = ("A", b) if a.startswith("REF") else ("B", a)
    return "tie" if candidate.startswith("PARA") else reference_slot


def poisoned_decide(prompt: str) -> str:
    a, _ = slots(prompt)
    return "A" if not a.startswith("REF") else "B"


def make_control_pairs() -> list[ControlPair]:
    return [
        ControlPair(row_id="r0", kind="paraphrase", candidate="PARA:r0", reference="REF:r0"),
        ControlPair(row_id="r0", kind="garbage", candidate="GARB:r0", reference="REF:r0"),
        ControlPair(row_id="r1", kind="paraphrase", candidate="PARA:r1", reference="REF:r1"),
        ControlPair(row_id="r1", kind="garbage", candidate="GARB:r1", reference="REF:r1"),
    ]


def verified(*, n: int = 6, floor: int = 4, rows: str = "rows-fp") -> VerifiedManifest:
    body = "packet"
    return verify_packet(
        packet_md=body,
        manifest={
            "packet_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "rows_sha256": rows,
            "gate": {"n": n, "floor": floor},
        },
    )


def green_proof(*, n: int = 6, floor: int = 4) -> GoldenProof:
    return prove_gate(report=AgreementReport(n=n, agree=n, panel_constant=False), manifest=verified(n=n, floor=floor))


def codex_judge(decide: Callable[[str], str] | None = None) -> tuple[Judge[Pairwise], StubBackend]:
    stub = StubBackend(provider="codex", models={"large": "gpt-5.2"}, decide=decide)
    return Judge(verdict_model=Pairwise, backend=stub), stub


# --- pure helpers -----------------------------------------------------------------------------


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
    assert {coin("row", seed) for seed in range(16)} == {True, False}


# --- JG2: fail-closed family mapping ----------------------------------------------------------


@pytest.mark.parametrize(
    "value, family",
    [
        pytest.param("gemini-2.5-pro", "google", id="gemini-model"),
        pytest.param("claude-3-7-sonnet", "anthropic", id="claude-model"),
        pytest.param("azure-openai", "openai", id="azure-openai"),
        pytest.param("gpt-4o", "openai", id="gpt-4o"),
    ],
)
def test_family_of_resolves_model_names(value: str, family: str) -> None:
    assert family_of(value) == family


@pytest.mark.parametrize("value", ["qwen3-coder", "", "claude-gpt-hybrid"])
def test_family_of_fails_closed_on_unknown_or_ambiguous(value: str) -> None:
    with pytest.raises(UnknownFamilyError):
        family_of(value)


def test_ensure_cross_family_refuses_same_family() -> None:
    with pytest.raises(CrossFamilyViolation, match="own family"):
        ensure_cross_family("anthropic", "anthropic")


@pytest.mark.parametrize(
    "judge_family, generator_family",
    [
        pytest.param("anthropic", "claude", id="claude-alias"),
        pytest.param("anthropic", "Claude", id="claude-alias-cased"),
        pytest.param("openai", "gpt", id="gpt-alias"),
        pytest.param("anthropic", " anthropic ", id="whitespace"),
        pytest.param("anthropic", "claude-3-7-sonnet", id="claude-model-name"),
    ],
)
def test_ensure_cross_family_refuses_an_aliased_same_family(judge_family: str, generator_family: str) -> None:
    with pytest.raises(CrossFamilyViolation, match="own family"):
        ensure_cross_family(judge_family, generator_family)


def test_ensure_cross_family_allows_a_different_family() -> None:
    ensure_cross_family("anthropic", "openai")


def test_ensure_cross_family_fails_closed_on_an_unknown_generator() -> None:
    # JG2: the old normalize_family fell back to the raw string, silently ALLOWING an unknown family.
    with pytest.raises(UnknownFamilyError):
        ensure_cross_family("anthropic", "qwen3-coder")


# --- JG1: Judge bound to a concrete backend ---------------------------------------------------


def test_judge_family_field_is_gone() -> None:
    with pytest.raises(TypeError):
        Judge(Pairwise, family="google")  # type: ignore[call-arg]


def test_judge_derives_family_and_identity_from_its_backend() -> None:
    judge, _ = codex_judge()
    assert judge.family == "openai"
    assert judge.model_name == "gpt-5.2"
    assert judge.identity == JudgeIdentity(
        provider="codex",
        model="gpt-5.2",
        verdict_schema_sha256=hashlib.sha256(canonical_json(Pairwise.model_json_schema())).hexdigest(),
    )


def test_bind_resolves_a_registry_backend_by_name() -> None:
    judge = Judge.bind(Pairwise, backend="codex")
    assert judge.family == "openai"
    assert judge.backend.provider == "codex"


@pytest.mark.parametrize("provider", ["mlx", "openai_endpoint"])
def test_judge_on_an_unbindable_provider_fails_at_construction(provider: str) -> None:
    with pytest.raises(UnknownFamilyError, match="no model family"):
        Judge(verdict_model=Pairwise, backend=StubBackend(provider=provider, models={"large": "x"}))


async def test_verdict_refuses_a_same_family_generator_before_any_spend() -> None:
    stub = StubBackend(provider="claude", models={"large": "opus"})
    judge: Judge[Pairwise] = Judge(verdict_model=Pairwise, backend=stub)
    with pytest.raises(CrossFamilyViolation, match="own family"):
        await judge.verdict("p", generator_family="claude-3-7-sonnet", grant=PanelGrant(verified()))
    assert stub.calls == []


async def test_run_verdict_uses_the_bound_backend_and_spends_exactly_once() -> None:
    judge, stub = codex_judge()
    result = await judge.verdict("p", generator_family="anthropic", grant=PanelGrant(verified()))
    assert isinstance(result, Pairwise)
    assert len(stub.calls) == 1  # the bound stub is the only executor; auto-select never ran
    assert stub.calls[0].model == "gpt-5.2"  # requested concrete model for the `large` tier


# --- JG3 + JG5: no ungated entrypoint ---------------------------------------------------------


def test_verdict_requires_a_grant() -> None:
    judge, _ = codex_judge()
    with pytest.raises(TypeError):
        judge.verdict("p", generator_family="anthropic")  # type: ignore[call-arg]


def test_pairwise_vote_requires_a_grant() -> None:
    judge, _ = codex_judge()
    with pytest.raises(TypeError):
        pairwise_vote(  # type: ignore[call-arg]
            judge,
            generator_family="anthropic",
            context=CTX,
            row_id="r",
            candidate="c",
            reference="ref",
            build_prompt=build,
            seed=1,
        )


def test_run_controls_requires_a_golden_proof() -> None:
    judge, _ = codex_judge()
    with pytest.raises(TypeError):
        run_controls(  # type: ignore[call-arg]
            judge,
            make_control_pairs(),
            generator_family="anthropic",
            context=CTX,
            build_prompt=build,
            seed=7,
        )


def test_judge_candidates_requires_a_golden_proof() -> None:
    judge, _ = codex_judge()
    with pytest.raises(TypeError):
        judge_candidates(  # type: ignore[call-arg]
            judge,
            [JudgeRow(row_id="r0", candidate="CAND:r0", reference="REF:r0")],
            generator_family="anthropic",
            controls=make_control_pairs(),
            context=CTX,
            build_prompt=build,
            seed=7,
        )


async def test_judge_candidates_blocks_on_flunked_controls_before_buying_candidates() -> None:
    # JG3: a poisoned judge must be caught by the controls; zero candidate rows are ever bought.
    judge, stub = codex_judge(decide=poisoned_decide)
    with pytest.raises(JudgeControlsViolation, match="garbage"):
        await judge_candidates(
            judge,
            [JudgeRow(row_id="r0", candidate="CAND:r0", reference="REF:r0")],
            generator_family="anthropic",
            controls=make_control_pairs(),
            golden=green_proof(),
            context=CTX,
            build_prompt=build,
            seed=7,
        )
    assert len(stub.calls) == 4  # the four control pairs
    assert all("CAND" not in spec.prompt for spec in stub.calls)


async def test_verdict_boundary_rechecks_a_red_clearance() -> None:
    # JG3: SpendClearance.check re-raises at the buy boundary, before any backend call.
    judge, stub = codex_judge()
    red_report = ControlsReport(
        epoch=HealthEpoch("e"), n_paraphrase=2, n_garbage=2, paraphrase_tie_rate=1.0, garbage_loss_rate=0.0
    )
    clearance = SpendClearance(golden=green_proof(), controls=red_report)
    with pytest.raises(JudgeControlsViolation, match="garbage"):
        await judge.verdict("p", generator_family="anthropic", grant=clearance)
    assert stub.calls == []


async def test_run_controls_blocks_behind_a_red_golden_proof() -> None:
    # JG5: controls run behind a green golden gate; a red hand-built proof blocks all spend.
    judge, stub = codex_judge(decide=healthy_decide)
    red_proof = GoldenProof(
        gate=GoldenGate(n=6, floor=4), report=AgreementReport(n=6, agree=1, panel_constant=False), rows_sha256="x"
    )
    with pytest.raises(GoldenGateViolation, match="agreement 1/6"):
        await run_controls(
            judge,
            make_control_pairs(),
            generator_family="anthropic",
            golden=red_proof,
            context=CTX,
            build_prompt=build,
            seed=7,
        )
    assert stub.calls == []


async def test_judge_candidates_refuses_a_same_family_judge() -> None:
    stub = StubBackend(provider="claude", models={"large": "opus"})
    judge: Judge[Pairwise] = Judge(verdict_model=Pairwise, backend=stub)
    with pytest.raises(CrossFamilyViolation, match="own family"):
        await judge_candidates(
            judge,
            [JudgeRow(row_id="r0", candidate="CAND:r0", reference="REF:r0")],
            generator_family="claude",
            controls=make_control_pairs(),
            golden=green_proof(),
            context=CTX,
            build_prompt=build,
            seed=7,
        )
    assert stub.calls == []


async def test_judge_candidates_returns_votes_after_controls_pass() -> None:
    judge, _ = codex_judge(decide=healthy_decide)
    votes = await judge_candidates(
        judge,
        [JudgeRow(row_id="r0", candidate="CAND:r0", reference="REF:r0")],
        generator_family="anthropic",
        controls=make_control_pairs(),
        golden=green_proof(),
        context=CTX,
        build_prompt=build,
        seed=7,
    )
    assert votes == [Vote.LOSS]


# --- JG4: cached controls can't mask a poisoned judge -----------------------------------------


def test_run_controls_has_no_cache_parameter() -> None:
    assert "cache" not in inspect.signature(run_controls).parameters


async def test_run_controls_ignores_a_warm_cache_and_rebuys_every_control(tmp_path: Path) -> None:
    judge, stub = codex_judge(decide=poisoned_decide)
    cache = VoteCache.open(tmp_path / "votes.json")
    seed = 7
    for pair in make_control_pairs():  # warm the cache with the healthy vote it "would" have served
        await cache.put(
            judge.identity,
            CTX,
            row_id=f"{pair.row_id}|control-{pair.kind}",
            candidate=pair.candidate,
            reference=pair.reference,
            seed=seed,
            vote=Vote.TIE if pair.kind == "paraphrase" else Vote.LOSS,
        )
    report = await run_controls(
        judge,
        make_control_pairs(),
        generator_family="anthropic",
        golden=green_proof(),
        context=CTX,
        build_prompt=build,
        seed=seed,
    )
    assert report.garbage_loss_rate == 0.0
    with pytest.raises(JudgeControlsViolation, match="garbage"):
        report.check()
    assert len(stub.calls) == 4  # every control re-bought; the warm cache masked nothing


async def test_two_run_controls_calls_double_spend_and_differ_in_epoch() -> None:
    judge, stub = codex_judge(decide=healthy_decide)

    async def once() -> ControlsReport:
        return await run_controls(
            judge,
            make_control_pairs(),
            generator_family="anthropic",
            golden=green_proof(),
            context=CTX,
            build_prompt=build,
            seed=7,
        )

    first, second = await once(), await once()
    assert len(stub.calls) == 8  # 4 + 4, no reuse across epochs
    assert first.epoch != second.epoch


def test_controls_check_requires_both_control_kinds() -> None:
    with pytest.raises(JudgeControlsViolation, match="no control pairs"):
        ControlsReport(
            epoch=HealthEpoch("e"), n_paraphrase=0, n_garbage=3, paraphrase_tie_rate=0.0, garbage_loss_rate=1.0
        ).check()


def test_controls_check_flags_a_low_paraphrase_tie_rate() -> None:
    with pytest.raises(JudgeControlsViolation, match="paraphrase tie rate"):
        ControlsReport(
            epoch=HealthEpoch("e"), n_paraphrase=10, n_garbage=10, paraphrase_tie_rate=0.2, garbage_loss_rate=1.0
        ).check()


# --- JG1 residual: canonical, schema-versioned vote-cache key ---------------------------------


def test_vote_key_is_delimiter_injection_proof() -> None:
    id_ = JudgeIdentity(provider="p", model="m", verdict_schema_sha256="s")
    assert VoteCache.key(id_, VoteContext("pv", "d|x"), row_id="r", candidate="c", reference="ref", seed=1) != (
        VoteCache.key(id_, VoteContext("pv", "d"), row_id="x|r", candidate="c", reference="ref", seed=1)
    )
    ia = JudgeIdentity(provider="a|b", model="c", verdict_schema_sha256="s")
    ib = JudgeIdentity(provider="a", model="b|c", verdict_schema_sha256="s")
    assert VoteCache.key(ia, CTX, row_id="r", candidate="c", reference="ref", seed=1) != (
        VoteCache.key(ib, CTX, row_id="r", candidate="c", reference="ref", seed=1)
    )


def test_vote_key_is_deterministic_and_field_sensitive() -> None:
    id_ = JudgeIdentity(provider="p", model="m", verdict_schema_sha256="s")
    base = VoteCache.key(id_, CTX, row_id="r", candidate="c", reference="ref", seed=1)
    assert base == VoteCache.key(id_, CTX, row_id="r", candidate="c", reference="ref", seed=1)
    variants = [
        VoteCache.key(
            JudgeIdentity(provider="p", model="m", verdict_schema_sha256="s2"),
            CTX,
            row_id="r",
            candidate="c",
            reference="ref",
            seed=1,
        ),
        VoteCache.key(id_, CTX, row_id="r2", candidate="c", reference="ref", seed=1),
        VoteCache.key(id_, CTX, row_id="r", candidate="c2", reference="ref", seed=1),
        VoteCache.key(id_, CTX, row_id="r", candidate="c", reference="ref2", seed=1),
        VoteCache.key(id_, CTX, row_id="r", candidate="c", reference="ref", seed=2),
        VoteCache.key(id_, VoteContext("v2", "d1"), row_id="r", candidate="c", reference="ref", seed=1),
        VoteCache.key(id_, VoteContext("v1", "d2"), row_id="r", candidate="c", reference="ref", seed=1),
    ]
    assert all(variant != base for variant in variants)
    assert len(set(variants)) == len(variants)


# --- vote cache + pairwise wiring -------------------------------------------------------------


async def test_vote_cache_roundtrips_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "votes.json"
    cache = VoteCache.open(path)
    judge, _ = codex_judge()
    identity = judge.identity
    assert cache.get(identity, CTX, row_id="r", candidate="cand", reference="ref", seed=1) is None
    await cache.put(identity, CTX, row_id="r", candidate="cand", reference="ref", seed=1, vote=Vote.WIN)
    assert cache.get(identity, CTX, row_id="r", candidate="cand", reference="ref", seed=1) is Vote.WIN
    reopened = VoteCache.open(path)
    assert reopened.get(identity, CTX, row_id="r", candidate="cand", reference="ref", seed=1) is Vote.WIN
    assert reopened.get(identity, CTX, row_id="r", candidate="cand", reference="ref", seed=2) is None


async def test_pairwise_vote_hits_cache_and_never_rebuys(tmp_path: Path) -> None:
    judge, stub = codex_judge()
    cache = VoteCache.open(tmp_path / "votes.json")
    grant = PanelGrant(verified())

    async def vote() -> Vote:
        return await pairwise_vote(
            judge,
            generator_family="anthropic",
            grant=grant,
            context=CTX,
            row_id="r",
            candidate="cand",
            reference="ref",
            build_prompt=build,
            seed=1,
            cache=cache,
        )

    first, second = await vote(), await vote()
    assert first is second
    assert len(stub.calls) == 1


async def test_cached_win_is_not_replayed_against_a_new_reference(tmp_path: Path) -> None:
    # JG1: a WIN cached against reference R1 must not be reused when the reference changes to R2.
    judge, stub = codex_judge()
    cache = VoteCache.open(tmp_path / "votes.json")
    grant = PanelGrant(verified())

    async def vote_against(reference: str) -> Vote:
        return await pairwise_vote(
            judge,
            generator_family="anthropic",
            grant=grant,
            context=CTX,
            row_id="r",
            candidate="cand",
            reference=reference,
            build_prompt=build,
            seed=1,
            cache=cache,
        )

    await vote_against("R1")
    await vote_against("R2")
    assert len(stub.calls) == 2
    assert cache.get(judge.identity, CTX, row_id="r", candidate="cand", reference="R1", seed=1) is not None
    assert cache.get(judge.identity, CTX, row_id="r", candidate="cand", reference="R2", seed=1) is not None


async def test_cache_misses_on_a_new_identity_prompt_version_or_digest(tmp_path: Path) -> None:
    # JG1: the cache key binds judge identity, prompt version, and dataset/config digest.
    cache = VoteCache.open(tmp_path / "votes.json")
    grant = PanelGrant(verified())
    stub_a = StubBackend(provider="codex", models={"large": "gpt-5.2"})
    stub_b = StubBackend(provider="codex", models={"large": "gpt-5.9"})  # different model -> different identity
    judge_a: Judge[Pairwise] = Judge(verdict_model=Pairwise, backend=stub_a)
    judge_b: Judge[Pairwise] = Judge(verdict_model=Pairwise, backend=stub_b)

    async def vote(judge: Judge[Pairwise], context: VoteContext) -> Vote:
        return await pairwise_vote(
            judge,
            generator_family="anthropic",
            grant=grant,
            context=context,
            row_id="r",
            candidate="cand",
            reference="ref",
            build_prompt=build,
            seed=1,
            cache=cache,
        )

    await vote(judge_a, CTX)
    await vote(judge_a, CTX)  # identical context -> cache hit, no new call
    assert len(stub_a.calls) == 1
    await vote(judge_b, CTX)  # different judge identity -> miss
    assert len(stub_b.calls) == 1
    await vote(judge_a, VoteContext(prompt_version="v2", digest="d1"))  # new prompt version -> miss
    await vote(judge_a, VoteContext(prompt_version="v1", digest="d2"))  # new digest -> miss
    assert len(stub_a.calls) == 3


async def test_pairwise_vote_refuses_same_family_before_the_cache_lookup(tmp_path: Path) -> None:
    # JG2: cross-family is checked before the cache, so a stored vote cannot leak a same-family verdict.
    stub = StubBackend(provider="claude", models={"large": "opus"})
    judge: Judge[Pairwise] = Judge(verdict_model=Pairwise, backend=stub)
    cache = VoteCache.open(tmp_path / "votes.json")
    await cache.put(judge.identity, CTX, row_id="r", candidate="c", reference="ref", seed=1, vote=Vote.WIN)
    for generator_family in ("anthropic", "claude-3-7-sonnet"):
        with pytest.raises(CrossFamilyViolation, match="own family"):
            await pairwise_vote(
                judge,
                generator_family=generator_family,
                grant=PanelGrant(verified()),
                context=CTX,
                row_id="r",
                candidate="c",
                reference="ref",
                build_prompt=build,
                seed=1,
                cache=cache,
            )
    assert stub.calls == []


async def test_pairwise_vote_fails_closed_on_unknown_family_before_the_cache_lookup(tmp_path: Path) -> None:
    # JG2: an unknown generator family raises before cache.get, so a seeded vote is never served.
    judge, stub = codex_judge()
    cache = VoteCache.open(tmp_path / "votes.json")
    await cache.put(judge.identity, CTX, row_id="r", candidate="c", reference="ref", seed=1, vote=Vote.WIN)
    with pytest.raises(UnknownFamilyError):
        await pairwise_vote(
            judge,
            generator_family="qwen3",
            grant=PanelGrant(verified()),
            context=CTX,
            row_id="r",
            candidate="c",
            reference="ref",
            build_prompt=build,
            seed=1,
            cache=cache,
        )
    assert stub.calls == []


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
async def test_pairwise_is_position_debiased(seed: int) -> None:
    judge, _ = codex_judge(decide=perfect_decide)
    vote = await pairwise_vote(
        judge,
        generator_family="anthropic",
        grant=PanelGrant(verified()),
        context=CTX,
        row_id="row",
        candidate="cand",
        reference="ref",
        build_prompt=build,
        seed=seed,
    )
    assert vote is Vote.WIN


async def test_run_controls_passes_a_healthy_judge() -> None:
    judge, _ = codex_judge(decide=healthy_decide)
    report = await run_controls(
        judge,
        make_control_pairs(),
        generator_family="anthropic",
        golden=green_proof(),
        context=CTX,
        build_prompt=build,
        seed=7,
    )
    assert report.n_paraphrase == report.n_garbage == 2
    assert report.paraphrase_tie_rate == 1.0
    assert report.garbage_loss_rate == 1.0
    report.check()


async def test_run_controls_fails_a_poisoned_judge() -> None:
    judge, _ = codex_judge(decide=poisoned_decide)
    report = await run_controls(
        judge,
        make_control_pairs(),
        generator_family="anthropic",
        golden=green_proof(),
        context=CTX,
        build_prompt=build,
        seed=7,
    )
    assert report.garbage_loss_rate == 0.0
    with pytest.raises(JudgeControlsViolation, match="garbage"):
        report.check()


# --- retry / concurrency helpers --------------------------------------------------------------


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
        Judge.bind(Pairwise, backend="claude"),
        generator_family="openai",
        grant=PanelGrant(verified(n=1, floor=1)),
        context=VoteContext(prompt_version="live-v1", digest="live"),
        row_id="live",
        candidate="Fix the failing test in test_math.py before adding features.",
        reference="Delete the whole repository and start over.",
        build_prompt=prompt,
        seed=1,
    )
    assert vote is Vote.WIN
