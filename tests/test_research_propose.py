from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from athome.registry import VersionInfo
from athome.research.cells import Cell
from athome.research.journal import JournalRow, Verdict
from athome.research.nightly import MorningReport
from athome.research.propose import (
    MAX_PROPOSAL_ATTEMPTS,
    REJECTION_HEADER,
    Proposal,
    ProposalViolation,
    ProposerContext,
    build_prompt,
    propose,
    validate_proposal,
)
from athome.research.spec import Budget, ImmutableViolation
from tests.test_research_policy import make_policy, make_template

if TYPE_CHECKING:
    from collections.abc import Sequence


def make_proposal(**overrides: object) -> Proposal:
    defaults: dict[str, object] = {
        "template": "sweep",
        "name": "lr-sweep",
        "mutable_paths": ("experiments/recipe.json",),
        "max_units": 4,
        "max_usd": 10.0,
        "max_wall_s": 7200.0,
        "hard_kill_s": 1800.0,
        "hypothesis": "A lower learning rate converges further.",
    }
    return Proposal.model_validate(defaults | overrides)


def scripted_backend(proposals: Sequence[Proposal]) -> object:
    spawnllm = pytest.importorskip("spawnllm")

    class ScriptedBackend(spawnllm.LlmBackend):
        def __init__(self) -> None:
            self.provider = "codex"  # type: ignore[misc]
            self.models = {"large": "gpt-5.2"}  # type: ignore[misc]
            self.calls: list[object] = []
            self.script = iter(proposals)

        async def aexecute(self, spec: object) -> object:
            self.calls.append(spec)
            return spawnllm.Response(
                spec=spec, output=spawnllm.Output(raw=""), result=spawnllm.Result(raw="", parsed=next(self.script))
            )

        def execute(self, spec: object) -> object:
            raise NotImplementedError

        def env(self, spec: object) -> dict[str, str]:
            return {}

        def is_authenticated(self, *, timeout: int) -> bool:
            return True

        def check_status(self, *, timeout: int = 10) -> object:
            raise NotImplementedError

    return ScriptedBackend()


def test_admitted_spec_copies_template_verbatim_and_prefixes_seq() -> None:
    spec = validate_proposal(make_proposal(), policy := make_policy(), seq=7)
    template = policy.templates[0]
    assert spec.name == "007-lr-sweep"
    assert spec.metric_command == template.metric_command
    assert spec.metric_key == template.metric_key
    assert spec.direction == template.direction
    assert spec.immutable_paths == template.immutable_paths
    assert spec.metric_file == template.metric_file
    assert spec.mutable_paths == ("experiments/recipe.json",)
    assert spec.budget == Budget(max_units=4, max_wall_s=7200.0, hard_kill_s=1800.0, max_usd=10.0)
    assert spec.hypothesis == "A lower learning rate converges further."


@pytest.mark.parametrize("seq", [0, -3, True], ids=["zero", "negative", "bool"])
def test_seq_must_be_a_positive_non_bool_int(seq: int) -> None:
    with pytest.raises(ProposalViolation, match="seq"):
        validate_proposal(make_proposal(), make_policy(), seq=seq)


def test_unknown_template_is_refused() -> None:
    with pytest.raises(ProposalViolation, match="unknown template"):
        validate_proposal(make_proposal(template="ghost"), make_policy(), seq=1)


@pytest.mark.parametrize(
    "name", ["", "a/b", "a b", "exp\n1", "$(rm -rf)"], ids=["empty", "slash", "space", "newline", "shell"]
)
def test_name_must_fully_match_name_re(name: str) -> None:
    with pytest.raises(ProposalViolation, match="fully match"):
        validate_proposal(make_proposal(name=name), make_policy(), seq=1)


def test_empty_mutable_paths_is_refused() -> None:
    with pytest.raises(ProposalViolation, match="empty"):
        validate_proposal(make_proposal(mutable_paths=()), make_policy(), seq=1)


def test_mutable_path_outside_allowlist_is_refused() -> None:
    with pytest.raises(ProposalViolation, match="outside the template allowlist"):
        validate_proposal(make_proposal(mutable_paths=("experiments/recipe.json", "train.py")), make_policy(), seq=1)


def test_allowlist_membership_is_exact_never_fnmatch() -> None:
    with pytest.raises(ProposalViolation, match="exact membership"):
        validate_proposal(make_proposal(mutable_paths=("experiments/foo.toml",)), make_policy(), seq=1)


@pytest.mark.parametrize("dropped", ["max_usd", "max_wall_s"], ids=["usd", "wall"])
def test_auto_mode_requires_usd_and_wall_backstops(dropped: str) -> None:
    with pytest.raises(ProposalViolation, match="auto mode requires"):
        validate_proposal(make_proposal(**{dropped: None}), make_policy(), seq=1)


def test_gated_mode_admits_missing_backstops() -> None:
    spec = validate_proposal(
        make_proposal(max_usd=None, max_wall_s=None, hard_kill_s=None), make_policy(mode="gated"), seq=1
    )
    assert spec.budget == Budget(max_units=4)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"max_units": 9}, id="units-over"),
        pytest.param({"max_usd": 25.01}, id="usd-over"),
        pytest.param({"max_wall_s": 999999.0}, id="wall-over"),
        pytest.param({"hard_kill_s": 3600.5}, id="kill-over"),
        pytest.param({"max_units": 0}, id="units-zero"),
        pytest.param({"max_usd": -1.0}, id="usd-negative"),
        pytest.param({"max_usd": float("inf")}, id="usd-inf"),
        pytest.param({"max_usd": float("nan")}, id="usd-nan"),
    ],
)
def test_budgets_are_bounded_by_refusal_never_clamped(overrides: dict[str, object]) -> None:
    with pytest.raises(ProposalViolation, match="ceiling"):
        validate_proposal(make_proposal(**overrides), make_policy(), seq=1)


def test_value_on_an_axis_with_no_ceiling_is_refused() -> None:
    policy = make_policy(ceilings=Budget(max_units=8, max_wall_s=14400.0, max_usd=25.0))
    with pytest.raises(ProposalViolation, match="no operator ceiling"):
        validate_proposal(make_proposal(), policy, seq=1)


def test_proposal_has_no_slot_for_executable_fields() -> None:
    for field, value in {
        "metric_command": ["rm", "-rf"],
        "metric_key": "forged",
        "direction": "min",
        "immutable_paths": [],
    }.items():
        with pytest.raises(ValidationError):
            make_proposal(**{field: value})


def test_admission_runs_through_experiment_spec_defense_in_depth() -> None:
    # Bypass the template's own allowlist guard to prove the ExperimentSpec layer fires independently.
    template = make_template(mutable_allowlist=("experiments/recipe.json",))
    object.__setattr__(template, "mutable_allowlist", ("**/*.py",))
    policy = make_policy(templates=(template,))
    with pytest.raises(ImmutableViolation, match="tight allowlist"):
        validate_proposal(make_proposal(mutable_paths=("**/*.py",)), policy, seq=1)


def make_context() -> ProposerContext:
    row = JournalRow(
        unit=0, commit="abc123", metric=0.91, verdict=Verdict.KEEP, resources={"usd": 1.0}, description="tuned lr"
    )
    return ProposerContext(
        retros=("Retro: the 4B arm plateaued early.",),
        reports=(MorningReport(experiment="001-lr", units=3, kept=1, crashes=1, best=row, rows=(row,)),),
        leaderboard=(Cell(experiment="001-lr", unit=0, arm="candidate", metric=0.91, verdict="keep"),),
        current=VersionInfo(name="watcher", version="v001-20260710-abcdef123456", path=Path("/tmp/v001"), metadata={}),
        failures=("round 2 rejected: unknown template 'ghost'",),
    )


def test_prompt_carries_policy_bounds_and_sanitized_history() -> None:
    prompt = build_prompt(make_policy(), make_context())
    assert "### sweep" in prompt and "experiments/recipe.json" in prompt
    assert "max_usd <= 25.0" in prompt and "required" in prompt
    assert "Retro: the 4B arm plateaued early." in prompt
    assert "001-lr: 3 units, 1 kept, 1 crashed; best kept metric 0.91" in prompt
    assert "watcher v001-20260710-abcdef123456" in prompt
    assert "round 2 rejected: unknown template 'ghost'" in prompt


def test_prompt_omits_empty_history_sections() -> None:
    prompt = build_prompt(make_policy(), ProposerContext())
    assert "Retros" not in prompt and "Leaderboard" not in prompt and "Prior" not in prompt


async def test_propose_reprompts_with_the_violation_then_admits() -> None:
    backend = scripted_backend([make_proposal(template="ghost"), make_proposal()])
    round_ = await propose(make_policy(), ProposerContext(), backend=backend, seq=3)
    assert round_.attempts == 2
    assert round_.spec.name == "003-lr-sweep"
    assert len(backend.calls) == 2
    retry_prompt = backend.calls[1].prompt
    assert REJECTION_HEADER in retry_prompt and "unknown template 'ghost'" in retry_prompt


async def test_propose_raises_after_max_attempts() -> None:
    backend = scripted_backend([make_proposal(max_usd=999.0)] * MAX_PROPOSAL_ATTEMPTS)
    with pytest.raises(ProposalViolation, match="ceiling"):
        await propose(make_policy(), ProposerContext(), backend=backend, seq=1)
    assert len(backend.calls) == MAX_PROPOSAL_ATTEMPTS
