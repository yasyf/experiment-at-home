from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

from athome.research.policy import CampaignBudget, ExperimentTemplate, PolicyViolation, ProposalPolicy
from athome.research.spec import Budget, UnknownSpecField

if TYPE_CHECKING:
    from pathlib import Path

POLICY_TOML = dedent("""
    mode = "auto"

    [campaign]
    max_experiments = 5
    max_total_usd = 100.0
    max_wall_s = 86400.0
    max_consecutive_failures = 3

    [ceilings]
    max_units = 8
    max_wall_s = 14400.0
    hard_kill_s = 3600.0
    max_usd = 25.0

    [[templates]]
    name = "sweep"
    metric_command = ["cc-steer", "retrain", "--score-only"]
    metric_key = "auc"
    direction = "max"
    immutable_paths = ["scorer/**"]
    mutable_allowlist = ["experiments/recipe.json", "experiments/*.toml"]
""")

CEILINGS = Budget(max_units=8, max_wall_s=14400.0, hard_kill_s=3600.0, max_usd=25.0)
CAMPAIGN = CampaignBudget(max_experiments=5, max_total_usd=100.0, max_wall_s=86400.0, max_consecutive_failures=3)


def make_template(**overrides: object) -> ExperimentTemplate:
    defaults: dict[str, object] = {
        "name": "sweep",
        "metric_command": ("cc-steer", "retrain", "--score-only"),
        "metric_key": "auc",
        "direction": "max",
        "immutable_paths": ("scorer/**",),
        "mutable_allowlist": ("experiments/recipe.json", "experiments/*.toml"),
    }
    return ExperimentTemplate(**(defaults | overrides))


def make_policy(**overrides: object) -> ProposalPolicy:
    defaults: dict[str, object] = {
        "mode": "auto",
        "campaign": CAMPAIGN,
        "ceilings": CEILINGS,
        "templates": (make_template(),),
    }
    return ProposalPolicy(**(defaults | overrides))


def write_policy(tmp_path: Path, content: str) -> Path:
    (path := tmp_path / "policy.toml").write_text(content)
    return path


def test_load_round_trips_policy_toml(tmp_path: Path) -> None:
    assert ProposalPolicy.load(write_policy(tmp_path, POLICY_TOML)) == make_policy()


def test_load_defaults_template_metric_file(tmp_path: Path) -> None:
    policy = ProposalPolicy.load(write_policy(tmp_path, POLICY_TOML))
    assert policy.templates[0].metric_file == ".athome-metric.json"
    assert isinstance(policy.templates[0].metric_command, tuple)
    assert isinstance(policy.templates[0].mutable_allowlist, tuple)


@pytest.mark.parametrize(
    "mutation, smuggled",
    [
        pytest.param(('mode = "auto"', 'mode = "auto"\nallow_everything = true'), "allow_everything", id="top"),
        pytest.param(("[campaign]", "[campaign]\nbonus = 1"), "bonus", id="campaign"),
        pytest.param(("[ceilings]", "[ceilings]\nmax_gpus = 4"), "max_gpus", id="ceilings"),
        pytest.param(('metric_key = "auc"', 'metric_key = "auc"\nshell = "sh"'), "shell", id="template"),
    ],
)
def test_load_refuses_unknown_fields_at_every_level(tmp_path: Path, mutation: tuple[str, str], smuggled: str) -> None:
    old, new = mutation
    with pytest.raises(UnknownSpecField, match=smuggled):
        ProposalPolicy.load(write_policy(tmp_path, POLICY_TOML.replace(old, new)))


def test_gated_mode_loads_without_usd_and_wall_ceilings() -> None:
    policy = make_policy(mode="gated", ceilings=Budget(max_units=8))
    assert policy.ceilings.max_usd is None and policy.ceilings.max_wall_s is None


@pytest.mark.parametrize(
    "ceilings",
    [
        pytest.param(Budget(max_units=8, max_wall_s=14400.0), id="no-usd"),
        pytest.param(Budget(max_units=8, max_usd=25.0), id="no-wall"),
    ],
)
def test_auto_mode_requires_usd_and_wall_ceilings(ceilings: Budget) -> None:
    with pytest.raises(PolicyViolation, match="auto mode requires"):
        make_policy(ceilings=ceilings)


def test_unknown_mode_is_refused() -> None:
    with pytest.raises(PolicyViolation, match="mode"):
        make_policy(mode="yolo")


def test_unknown_template_direction_is_refused() -> None:
    with pytest.raises(PolicyViolation, match="direction"):
        make_template(direction="sideways")


def test_unbounded_allowlist_glob_is_refused() -> None:
    with pytest.raises(PolicyViolation, match="unbounded"):
        make_template(mutable_allowlist=("experiments/recipe.json", "**/*.py"))


def test_empty_allowlist_is_refused() -> None:
    with pytest.raises(PolicyViolation, match="empty mutable_allowlist"):
        make_template(mutable_allowlist=())


def test_policy_without_templates_is_refused() -> None:
    with pytest.raises(PolicyViolation, match="no templates"):
        make_policy(templates=())


def test_duplicate_template_names_are_refused() -> None:
    with pytest.raises(PolicyViolation, match="duplicate"):
        make_policy(templates=(make_template(), make_template(metric_key="loss")))
