from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

from athome.research.spec import Budget, ExperimentSpec, ImmutableViolation, unbounded_glob

if TYPE_CHECKING:
    from pathlib import Path

FULL_TOML = dedent("""
    name = "toy"
    metric_command = ["python", "score.py"]
    metric_key = "accuracy"
    direction = "max"
    mutable_paths = ["train.py", "model/*.py"]
    immutable_paths = ["score.py", "data/*"]
    metric_file = ".metric.json"

    [budget]
    max_units = 20
    max_wall_s = 3600.0
    hard_kill_s = 300.0
    max_usd = 10.0
""")

MINIMAL_TOML = dedent("""
    name = "toy"
    metric_command = ["python", "score.py"]
    metric_key = "loss"
    direction = "min"
    mutable_paths = ["train.py"]
    immutable_paths = ["score.py"]

    [budget]
    max_units = 5
""")


def test_load_round_trips_full_toml(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text(FULL_TOML)
    assert ExperimentSpec.load(path) == ExperimentSpec(
        name="toy",
        metric_command=("python", "score.py"),
        metric_key="accuracy",
        direction="max",
        mutable_paths=("train.py", "model/*.py"),
        immutable_paths=("score.py", "data/*"),
        budget=Budget(max_units=20, max_wall_s=3600.0, hard_kill_s=300.0, max_usd=10.0),
        metric_file=".metric.json",
    )


def test_load_applies_metric_file_default_and_optional_budget_fields(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text(MINIMAL_TOML)
    spec = ExperimentSpec.load(path)
    assert spec.metric_file == ".athome-metric.json"
    assert spec.direction == "min"
    assert spec.budget == Budget(max_units=5)
    assert spec.budget.max_wall_s is None and spec.budget.hard_kill_s is None and spec.budget.max_usd is None


def test_lists_become_tuples(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text(MINIMAL_TOML)
    spec = ExperimentSpec.load(path)
    assert isinstance(spec.metric_command, tuple)
    assert isinstance(spec.mutable_paths, tuple)
    assert isinstance(spec.immutable_paths, tuple)


@pytest.mark.parametrize(
    "pattern, expected",
    [
        pytest.param("*", True, id="bare-star"),
        pytest.param("**", True, id="bare-doublestar"),
        pytest.param("*/train.py", True, id="star-prefixed"),
        pytest.param("**/x.py", True, id="doublestar-prefixed"),
        pytest.param("train.py", False, id="anchored-file"),
        pytest.param("src/*.py", False, id="anchored-dir-glob"),
        pytest.param("eval/**", False, id="anchored-recursive"),
    ],
)
def test_unbounded_glob(pattern: str, expected: bool) -> None:
    assert unbounded_glob(pattern) is expected


def make_spec(*, mutable_paths: tuple[str, ...]) -> ExperimentSpec:
    return ExperimentSpec(
        name="toy",
        metric_command=("python", "score.py"),
        metric_key="loss",
        direction="min",
        mutable_paths=mutable_paths,
        immutable_paths=("score.py",),
        budget=Budget(max_units=1),
    )


@pytest.mark.parametrize(
    "pattern", ["*", "**", "*/train.py", "**/x.py"], ids=["star", "doublestar", "star-x", "dstar-x"]
)
def test_post_init_rejects_unbounded_mutable_globs(pattern: str) -> None:
    with pytest.raises(ImmutableViolation, match="tight allowlist"):
        make_spec(mutable_paths=("train.py", pattern))


def test_post_init_accepts_anchored_mutable_globs() -> None:
    spec = make_spec(mutable_paths=("train.py", "src/*.py", "eval/**"))
    assert spec.mutable_paths == ("train.py", "src/*.py", "eval/**")
