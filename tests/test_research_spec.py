from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

from athome.research.spec import (
    Budget,
    ExperimentSpec,
    ImmutableViolation,
    InvalidBudget,
    UnconfinedPath,
    UnknownSpecField,
    unbounded_glob,
)

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


def test_load_refuses_unknown_spec_fields(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text(MINIMAL_TOML.replace("[budget]", 'smuggled = "x"\n\n[budget]'))
    with pytest.raises(UnknownSpecField, match="smuggled"):
        ExperimentSpec.load(path)


def test_load_refuses_unknown_budget_fields(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text(MINIMAL_TOML + "max_gpus = 4\n")
    with pytest.raises(UnknownSpecField, match="max_gpus"):
        ExperimentSpec.load(path)


def test_load_round_trips_hypothesis(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text(MINIMAL_TOML.replace("[budget]", 'hypothesis = "smaller batches help"\n\n[budget]'))
    assert ExperimentSpec.load(path).hypothesis == "smaller batches help"


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


def make_spec(*, mutable_paths: tuple[str, ...], metric_file: str = ".athome-metric.json") -> ExperimentSpec:
    return ExperimentSpec(
        name="toy",
        metric_command=("python", "score.py"),
        metric_key="loss",
        direction="min",
        mutable_paths=mutable_paths,
        immutable_paths=("score.py",),
        budget=Budget(max_units=1),
        metric_file=metric_file,
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


@pytest.mark.parametrize(
    "metric_file",
    ["/etc/passwd", "../outside.json", "logs/../../outside.json", ""],
    ids=["absolute", "traversal", "nested-traversal", "empty"],
)
def test_post_init_refuses_a_metric_file_escaping_the_workdir(metric_file: str) -> None:
    with pytest.raises(UnconfinedPath, match="work directory"):
        make_spec(mutable_paths=("train.py",), metric_file=metric_file)


def test_post_init_accepts_a_nested_relative_metric_file() -> None:
    assert make_spec(mutable_paths=("train.py",), metric_file="logs/.metric.json").metric_file == "logs/.metric.json"


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"max_units": 0}, id="units-zero"),
        pytest.param({"max_units": -2}, id="units-negative"),
        pytest.param({"max_units": True}, id="units-bool"),
        pytest.param({"max_usd": float("inf")}, id="usd-inf"),
        pytest.param({"max_wall_s": float("nan")}, id="wall-nan"),
        pytest.param({"hard_kill_s": 0.0}, id="kill-zero"),
        pytest.param({"max_usd": -1.0}, id="usd-negative"),
        pytest.param({"max_wall_s": True}, id="wall-bool"),
    ],
)
def test_budget_refuses_non_finite_or_non_positive_caps(overrides: dict[str, object]) -> None:
    with pytest.raises(InvalidBudget):
        Budget(**{"max_units": 5} | overrides)
