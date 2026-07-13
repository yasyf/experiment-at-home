from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

from athome.research.spec import Budget, ExperimentSpec

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
