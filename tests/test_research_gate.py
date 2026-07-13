from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from athome.research.errors import ResearchError
from athome.research.gate import (
    PromotionVerdict,
    TreeChange,
    blocking_invariants,
    bootstrap_ci_gate,
    immutable_violations,
    is_autoloader,
    monotone_gate,
    parse_diff_tree,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

MUTABLE = ("train.py", "src/**")
IMMUTABLE = ("score.py", "eval/**")


class ConstantBatch(ResearchError):
    pass


@pytest.mark.parametrize(
    "candidate, incumbent, direction, floor, expected",
    [
        pytest.param(0.6, 0.5, "max", 0.0, True, id="max-strict-improvement-keeps"),
        pytest.param(0.5, 0.5, "max", 0.0, False, id="max-tie-discards"),
        pytest.param(0.5, 0.6, "max", 0.0, False, id="max-regression-discards"),
        pytest.param(0.4, 0.5, "min", 0.0, True, id="min-improvement-keeps"),
        pytest.param(0.5, 0.5, "min", 0.0, False, id="min-tie-discards"),
        pytest.param(0.6, 0.5, "min", 0.0, False, id="min-regression-discards"),
        pytest.param(0.55, 0.5, "max", 0.1, False, id="max-within-floor-discards"),
        pytest.param(0.65, 0.5, "max", 0.1, True, id="max-clears-floor-keeps"),
        pytest.param(0.45, 0.5, "min", 0.1, False, id="min-within-floor-discards"),
        pytest.param(0.35, 0.5, "min", 0.1, True, id="min-clears-floor-keeps"),
        pytest.param(0.7, None, "max", 0.0, True, id="max-first-candidate-keeps"),
        pytest.param(0.7, None, "min", 0.5, True, id="min-first-candidate-keeps"),
    ],
)
def test_monotone_gate(candidate: float, incumbent: float | None, direction: str, floor: float, expected: bool) -> None:
    assert monotone_gate(candidate, incumbent, direction=direction, floor=floor) is expected


def test_bootstrap_overlapping_cis_keep_incumbent() -> None:
    incumbent = [0.50, 0.51, 0.49, 0.50, 0.52, 0.48]
    candidate = [0.51, 0.50, 0.52, 0.49, 0.50, 0.51]
    verdict = bootstrap_ci_gate(incumbent, candidate, direction="max")
    assert isinstance(verdict, PromotionVerdict)
    assert verdict.promote is False
    assert "overlapping" in verdict.reason


def test_bootstrap_clear_win_promotes() -> None:
    incumbent = [0.50, 0.51, 0.49, 0.50, 0.52, 0.48]
    candidate = [0.90, 0.91, 0.89, 0.92, 0.90, 0.91]
    assert bootstrap_ci_gate(incumbent, candidate, direction="max").promote is True


def test_bootstrap_clear_loss_keeps_incumbent() -> None:
    incumbent = [0.90, 0.91, 0.89, 0.92, 0.90, 0.91]
    candidate = [0.50, 0.51, 0.49, 0.50, 0.52, 0.48]
    assert bootstrap_ci_gate(incumbent, candidate, direction="max").promote is False


def test_bootstrap_min_direction_promotes_the_lower_sample() -> None:
    incumbent = [0.90, 0.91, 0.89, 0.92, 0.90]
    candidate = [0.10, 0.11, 0.09, 0.12, 0.10]
    assert bootstrap_ci_gate(incumbent, candidate, direction="min").promote is True


def test_bootstrap_high_floor_blocks_a_separated_but_small_win() -> None:
    incumbent = [0.500, 0.501, 0.499, 0.500, 0.502, 0.498]
    candidate = [0.520, 0.521, 0.519, 0.520, 0.522, 0.518]
    assert bootstrap_ci_gate(incumbent, candidate, direction="max", floor=0.1).promote is False


def test_bootstrap_verdict_is_deterministic() -> None:
    incumbent = [0.50, 0.51, 0.49, 0.50, 0.52, 0.48]
    candidate = [0.62, 0.63, 0.61, 0.64, 0.60, 0.62]
    first = bootstrap_ci_gate(incumbent, candidate, direction="max")
    second = bootstrap_ci_gate(incumbent, candidate, direction="max")
    assert first == second


def test_blocking_invariants_raises_when_a_check_fails() -> None:
    def constant_decider(rows: Sequence[object]) -> None:
        if len(set(rows)) == 1:
            raise ConstantBatch("every row identical")

    with pytest.raises(ConstantBatch):
        blocking_invariants(["x", "x", "x"], [constant_decider])


def test_blocking_invariants_passes_when_all_checks_pass() -> None:
    calls: list[str] = []

    def record(rows: Sequence[object]) -> None:
        calls.append("ran")

    blocking_invariants([1, 2, 3], [record, record])
    assert calls == ["ran", "ran"]


@pytest.mark.parametrize(
    "change, expected",
    [
        pytest.param(TreeChange("train.py", "100644"), False, id="mutable-top-level-allowed"),
        pytest.param(TreeChange("src/a/b.py", "100644"), False, id="mutable-nested-allowed"),
        pytest.param(TreeChange("score.py", "100644"), True, id="immutable-edit-rejected"),
        pytest.param(TreeChange("eval/a/b.py", "100644"), True, id="immutable-nested-rejected"),
        pytest.param(TreeChange("score.py", "000000"), True, id="immutable-delete-rejected"),
        pytest.param(TreeChange("json.py", "100644"), True, id="undeclared-file-rejected-by-allowlist"),
        pytest.param(TreeChange("train.py", "120000"), True, id="symlink-rejected-even-in-allowlist"),
        pytest.param(TreeChange("src/__init__.py", "100644"), True, id="autoloader-init-in-allowlist-rejected"),
        pytest.param(TreeChange("src/conftest.py", "100644"), True, id="autoloader-conftest-in-allowlist-rejected"),
        pytest.param(TreeChange("src/sitecustomize.py", "100644"), True, id="autoloader-sitecustomize-rejected"),
        pytest.param(TreeChange("src/usercustomize.py", "100644"), True, id="autoloader-usercustomize-rejected"),
        pytest.param(TreeChange("src/hook.pth", "100644"), True, id="autoloader-pth-rejected"),
        pytest.param(TreeChange("src/__init__.py", "000000"), True, id="autoloader-init-deletion-rejected"),
    ],
)
def test_immutable_violations(change: TreeChange, expected: bool) -> None:
    assert bool(immutable_violations([change], mutable=MUTABLE, immutable=IMMUTABLE)) is expected


@pytest.mark.parametrize(
    "path, expected",
    [
        pytest.param("__init__.py", True, id="top-level-init"),
        pytest.param("pkg/sub/__init__.py", True, id="nested-init"),
        pytest.param("conftest.py", True, id="conftest"),
        pytest.param("sitecustomize.py", True, id="sitecustomize"),
        pytest.param("usercustomize.py", True, id="usercustomize"),
        pytest.param("a/b/hook.pth", True, id="pth-file"),
        pytest.param("train.py", False, id="leaf-module"),
        pytest.param("src/model.py", False, id="nested-leaf-module"),
        pytest.param("init.py", False, id="init-not-dunder"),
        pytest.param("conftest_helpers.py", False, id="conftest-prefix-only"),
    ],
)
def test_is_autoloader(path: str, expected: bool) -> None:
    assert is_autoloader(path) is expected


def test_immutable_violations_reports_every_offending_path() -> None:
    changes = [TreeChange("train.py", "100644"), TreeChange("score.py", "100644"), TreeChange("new.py", "100644")]
    assert immutable_violations(changes, mutable=MUTABLE, immutable=IMMUTABLE) == ["score.py", "new.py"]


def test_parse_diff_tree_recovers_paths_and_modes() -> None:
    raw = ":100644 000000 aaaa bbbb D\0eval/a/b.py\0:000000 120000 cccc dddd A\0link.py\0"
    assert parse_diff_tree(raw) == [TreeChange("eval/a/b.py", "000000"), TreeChange("link.py", "120000")]
