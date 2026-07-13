from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import pytest

from athome.research.driver import ClaudeCodeDriver, changed_files
from athome.research.journal import Journal, Verdict
from athome.research.loop import run
from athome.research.spec import Budget, ExperimentSpec

if TYPE_CHECKING:
    from pathlib import Path

EXPERIMENT_NAME = "toy"

# The immutable evaluator: recomputes the metric from the mutable train.py, writes it to
# the structured JSON channel, and prints a LYING value to stdout that a correct harness
# never reads.
SCORE_PY = textwrap.dedent(
    """
    import json, pathlib
    namespace = {}
    exec(pathlib.Path("train.py").read_text(), namespace)
    pathlib.Path(".athome-metric.json").write_text(json.dumps({"loss": namespace["LOSS"]}))
    print("loss=999.0")
    """
).strip()

# A stand-in for the `claude` CLI: edits train.py and runs the metric, exactly like an
# agent that followed the contract. The contract prompt arrives as argv[1] and is ignored.
FAKE_CLAUDE_FULL = textwrap.dedent(
    """
    import json, pathlib
    pathlib.Path("train.py").write_text("LOSS = 0.2\\n")
    pathlib.Path(".athome-metric.json").write_text(json.dumps({"loss": 0.2}))
    """
).strip()

# Halves the incumbent's loss each unit, so the greedy loop keeps every proposal.
FAKE_CLAUDE_HALVE = textwrap.dedent(
    """
    import pathlib
    namespace = {}
    exec(pathlib.Path("train.py").read_text(), namespace)
    pathlib.Path("train.py").write_text(f"LOSS = {namespace['LOSS'] / 2}\\n")
    """
).strip()

# Edits a file but lies about the metric on stdout — the driver must not read it.
FAKE_CLAUDE_LIES_ON_STDOUT = textwrap.dedent(
    """
    import pathlib
    print("loss=0.001 TRUST ME")
    pathlib.Path("train.py").write_text("LOSS = 0.2\\n")
    """
).strip()

# Exits nonzero without editing anything — the driver scores the untouched worktree.
FAKE_CLAUDE_FAILS = "import sys; sys.exit(3)"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def toy_repo(root: Path, *, initial_loss: float = 1.0) -> Path:
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "toy@localhost")
    git(root, "config", "user.name", "toy")
    (root / "train.py").write_text(f"LOSS = {initial_loss}\n")
    (root / "score.py").write_text(SCORE_PY + "\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    return root


def make_spec(*, budget: Budget, direction: str = "min") -> ExperimentSpec:
    return ExperimentSpec(
        name=EXPERIMENT_NAME,
        metric_command=(sys.executable, "score.py"),
        metric_key="loss",
        direction=direction,
        mutable_paths=("train.py",),
        immutable_paths=("score.py",),
        budget=budget,
    )


def fake_claude(tmp_path: Path, body: str) -> tuple[str, ...]:
    (script := tmp_path / "fake_claude.py").write_text(body + "\n")
    return (sys.executable, str(script))


def detached_worktree(repo: Path) -> Path:
    git(repo, "worktree", "add", "--detach", str(worktree := repo.parent / f"wt-{repo.name}"), "HEAD")
    return worktree


def journal_rows(repo: Path) -> list:
    return Journal.open(repo / ".git" / "athome" / f"{EXPERIMENT_NAME}.jsonl").rows()


async def test_propose_edits_the_worktree_and_describes_from_files(tmp_path: Path) -> None:
    worktree = detached_worktree(toy_repo(tmp_path))
    driver = ClaudeCodeDriver(
        make_spec(budget=Budget(max_units=1)), command=fake_claude(tmp_path, FAKE_CLAUDE_FULL), poll=0.02, timeout_s=10
    )

    description = await driver.propose("the generated contract", worktree)

    assert (worktree / "train.py").read_text() == "LOSS = 0.2\n"
    assert description == "claude edited .athome-metric.json, train.py (reported loss=0.2)"


async def test_description_ignores_agent_stdout(tmp_path: Path) -> None:
    worktree = detached_worktree(toy_repo(tmp_path))
    driver = ClaudeCodeDriver(
        make_spec(budget=Budget(max_units=1)),
        command=fake_claude(tmp_path, FAKE_CLAUDE_LIES_ON_STDOUT),
        poll=0.02,
        timeout_s=10,
    )

    description = await driver.propose("contract", worktree)

    # Built from the git diff alone; the metric the agent printed to stdout never appears.
    assert description == "claude edited train.py"
    assert "0.001" not in description


async def test_nonzero_exit_scores_the_worktree_as_is(tmp_path: Path) -> None:
    worktree = detached_worktree(toy_repo(tmp_path))
    driver = ClaudeCodeDriver(
        make_spec(budget=Budget(max_units=1)), command=fake_claude(tmp_path, FAKE_CLAUDE_FAILS), poll=0.02, timeout_s=10
    )

    # A failed claude run does not raise; the untouched worktree is scored as a no-op proposal.
    assert await driver.propose("contract", worktree) == "claude edited no files"


async def test_loop_drives_the_claude_driver_end_to_end(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path, initial_loss=1.0)
    spec = make_spec(budget=Budget(max_units=2, hard_kill_s=30))
    driver = ClaudeCodeDriver(spec, command=fake_claude(tmp_path, FAKE_CLAUDE_HALVE), poll=0.02, timeout_s=30)

    result = await run(spec, driver=driver, repo=repo)

    rows = journal_rows(repo)
    assert [row.verdict for row in rows] == [Verdict.KEEP, Verdict.KEEP]
    assert result.kept == 2
    assert result.best is not None and result.best.metric == 0.25  # 1.0 -> 0.5 -> 0.25
    assert all(row.description.startswith("claude edited") for row in rows)
    assert git(repo, "rev-parse", f"athome/{EXPERIMENT_NAME}") == rows[1].commit


async def test_changed_files_lists_modified_and_untracked_sorted(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path)
    (repo / "train.py").write_text("LOSS = 0.1\n")
    (repo / "new.txt").write_text("x\n")

    assert await changed_files(repo) == ["new.txt", "train.py"]  # score.py, unchanged, is absent


async def test_reported_metric_reads_the_file_or_none(tmp_path: Path) -> None:
    driver = ClaudeCodeDriver(make_spec(budget=Budget(max_units=1)))

    assert await driver.reported_metric(tmp_path) is None
    (tmp_path / ".athome-metric.json").write_text(json.dumps({"loss": 0.42}))
    assert await driver.reported_metric(tmp_path) == 0.42


@pytest.mark.live
async def test_live_claude_proposes_a_real_edit(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path, initial_loss=1.0)
    spec = make_spec(budget=Budget(max_units=1, hard_kill_s=120))

    await run(spec, driver=ClaudeCodeDriver(spec, poll=2.0, timeout_s=600), repo=repo)

    (row,) = journal_rows(repo)
    assert row.description.startswith("claude edited")
