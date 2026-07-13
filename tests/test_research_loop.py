from __future__ import annotations

import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

from athome.research.contract import build_contract
from athome.research.driver import StubDriver, StubProposal
from athome.research.journal import Journal, JournalRow, Verdict
from athome.research.loop import run, run_metric
from athome.research.spec import Budget, ExperimentSpec

if TYPE_CHECKING:
    from pathlib import Path

EXPERIMENT_NAME = "toy"

# Immutable evaluator: computes the metric from the mutable train.py, writes it to the
# structured JSON channel, and prints a LYING value to stdout. A loop that grepped
# stdout would score 999.0; a correct loop reads the file.
SCORE_PY = textwrap.dedent(
    """
    import json, pathlib
    namespace = {}
    exec(pathlib.Path("train.py").read_text(), namespace)
    pathlib.Path(".athome-metric.json").write_text(json.dumps({"loss": namespace["LOSS"]}))
    print("loss=999.0")
    """
).strip()


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


def journal_rows(repo: Path) -> list[JournalRow]:
    return Journal.open(repo / ".git" / "athome" / f"{EXPERIMENT_NAME}.jsonl").rows()


def loss_proposal(loss: float | str, description: str = "edit") -> StubProposal:
    return StubProposal({"train.py": f"LOSS = {loss}\n"}, description)


async def test_end_to_end_all_gates_in_one_run(tmp_path: Path) -> None:
    # One 3-unit run exercising every loop invariant at once: keep, immutable-discard, keep.
    repo = toy_repo(tmp_path, initial_loss=1.0)
    proposals = iter(
        [
            loss_proposal(0.5, "half"),
            StubProposal({"score.py": "HACKED = 1\n"}, "tamper"),
            loss_proposal(0.3, "third"),
            loss_proposal(0.1, "unreached"),
        ]
    )
    result = await run(make_spec(budget=Budget(max_units=3)), driver=StubDriver(proposals), repo=repo)

    rows = journal_rows(repo)
    assert [row.unit for row in rows] == [0, 1, 2]
    assert [row.verdict for row in rows] == [Verdict.KEEP, Verdict.DISCARD, Verdict.KEEP]
    assert rows[0].description == "half" and rows[2].description == "third"
    assert result.kept == 2

    # The immutable-path mutation was rejected structurally and never scored.
    assert rows[1].metric is None
    assert rows[1].description.startswith("tamper") and "ImmutableViolation" in rows[1].description
    assert "score.py" in rows[1].description
    assert "HACKED" not in git(repo, "show", f"athome/{EXPERIMENT_NAME}:score.py")

    # The metric came from the JSON file, not the 999.0 score.py prints to stdout.
    assert [rows[0].metric, rows[2].metric] == [0.5, 0.3]
    assert result.best is not None and result.best.metric == 0.3

    # The two KEEPs chain off the base; the DISCARD kept the incumbent commit.
    tip = git(repo, "rev-parse", f"athome/{EXPERIMENT_NAME}")
    assert tip == rows[2].commit
    assert git(repo, "rev-parse", f"{rows[2].commit}^") == rows[0].commit
    assert rows[1].commit == rows[0].commit

    # Budget-cap stop: exactly 3 units ran; the 4th proposal was never consumed.
    assert next(proposals).description == "unreached"


async def test_three_units_keep_keep_discard(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path, initial_loss=1.0)
    driver = StubDriver(iter([loss_proposal(0.5, "half"), loss_proposal(0.3, "third"), loss_proposal(0.4, "worse")]))

    result = await run(make_spec(budget=Budget(max_units=3)), driver=driver, repo=repo)

    rows = journal_rows(repo)
    assert [row.verdict for row in rows] == [Verdict.KEEP, Verdict.KEEP, Verdict.DISCARD]
    assert [row.metric for row in rows] == [0.5, 0.3, 0.4]
    assert [row.description for row in rows] == ["half", "third", "worse"]
    assert result.kept == 2
    assert result.best is not None and result.best.metric == 0.3

    # The experiment branch advanced to the last KEEP and stopped there — the discard
    # neither advanced it nor rewound it.
    tip = git(repo, "rev-parse", f"athome/{EXPERIMENT_NAME}")
    assert tip == rows[1].commit
    assert rows[2].commit != tip
    # KEEP builds a linear chain off the incumbent.
    assert git(repo, "rev-parse", f"{rows[1].commit}^") == rows[0].commit


async def test_immutable_mutation_is_rejected_structurally(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path)
    driver = StubDriver(iter([StubProposal({"score.py": "HACKED = 1\n"}, "tamper")]))

    result = await run(make_spec(budget=Budget(max_units=1)), driver=driver, repo=repo)

    (row,) = journal_rows(repo)
    assert row.verdict is Verdict.DISCARD
    assert row.metric is None  # the metric command never ran
    assert "ImmutableViolation" in row.description and "score.py" in row.description
    assert result.kept == 0
    # The scoring boundary held: the experiment branch never left the base commit.
    assert git(repo, "rev-parse", f"athome/{EXPERIMENT_NAME}") == git(repo, "rev-parse", "HEAD")


async def test_metric_is_read_from_the_file_not_stdout(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path, initial_loss=1.0)
    driver = StubDriver(iter([loss_proposal(0.25)]))

    result = await run(make_spec(budget=Budget(max_units=1)), driver=driver, repo=repo)

    assert result.best is not None
    assert result.best.metric == 0.25  # the file value, not the 999.0 printed to stdout
    assert journal_rows(repo)[0].metric == 0.25


async def test_budget_cap_stops_the_loop(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path)
    proposals = iter([loss_proposal(0.5), loss_proposal(0.4), loss_proposal(0.3, "never")])
    driver = StubDriver(proposals)

    await run(make_spec(budget=Budget(max_units=2)), driver=driver, repo=repo)

    rows = journal_rows(repo)
    assert [row.unit for row in rows] == [0, 1]  # stopped at max_units, unit 2 never ran
    assert next(proposals).description == "never"  # the third proposal was never consumed


async def test_resume_skips_completed_units(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path)
    await run(make_spec(budget=Budget(max_units=1)), driver=StubDriver(iter([loss_proposal(0.5)])), repo=repo)

    # A second run with a larger budget resumes: only unit 1 runs, off the kept incumbent.
    result = await run(make_spec(budget=Budget(max_units=2)), driver=StubDriver(iter([loss_proposal(0.3)])), repo=repo)

    rows = journal_rows(repo)
    assert [row.unit for row in rows] == [0, 1]
    assert [row.verdict for row in rows] == [Verdict.KEEP, Verdict.KEEP]
    assert result.kept == 2
    assert result.best is not None and result.best.metric == 0.3


async def test_broken_candidate_is_journaled_as_crash(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path)
    # Unit 0 keeps a valid model; unit 1's edit breaks the evaluator (NameError -> nonzero exit).
    driver = StubDriver(iter([loss_proposal(0.4, "good"), loss_proposal("undefined_symbol", "broken")]))

    result = await run(make_spec(budget=Budget(max_units=2)), driver=driver, repo=repo)

    rows = journal_rows(repo)
    assert [row.verdict for row in rows] == [Verdict.KEEP, Verdict.CRASH]
    assert rows[1].metric is None
    # A crash never advances the incumbent.
    assert result.best is not None and result.best.metric == 0.4
    assert git(repo, "rev-parse", f"athome/{EXPERIMENT_NAME}") == rows[0].commit


async def test_run_metric_hard_kill_times_out(tmp_path: Path) -> None:
    returncode, log = await run_metric((sys.executable, "-c", "import time; time.sleep(30)"), tmp_path, hard_kill_s=0.1)
    assert returncode is None and log == b""


async def test_run_metric_captures_combined_output(tmp_path: Path) -> None:
    returncode, log = await run_metric(
        (sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"),
        tmp_path,
        hard_kill_s=None,
    )
    assert returncode == 0
    assert b"out" in log and b"err" in log  # stderr folded into the captured (untrusted) log


def test_contract_states_manifest_metric_and_keep_rule() -> None:
    contract = build_contract(make_spec(budget=Budget(max_units=3)), budget_low=False)
    assert "minimize" in contract and "`loss`" in contract
    assert "- `train.py`" in contract and "- `score.py`" in contract
    assert ".athome-metric.json" in contract
    assert "must fall strictly below" in contract
    assert "Simplicity" in contract
    assert "Budget is nearly exhausted" not in contract


def test_contract_maximize_wording_and_budget_low_warning() -> None:
    contract = build_contract(make_spec(budget=Budget(max_units=1), direction="max"), budget_low=True)
    assert "maximize" in contract and "must strictly exceed" in contract
    assert "Budget is nearly exhausted" in contract
