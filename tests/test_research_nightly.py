from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
from click.testing import CliRunner

from athome import launchd
from athome.research import nightly
from athome.research.cli import cli as research_cli
from athome.research.driver import StubDriver, StubProposal
from athome.research.loop import run
from athome.research.spec import Budget, ExperimentSpec

if TYPE_CHECKING:
    import pytest

EXPERIMENT_NAME = "toy"

SCORE_PY = textwrap.dedent(
    """
    import json, pathlib
    namespace = {}
    exec(pathlib.Path("train.py").read_text(), namespace)
    pathlib.Path(".athome-metric.json").write_text(json.dumps({"loss": namespace["LOSS"]}))
    """
).strip()

SPEC_TOML = textwrap.dedent(
    """
    name = "toy"
    metric_command = ["python", "score.py"]
    metric_key = "loss"
    direction = "min"
    mutable_paths = ["train.py"]
    immutable_paths = ["score.py"]

    [budget]
    max_units = 3
    """
).strip()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def toy_repo(root: Path) -> Path:
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "toy@localhost")
    git(root, "config", "user.name", "toy")
    (root / "train.py").write_text("LOSS = 1.0\n")
    (root / "score.py").write_text(SCORE_PY + "\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    return root


def write_spec(repo: Path) -> Path:
    (spec_path := repo / "experiment.toml").write_text(SPEC_TOML + "\n")
    return spec_path


def make_spec(budget: Budget) -> ExperimentSpec:
    return ExperimentSpec(
        name=EXPERIMENT_NAME,
        metric_command=(sys.executable, "score.py"),
        metric_key="loss",
        direction="min",
        mutable_paths=("train.py",),
        immutable_paths=("score.py",),
        budget=budget,
    )


async def drive(repo: Path, *proposals: StubProposal) -> ExperimentSpec:
    spec = make_spec(Budget(max_units=len(proposals)))
    await run(spec, driver=StubDriver(iter(proposals)), repo=repo)
    return spec


async def test_journal_path_lives_under_the_git_common_dir(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path)
    assert await nightly.journal_path(repo, "toy") == repo / ".git" / "athome" / "toy.jsonl"


async def test_repo_root_resolves_the_toplevel(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path)
    assert await nightly.repo_root(repo / "experiment.toml") == Path(git(repo, "rev-parse", "--show-toplevel"))


async def test_report_summarizes_the_journal(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path)
    spec = await drive(
        repo,
        StubProposal({"train.py": "LOSS = 0.5\n"}),
        StubProposal({"train.py": "LOSS = 0.6\n"}),
        StubProposal({"train.py": "LOSS = undefined_symbol\n"}),
    )

    report = await nightly.report(spec, repo=repo)

    assert (report.experiment, report.units, report.kept, report.crashes) == ("toy", 3, 1, 1)
    assert report.best is not None and report.best.metric == 0.5
    assert len(report.rows) == 3
    assert report.infra_retries == 0  # no infra sidecar for this clean run


async def test_report_counts_infra_retries_from_the_sidecar(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path)
    spec = await drive(repo, StubProposal({"train.py": "LOSS = 0.5\n"}))
    events = repo / ".git" / "athome" / f"{EXPERIMENT_NAME}.events.jsonl"
    events.write_text(
        json.dumps({"unit": 1, "attempt": 0, "reason": "OSError('reset')"})
        + "\n"
        + json.dumps({"unit": 1, "attempt": 1, "reason": "OSError('reset')"})
        + "\n"
    )

    report = await nightly.report(spec, repo=repo)

    assert report.infra_retries == 2  # both sidecar retry records counted
    assert report.units == 1  # the journal itself is untouched by infra events


async def test_install_wires_the_agent_from_the_spec(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = toy_repo(tmp_path)
    spec_path = write_spec(repo)
    captured: dict[str, launchd.AgentSpec] = {}

    async def fake_install(agent: launchd.AgentSpec) -> Path:
        captured["agent"] = agent
        return tmp_path / "written.plist"

    monkeypatch.setattr(launchd, "install", fake_install)

    path = await nightly.install(spec_path)

    agent = captured["agent"]
    assert agent.label == "com.athome.research.toy"
    assert agent.command == ("athome", "research", "run", str(spec_path.resolve()))
    assert agent.working_dir == Path(git(repo, "rev-parse", "--show-toplevel"))
    assert agent.schedule == launchd.Calendar(hour=2, minute=0)
    assert path == tmp_path / "written.plist"


async def test_install_honors_a_custom_calendar(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = toy_repo(tmp_path)
    captured: dict[str, launchd.AgentSpec] = {}

    async def fake_install(agent: launchd.AgentSpec) -> Path:
        captured["agent"] = agent
        return tmp_path / "written.plist"

    monkeypatch.setattr(launchd, "install", fake_install)

    await nightly.install(write_spec(repo), calendar=launchd.Calendar(hour=5, minute=30))

    assert captured["agent"].schedule == launchd.Calendar(hour=5, minute=30)


def test_cli_init_scaffolds_a_loadable_spec(tmp_path: Path) -> None:
    result = CliRunner().invoke(research_cli, ["init", str(dest := tmp_path / "exp.toml"), "--name", "myexp"])

    assert result.exit_code == 0, result.output
    spec = ExperimentSpec.load(dest)
    assert spec.name == "myexp"
    assert spec.direction == "min"
    assert spec.budget.max_units == 20


def test_cli_nightly_install_is_wired(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spec_path = write_spec(toy_repo(tmp_path))

    async def fake_install(agent: launchd.AgentSpec) -> Path:
        return tmp_path / f"{agent.label}.plist"

    monkeypatch.setattr(launchd, "install", fake_install)

    result = CliRunner().invoke(
        research_cli, ["nightly", "install", str(spec_path), "--hour", "3", "--minute", "15", "--json"]
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["installed"].endswith("com.athome.research.toy.plist")


def test_cli_status_and_report_read_the_journal(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path)
    spec_path = write_spec(repo)
    anyio.run(
        drive,
        repo,
        StubProposal({"train.py": "LOSS = 0.5\n"}),
        StubProposal({"train.py": "LOSS = 0.6\n"}),
    )
    runner = CliRunner()

    status = json.loads(runner.invoke(research_cli, ["status", str(spec_path), "--repo", str(repo), "--json"]).output)
    assert (status["units"], status["kept"], status["crashes"]) == (2, 1, 0)
    assert status["best"]["metric"] == 0.5
    assert "rows" not in status

    report = json.loads(runner.invoke(research_cli, ["report", str(spec_path), "--repo", str(repo), "--json"]).output)
    assert len(report["rows"]) == 2
    assert [row["verdict"] for row in report["rows"]] == ["keep", "discard"]
