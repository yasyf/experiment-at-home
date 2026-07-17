from __future__ import annotations

import fcntl
import json
import os
import textwrap
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import pytest
from click.testing import CliRunner

from athome import launchd
from athome.config import AthomeSettings, load
from athome.research import nightly, watchdog
from athome.research.cli import cli as research_cli
from athome.research.spec import Budget, ExperimentSpec

if TYPE_CHECKING:
    from collections.abc import Iterator

SPEC_TOML = textwrap.dedent(
    """\
    name = "toy"
    metric_command = ["python", "score.py"]
    metric_key = "loss"
    direction = "min"
    mutable_paths = ["train.py"]
    immutable_paths = ["score.py"]

    [budget]
    max_units = 1
    """
)


def make_spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="toy",
        metric_command=("python", "score.py"),
        metric_key="loss",
        direction="min",
        mutable_paths=("train.py",),
        immutable_paths=("score.py",),
        budget=Budget(max_units=1),
    )


def write_spec(root: Path) -> Path:
    path = root / "experiment.toml"
    path.write_text(SPEC_TOML)
    return path


@contextmanager
def held_exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_probe_live_distinguishes_an_exclusive_holder(tmp_path: Path) -> None:
    lock = tmp_path / "toy.lock"

    assert watchdog.probe_live(lock) is False
    with held_exclusive_lock(lock):
        assert watchdog.probe_live(lock) is True
    assert watchdog.probe_live(lock) is False


async def test_observe_progress_tracks_growth_static_truncation_and_missing_paths(tmp_path: Path) -> None:
    watched = tmp_path / "journal.jsonl"
    watched.write_bytes(b"abc")

    first = await watchdog.observe_progress(watched, now=100.0)
    static = await watchdog.observe_progress(watched, now=200.0)
    watched.write_bytes(b"abcd")
    grown = await watchdog.observe_progress(watched, now=300.0)
    watched.write_bytes(b"x")
    shrunk = await watchdog.observe_progress(watched, now=400.0)
    missing = await watchdog.observe_progress(tmp_path / "missing.jsonl", now=500.0)

    assert (first.offset, first.last_growth_ts, first.checked_at) == (3, 100.0, 100.0)
    assert (static.offset, static.last_growth_ts, static.checked_at) == (3, 100.0, 200.0)
    assert (grown.offset, grown.last_growth_ts, grown.checked_at) == (4, 300.0, 300.0)
    assert (shrunk.offset, shrunk.last_growth_ts, shrunk.checked_at) == (1, 300.0, 400.0)
    assert (missing.offset, missing.last_growth_ts, missing.checked_at, missing.device, missing.inode) == (
        0,
        500.0,
        500.0,
        None,
        None,
    )
    assert json.loads(Path(f"{watched}.watch-state.json").read_text())["offset"] == 1


async def test_live_static_run_trips_and_alerts_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    journal = tmp_path / "athome" / "toy.jsonl"
    commands: list[list[str]] = []

    async def fake_journal_path(_repo: Path, name: str) -> Path:
        assert name == "toy"
        return journal

    async def fake_run_process(command: list[str], *args: object, **kwargs: object) -> None:
        commands.append(command)

    monkeypatch.setattr(nightly, "journal_path", fake_journal_path)
    monkeypatch.setattr(anyio, "run_process", fake_run_process)

    with held_exclusive_lock(journal.with_suffix(".lock")):
        first = await watchdog.check(make_spec(), repo=tmp_path, quiet_s=5400.0, now=lambda: 0.0)
        alarm = await watchdog.check(make_spec(), repo=tmp_path, quiet_s=5400.0, now=lambda: 5400.0)

    detail = "no journal or launchd log growth for 5400s while the experiment lock is held"
    lines = journal.with_name("toy.events.jsonl").read_text().splitlines()
    assert first == watchdog.WatchResult(live=True, alarm=False)
    assert alarm == watchdog.WatchResult(live=True, alarm=True)
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"type": "quiet_alarm", "unit": "toy", "detail": detail}
    assert commands == [
        [
            "/opt/homebrew/bin/cc-notes",
            "note",
            "add",
            "athome quiet alarm [toy]",
            "--body",
            detail,
            "--label",
            "athome-research-watchdog",
        ]
    ]


async def test_log_growth_resets_the_live_quiet_timer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    journal = tmp_path / "athome" / "toy.jsonl"

    async def fake_journal_path(_repo: Path, _name: str) -> Path:
        return journal

    monkeypatch.setattr(nightly, "journal_path", fake_journal_path)

    with held_exclusive_lock(journal.with_suffix(".lock")):
        initial = await watchdog.check(make_spec(), repo=tmp_path, now=lambda: 0.0)
        log = load(AthomeSettings).logs_root / f"{nightly.RESEARCH_LABEL_PREFIX}toy.log"
        log.write_text("still working\n")
        after_growth = await watchdog.check(make_spec(), repo=tmp_path, now=lambda: 5400.0)
        before_deadline = await watchdog.check(make_spec(), repo=tmp_path, now=lambda: 10799.0)

    assert initial == watchdog.WatchResult(live=True, alarm=False)
    assert after_growth == watchdog.WatchResult(live=True, alarm=False)
    assert before_deadline == watchdog.WatchResult(live=True, alarm=False)


async def test_not_live_never_trips_and_a_new_live_episode_starts_fresh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    journal = tmp_path / "athome" / "toy.jsonl"

    async def fake_journal_path(_repo: Path, _name: str) -> Path:
        return journal

    monkeypatch.setattr(nightly, "journal_path", fake_journal_path)

    first = await watchdog.check(make_spec(), repo=tmp_path, now=lambda: 0.0)
    stale = await watchdog.check(make_spec(), repo=tmp_path, now=lambda: 50_000.0)
    with held_exclusive_lock(journal.with_suffix(".lock")):
        new_run = await watchdog.check(make_spec(), repo=tmp_path, now=lambda: 50_001.0)

    assert first == watchdog.WatchResult(live=False, alarm=False)
    assert stale == watchdog.WatchResult(live=False, alarm=False)
    assert new_run == watchdog.WatchResult(live=True, alarm=False)


async def test_install_wires_the_interval_agent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spec_path = write_spec(tmp_path)
    repo = tmp_path / "repo"
    captured: dict[str, launchd.AgentSpec] = {}

    async def fake_repo_root(_spec_path: Path) -> Path:
        return repo

    async def fake_install(agent: launchd.AgentSpec) -> Path:
        captured["agent"] = agent
        return tmp_path / "watch.plist"

    monkeypatch.setattr(nightly, "repo_root", fake_repo_root)
    monkeypatch.setattr(launchd, "install", fake_install)

    path = await watchdog.install(spec_path)

    assert captured["agent"] == launchd.AgentSpec(
        label="com.athome.research.watch.toy",
        command=("athome", "research", "watch", str(spec_path.resolve())),
        schedule=launchd.Interval(seconds=600),
        working_dir=repo,
    )
    assert path == tmp_path / "watch.plist"


@pytest.mark.parametrize(("alarm", "exit_code"), [(False, 0), (True, 1)])
def test_cli_watch_exit_code_follows_the_alarm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, alarm: bool, exit_code: int
) -> None:
    spec_path = write_spec(tmp_path)

    async def fake_check(spec: ExperimentSpec, *, repo: Path) -> watchdog.WatchResult:
        assert spec.name == "toy"
        assert repo == tmp_path.resolve()
        return watchdog.WatchResult(live=alarm, alarm=alarm)

    monkeypatch.setattr(watchdog, "check", fake_check)

    result = CliRunner().invoke(
        research_cli,
        ["watch", str(spec_path), "--repo", str(tmp_path), "--json"],
    )

    assert result.exit_code == exit_code, result.output
    assert json.loads(result.output) == {"live": alarm, "alarm": alarm}


def test_cli_install_watch_is_wired(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spec_path = write_spec(tmp_path)
    captured: dict[str, launchd.AgentSpec] = {}

    async def fake_repo_root(_spec_path: Path) -> Path:
        return tmp_path

    async def fake_install(agent: launchd.AgentSpec) -> Path:
        captured["agent"] = agent
        return tmp_path / f"{agent.label}.plist"

    monkeypatch.setattr(nightly, "repo_root", fake_repo_root)
    monkeypatch.setattr(launchd, "install", fake_install)

    result = CliRunner().invoke(research_cli, ["nightly", "install-watch", str(spec_path), "--json"])

    assert result.exit_code == 0, result.output
    assert captured["agent"].schedule == launchd.Interval(seconds=600)
    assert json.loads(result.output) == {"installed": str(tmp_path / "com.athome.research.watch.toy.plist")}
