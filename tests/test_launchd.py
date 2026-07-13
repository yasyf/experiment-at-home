from __future__ import annotations

import os
import plistlib
import subprocess
from dataclasses import astuple
from typing import TYPE_CHECKING

import pytest

from athome import launchd
from athome.config import AthomeSettings, load
from athome.launchd import (
    AgentSpec,
    AgentStatus,
    Calendar,
    Interval,
    KeepAlive,
    LaunchdError,
    installed,
    plist_dict,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class FakeLaunchctl:
    def __init__(self, **overrides: tuple[int, bytes, bytes]) -> None:
        self.calls: list[list[str]] = []
        self.overrides = overrides

    async def __call__(
        self, command: Sequence[str], *, check: bool = True, **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        self.calls.append(list(command))
        returncode, stdout, stderr = self.overrides.get(command[1], (0, b"", b""))
        return subprocess.CompletedProcess(list(command), returncode, stdout, stderr)


@pytest.fixture
def logs_root() -> Path:
    return load(AthomeSettings).logs_root


def base_spec(schedule: object) -> AgentSpec:
    return AgentSpec(label="com.athome.demo", command=("athome", "batch", "collect"), schedule=schedule)


@pytest.mark.parametrize(
    ("schedule", "expected_key"),
    [
        pytest.param(Calendar(hour=3, minute=0), {"StartCalendarInterval": {"Hour": 3, "Minute": 0}}, id="calendar"),
        pytest.param(
            Calendar(hour=4, minute=30, weekday=0),
            {"StartCalendarInterval": {"Weekday": 0, "Hour": 4, "Minute": 30}},
            id="calendar-weekday",
        ),
        pytest.param(Interval(seconds=3600), {"StartInterval": 3600}, id="interval"),
        pytest.param(KeepAlive(), {"KeepAlive": True, "RunAtLoad": True}, id="keepalive"),
    ],
)
def test_plist_dict_schedule_shapes(schedule: object, expected_key: dict[str, object], logs_root: Path) -> None:
    assert (
        plist_dict(base_spec(schedule))
        == {
            "Label": "com.athome.demo",
            "ProgramArguments": ["/bin/sh", "-lc", "exec athome batch collect"],
            "StandardOutPath": str(logs_root / "com.athome.demo.log"),
            "StandardErrorPath": str(logs_root / "com.athome.demo.log"),
        }
        | expected_key
    )


def test_plist_dict_env_prefix_present(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text('env_prefix_cmd = "eval \\"$(ccp env)\\""\n')
    load.cache_clear()
    assert plist_dict(base_spec(KeepAlive()))["ProgramArguments"] == [
        "/bin/sh",
        "-lc",
        'eval "$(ccp env)"; exec athome batch collect',
    ]


def test_plist_dict_env_prefix_absent() -> None:
    assert plist_dict(base_spec(KeepAlive()))["ProgramArguments"] == [
        "/bin/sh",
        "-lc",
        "exec athome batch collect",
    ]


def test_plist_dict_log_name_override(logs_root: Path) -> None:
    spec = AgentSpec(label="com.athome.demo", command=("athome",), schedule=KeepAlive(), log_name="collector")
    assert plist_dict(spec)["StandardOutPath"] == str(logs_root / "collector.log")
    assert plist_dict(spec)["StandardErrorPath"] == str(logs_root / "collector.log")


def test_plist_dict_working_dir_and_env(tmp_path: Path) -> None:
    spec = AgentSpec(
        label="com.athome.demo",
        command=("athome",),
        schedule=KeepAlive(),
        working_dir=tmp_path / "work",
        env=(("PATH", "/usr/bin"), ("TZ", "UTC")),
    )
    rendered = plist_dict(spec)
    assert rendered["WorkingDirectory"] == str(tmp_path / "work")
    assert rendered["EnvironmentVariables"] == {"PATH": "/usr/bin", "TZ": "UTC"}


async def test_install_writes_plist_and_bootstraps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launchd, "LAUNCH_AGENTS", tmp_path / "LaunchAgents")
    fake = FakeLaunchctl()
    monkeypatch.setattr("anyio.run_process", fake)
    spec = base_spec(Calendar(hour=3, minute=0))

    path = await launchd.install(spec)

    assert path == tmp_path / "LaunchAgents" / "com.athome.demo.plist"
    assert plistlib.loads(path.read_bytes()) == plist_dict(spec)
    domain = f"gui/{os.getuid()}"
    assert fake.calls == [
        ["launchctl", "bootout", domain, str(path)],
        ["launchctl", "bootstrap", domain, str(path)],
    ]


async def test_install_raises_on_bootstrap_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launchd, "LAUNCH_AGENTS", tmp_path / "LaunchAgents")
    fake = FakeLaunchctl(bootstrap=(5, b"", b"Bootstrap failed: 5: Input/output error"))
    monkeypatch.setattr("anyio.run_process", fake)

    with pytest.raises(LaunchdError, match="Input/output error"):
        await launchd.install(base_spec(KeepAlive()))


async def test_uninstall_boots_out_and_unlinks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launchd, "LAUNCH_AGENTS", tmp_path / "LaunchAgents")
    fake = FakeLaunchctl(print=(113, b"", b"Could not find service"))
    monkeypatch.setattr("anyio.run_process", fake)
    path = await launchd.install(base_spec(KeepAlive()))
    fake.calls.clear()

    await launchd.uninstall("com.athome.demo")

    assert not path.exists()
    assert fake.calls == [
        ["launchctl", "bootout", f"gui/{os.getuid()}", str(path)],
        ["launchctl", "print", f"gui/{os.getuid()}/com.athome.demo"],
    ]


async def test_status_running(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launchd, "LAUNCH_AGENTS", tmp_path / "LaunchAgents")
    (tmp_path / "LaunchAgents").mkdir()
    launchd.agent_path("com.athome.demo").write_bytes(b"<plist/>")
    stdout = b"com.athome.demo = {\n\tstate = running\n\tpid = 4242\n\tlast exit code = 0\n}\n"
    monkeypatch.setattr("anyio.run_process", FakeLaunchctl(print=(0, stdout, b"")))

    assert astuple(await launchd.status("com.athome.demo")) == ("com.athome.demo", True, True, 4242, 0)


async def test_status_not_loaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launchd, "LAUNCH_AGENTS", tmp_path / "LaunchAgents")
    fake = FakeLaunchctl(print=(113, b"", b"Could not find service"))
    monkeypatch.setattr("anyio.run_process", fake)

    assert await launchd.status("com.athome.demo") == AgentStatus(
        label="com.athome.demo", installed=False, running=False, pid=None, last_exit=None
    )
    assert fake.calls == [["launchctl", "print", f"gui/{os.getuid()}/com.athome.demo"]]


def test_agent_spec_rejects_path_label() -> None:
    with pytest.raises(LaunchdError):
        AgentSpec(label="/tmp/victim", command=("athome",), schedule=KeepAlive())


async def test_uninstall_rejects_path_label(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launchd, "LAUNCH_AGENTS", tmp_path / "LaunchAgents")
    victim = tmp_path / "victim.plist"
    victim.write_bytes(b"do not touch")
    fake = FakeLaunchctl()
    monkeypatch.setattr("anyio.run_process", fake)

    with pytest.raises(LaunchdError):
        await launchd.uninstall(str(tmp_path / "victim"))

    assert victim.exists()
    assert fake.calls == []


async def test_uninstall_raises_when_still_loaded_after_bootout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(launchd, "LAUNCH_AGENTS", tmp_path / "LaunchAgents")
    fake = FakeLaunchctl()
    monkeypatch.setattr("anyio.run_process", fake)
    path = await launchd.install(base_spec(KeepAlive()))
    fake.overrides["bootout"] = (5, b"", b"Boot-out failed: 5: Input/output error")
    fake.overrides["print"] = (0, b"com.athome.demo = {\n\tstate = running\n\tpid = 4242\n}\n", b"")
    fake.calls.clear()

    with pytest.raises(LaunchdError, match="still loaded"):
        await launchd.uninstall("com.athome.demo")

    assert path.exists()
    assert fake.calls == [
        ["launchctl", "bootout", f"gui/{os.getuid()}", str(path)],
        ["launchctl", "print", f"gui/{os.getuid()}/com.athome.demo"],
    ]


@pytest.mark.parametrize("bootout_code", [3, 5, 113])
async def test_uninstall_unlinks_when_not_loaded_despite_bootout_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bootout_code: int
) -> None:
    monkeypatch.setattr(launchd, "LAUNCH_AGENTS", tmp_path / "LaunchAgents")
    fake = FakeLaunchctl(print=(113, b"", b"Could not find service"))
    monkeypatch.setattr("anyio.run_process", fake)
    path = await launchd.install(base_spec(KeepAlive()))
    fake.overrides["bootout"] = (bootout_code, b"", f"Boot-out failed: {bootout_code}".encode())
    fake.calls.clear()

    await launchd.uninstall("com.athome.demo")

    assert not path.exists()
    assert fake.calls == [
        ["launchctl", "bootout", f"gui/{os.getuid()}", str(path)],
        ["launchctl", "print", f"gui/{os.getuid()}/com.athome.demo"],
    ]


def test_installed_scans_namespace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agents = tmp_path / "LaunchAgents"
    agents.mkdir()
    monkeypatch.setattr(launchd, "LAUNCH_AGENTS", agents)
    for name in ("com.athome.watch.plist", "com.athome.batch-collect.plist", "com.other.thing.plist"):
        (agents / name).write_bytes(b"")

    assert installed() == ["com.athome.batch-collect", "com.athome.watch"]
    assert installed(prefix="batch") == ["com.athome.batch-collect"]
