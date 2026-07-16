from __future__ import annotations

import os
import plistlib
import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import click

from athome.cli import coro, emit, json_option
from athome.config import AthomeSettings, load
from athome.errors import AthomeError

if TYPE_CHECKING:
    import subprocess

LAUNCH_AGENTS = Path.home() / "Library" / "LaunchAgents"
LABEL_NAMESPACE = "com.athome."
LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]*")
PID_RE = re.compile(r"\bpid = (\d+)")
EXIT_RE = re.compile(r"last exit code = (\d+)")


class LaunchdError(AthomeError):
    """Raised when a launchctl invocation fails — most often a non-zero bootstrap."""


@dataclass(frozen=True, slots=True)
class Calendar:
    """A wall-clock trigger: every day at ``hour``:``minute``, or a single ``weekday`` (0=Sunday)."""

    hour: int
    minute: int
    weekday: int | None = None


@dataclass(frozen=True, slots=True)
class Interval:
    """A relative trigger: fire every ``seconds`` since the last launch."""

    seconds: int


@dataclass(frozen=True, slots=True)
class KeepAlive:
    """An always-on daemon: launchd respawns it whenever it exits, and runs it at load."""


type Schedule = Calendar | Interval | KeepAlive


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """A declarative launchd LaunchAgent: what to run, when, and where its logs go.

    Example:
        >>> spec = AgentSpec(
        ...     label="com.athome.batch-collect",
        ...     command=("athome", "batch", "collect"),
        ...     schedule=Calendar(hour=3, minute=0),
        ... )
        >>> path = await install(spec)
    """

    label: str
    command: tuple[str, ...]
    schedule: Schedule
    log_name: str | None = None
    log_dir: Path | None = None
    working_dir: Path | None = None
    env: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        validate_label(self.label)


@dataclass(frozen=True, slots=True)
class AgentStatus:
    """The live state of one agent: whether its plist is installed and its bootstrapped process running."""

    label: str
    installed: bool
    running: bool
    pid: int | None
    last_exit: int | None


def validate_label(label: str) -> str:
    if not LABEL_RE.fullmatch(label):
        raise LaunchdError(f"invalid launchd label {label!r}: expected reverse-DNS, no slashes or '..'")
    return label


def agent_path(label: str) -> Path:
    return LAUNCH_AGENTS / f"{validate_label(label)}.plist"


def wrapper_command(command: tuple[str, ...]) -> str:
    form = f"exec {shlex.join(command)}"
    return f"{prefix}; {form}" if (prefix := load(AthomeSettings).env_prefix_cmd) else form


def schedule_keys(schedule: Schedule) -> dict[str, object]:
    match schedule:
        case Calendar(hour=hour, minute=minute, weekday=None):
            return {"StartCalendarInterval": {"Hour": hour, "Minute": minute}}
        case Calendar(hour=hour, minute=minute, weekday=weekday):
            return {"StartCalendarInterval": {"Weekday": weekday, "Hour": hour, "Minute": minute}}
        case Interval(seconds=seconds):
            return {"StartInterval": seconds}
        case KeepAlive():
            return {"KeepAlive": True, "RunAtLoad": True}


def plist_dict(spec: AgentSpec) -> dict[str, object]:
    """Render ``spec`` to the launchd plist dictionary launchctl loads."""
    log = (spec.log_dir or load(AthomeSettings).logs_root) / f"{spec.log_name or spec.label}.log"
    return (
        {
            "Label": spec.label,
            "ProgramArguments": ["/bin/sh", "-lc", wrapper_command(spec.command)],
            "StandardOutPath": str(log),
            "StandardErrorPath": str(log),
        }
        | schedule_keys(spec.schedule)
        | ({"WorkingDirectory": str(spec.working_dir)} if spec.working_dir else {})
        | ({"EnvironmentVariables": dict(spec.env)} if spec.env else {})
    )


async def launchctl_print(label: str) -> subprocess.CompletedProcess[bytes]:
    return await anyio.run_process(["launchctl", "print", f"gui/{os.getuid()}/{label}"], check=False)


async def install(spec: AgentSpec) -> Path:
    """Write ``spec``'s plist and (re)load it into the user's GUI domain; return the plist path.

    Raises:
        LaunchdError: ``launchctl bootstrap`` exited non-zero (stderr attached).
    """
    path = agent_path(spec.label)
    await anyio.Path(spec.log_dir or load(AthomeSettings).logs_root).mkdir(parents=True, exist_ok=True)
    await anyio.Path(path.parent).mkdir(parents=True, exist_ok=True)
    await anyio.Path(path).write_bytes(plistlib.dumps(plist_dict(spec)))
    domain = f"gui/{os.getuid()}"
    await anyio.run_process(["launchctl", "bootout", domain, str(path)], check=False)
    if (result := await anyio.run_process(["launchctl", "bootstrap", domain, str(path)], check=False)).returncode:
        raise LaunchdError(f"launchctl bootstrap failed for {spec.label}: {result.stderr.decode().strip()}")
    return path


async def uninstall(label: str) -> None:
    """Boot the agent out of the GUI domain and remove its plist.

    Raises:
        LaunchdError: the agent is still loaded after ``launchctl bootout``, so its
            plist is kept rather than orphaning a live agent with no on-disk definition.
    """
    path = agent_path(label)
    # bootout's exit code is not portable: an already-unloaded agent returns 5
    # (Input/output error) on a real host, indistinguishable by code from a genuine
    # failure. So we ignore the code and trust the observable post-state — if
    # `launchctl print` still resolves the service, the agent is still loaded.
    result = await anyio.run_process(["launchctl", "bootout", f"gui/{os.getuid()}", str(path)], check=False)
    if not (await launchctl_print(label)).returncode:
        raise LaunchdError(f"{label} still loaded after launchctl bootout: {result.stderr.decode().strip()}")
    await anyio.Path(path).unlink()


async def status(label: str) -> AgentStatus:
    """Report ``label``'s installed/running state by parsing ``launchctl print gui/<uid>/<label>``."""
    result = await launchctl_print(label)
    if result.returncode:
        return AgentStatus(label, agent_path(label).exists(), running=False, pid=None, last_exit=None)
    text = result.stdout.decode()
    pid = int(match.group(1)) if (match := PID_RE.search(text)) else None
    last_exit = int(match.group(1)) if (match := EXIT_RE.search(text)) else None
    return AgentStatus(label, agent_path(label).exists(), running=pid is not None, pid=pid, last_exit=last_exit)


def installed(*, prefix: str = "") -> list[str]:
    """List the labels of every athome-written LaunchAgent, optionally narrowed to those under ``prefix``."""
    return sorted(path.stem for path in LAUNCH_AGENTS.glob(f"{LABEL_NAMESPACE}{prefix}*.plist"))


@click.group("launchd")
def cli() -> None:
    """Inspect installed athome launchd agents."""


@cli.command("list")
@json_option
def list_command(*, as_json: bool) -> None:
    """List every installed athome agent by label."""
    emit(installed(), as_json=as_json)


@cli.command("status")
@click.argument("label")
@json_option
@coro
async def status_command(label: str, *, as_json: bool) -> None:
    """Print LABEL's installed and running state."""
    emit(asdict(await status(label)), as_json=as_json)


@cli.command("uninstall")
@click.argument("label")
@json_option
@coro
async def uninstall_command(label: str, *, as_json: bool) -> None:
    """Boot LABEL out of launchd and delete its plist."""
    await uninstall(label)
    emit({"uninstalled": label}, as_json=as_json)
