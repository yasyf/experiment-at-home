from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio
import click

from athome.cli import coro, emit, json_option
from athome.config import AthomeSettings, load
from athome.errors import AthomeError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

NAME_RE = re.compile(r"[A-Za-z0-9._-]+")


def run_log(name: str) -> Path:
    return load(AthomeSettings).logs_root / "runs" / f"{name}.log"


def run_pidfile(name: str) -> Path:
    return load(AthomeSettings).logs_root / "runs" / f"{name}.pid"


def run_exitfile(name: str) -> Path:
    return load(AthomeSettings).logs_root / "runs" / f"{name}.exit"


class DetachError(AthomeError):
    """Raised when a detached run cannot start — most often a live run already owns the name."""


@dataclass(frozen=True, slots=True)
class DetachedRun:
    """A launched detached run: its name, session-leader pid, and the log receiving its output.

    Example:
        >>> run = await launch(["make", "corpus"], name="overnight")
        >>> await wait(run.name)
        0
    """

    name: str
    pid: int
    log_path: Path


def running(name: str) -> int | None:
    """Return the pid of the live run named ``name`` (pid file + ``kill -0``), or None if none is alive."""
    pid_path = run_pidfile(name)
    if not pid_path.exists():
        return None
    pid = int(pid_path.read_text())
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    else:
        if reaped == pid:
            return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    return pid


async def launch(
    command: Sequence[str], *, name: str, on_spawn: Callable[[DetachedRun], None] | None = None
) -> DetachedRun:
    """Launch ``command`` as a detached, session-leader subprocess that outlives the caller.

    The command runs under ``/bin/sh -c`` with stdout and stderr appended to
    ``logs_root/runs/<name>.log``; once it exits the wrapper writes the real exit code to
    ``logs_root/runs/<name>.exit`` — a file separate from the shared log, so the detached
    child's stdout cannot forge it — and :func:`wait` recovers the code from there. The
    configured ``env_prefix_cmd`` is prepended like a launchd agent.

    Args:
        command: The argv of the program to run.
        name: A unique name for the run; also names its log, pid, and exit files.
        on_spawn: Synchronous callback invoked with the run before post-spawn I/O.

    Returns:
        The launched run's name, pid, and log path.

    Raises:
        DetachError: A live run already holds ``name``.
    """
    if not NAME_RE.fullmatch(name):
        raise DetachError(f"invalid run name {name!r}: must match [A-Za-z0-9._-]+")
    if (pid := running(name)) is not None:
        raise DetachError(f"a run named {name!r} is already live (pid {pid})")
    log_path, pid_path, exit_path = run_log(name), run_pidfile(name), run_exitfile(name)
    await anyio.Path(log_path.parent).mkdir(parents=True, exist_ok=True)
    await anyio.Path(exit_path).unlink(missing_ok=True)
    prefix = f"{env}; " if (env := load(AthomeSettings).env_prefix_cmd) else ""
    inner = f"{prefix}{shlex.join(command)}; rc=$?; echo $rc > {shlex.quote(str(exit_path))}"
    with log_path.open("ab") as log_file:
        with anyio.CancelScope(shield=True):
            process = await anyio.open_process(
                ["/bin/sh", "-c", inner],
                stdin=subprocess.DEVNULL,
                stdout=log_file.fileno(),
                stderr=log_file.fileno(),
                start_new_session=True,
            )
            run = DetachedRun(name=name, pid=process.pid, log_path=log_path)
            match on_spawn:
                case None:
                    pass
                case retain:
                    retain(run)
    await anyio.Path(pid_path).write_text(str(process.pid))
    return run


async def wait(name: str, *, poll: float = 5.0, timeout: float | None = None) -> int:
    """Poll until the run's process has exited and its ``.exit`` file appears, then return that code.

    Completion is proven by the process's real exit — its pid is gone, per
    :func:`running` — together with the wrapper-written ``<name>.exit`` file, never by a
    line the detached child could print to the shared log. A child that forges an
    ``ATHOME-RUN-DONE`` line on stdout, or writes its own ``.exit`` file while still alive,
    cannot make this return early.

    Args:
        name: The run to wait on.
        poll: Seconds to sleep between polls.
        timeout: Give up after this many seconds; ``None`` waits indefinitely.

    Returns:
        The command's exit code.

    Raises:
        TimeoutError: ``timeout`` elapsed before the run finished.
    """
    exit_path = anyio.Path(run_exitfile(name))
    deadline = None if timeout is None else anyio.current_time() + timeout
    while True:
        if running(name) is None and await exit_path.exists():
            return int((await exit_path.read_text()).strip())
        if deadline is None:
            await anyio.sleep(poll)
            continue
        if (remaining := deadline - anyio.current_time()) <= 0:
            raise TimeoutError(f"run {name!r} did not finish within {timeout}s")
        await anyio.sleep(min(poll, remaining))


class RunGroup(click.Group):
    """Routes a bare ``run --detach ... -- CMD`` invocation to the hidden launch command."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and args[0] not in ("--help", "-h"):
            args = ["launch", *args]
        return super().parse_args(ctx, args)


@click.group(
    cls=RunGroup,
    name="run",
    help="Launch and manage detached overnight runs (launch: run --detach --name NAME -- CMD...).",
)
def cli() -> None:
    pass


@cli.command("launch", hidden=True, context_settings={"ignore_unknown_options": True})
@click.option("--detach", is_flag=True, help="Run detached (the only mode in v0.1).")
@click.option("--name", required=True, help="Unique name for the run.")
@click.argument("command", nargs=-1, type=click.UNPROCESSED, required=True)
@json_option
@coro
async def launch_command(command: tuple[str, ...], name: str, detach: bool, as_json: bool) -> None:
    """Launch a detached run: run --detach --name NAME -- CMD..."""
    run = await launch(list(command), name=name)
    emit({"pid": run.pid, "log": run.log_path}, as_json=as_json)


@cli.command("wait")
@click.argument("name")
@click.option("--poll", default=5.0, show_default=True, help="Seconds between log polls.")
@click.option("--timeout", type=float, default=None, help="Give up after this many seconds.")
@json_option
@coro
async def wait_command(name: str, poll: float, timeout: float | None, as_json: bool) -> None:
    """Wait for run NAME to finish and print its exit code."""
    emit({"name": name, "exit": await wait(name, poll=poll, timeout=timeout)}, as_json=as_json)


@cli.command("log")
@click.argument("name")
@json_option
def log_command(name: str, as_json: bool) -> None:
    """Print the log path for run NAME."""
    emit(run_log(name), as_json=as_json)
