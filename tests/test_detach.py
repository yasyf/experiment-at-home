from __future__ import annotations

import os
import signal
from typing import TYPE_CHECKING

import anyio
import pytest
from click.testing import CliRunner

from athome.detach import DetachedRun, DetachError, cli, launch, run_log, running, wait

if TYPE_CHECKING:
    from pathlib import Path


def kill_run(run: DetachedRun) -> None:
    try:
        os.killpg(os.getpgid(run.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


async def until_gone(name: str, *, deadline: float = 3.0) -> None:
    with anyio.fail_after(deadline):
        while running(name) is not None:
            await anyio.sleep(0.02)


async def test_launch_returns_run_metadata() -> None:
    run = await launch(["/bin/sh", "-c", "exit 0"], name="meta")
    assert run.name == "meta"
    assert run.pid > 0
    assert run.log_path == run_log("meta")
    assert await wait("meta", poll=0.02, timeout=5) == 0


async def test_wait_round_trips_success_exit() -> None:
    await launch(["/bin/sh", "-c", "echo hi; exit 0"], name="ok")
    assert await wait("ok", poll=0.02, timeout=5) == 0


async def test_wait_round_trips_failure_exit() -> None:
    await launch(["/bin/sh", "-c", "exit 7"], name="fail")
    assert await wait("fail", poll=0.02, timeout=5) == 7


async def test_log_captures_stdout_stderr_and_sentinel() -> None:
    await launch(["/bin/sh", "-c", "echo out-line; echo err-line 1>&2; exit 3"], name="logs")
    assert await wait("logs", poll=0.02, timeout=5) == 3
    text = run_log("logs").read_text()
    assert "out-line" in text
    assert "err-line" in text
    assert "ATHOME-RUN-DONE name=logs exit=3" in text


async def test_env_prefix_is_prepended(monkeypatch: pytest.MonkeyPatch) -> None:
    from athome.config import load

    monkeypatch.setenv("ATHOME_ENV_PREFIX_CMD", "export ATHOME_MARKER=on")
    load.cache_clear()
    await launch(["/bin/sh", "-c", 'echo "marker=$ATHOME_MARKER"'], name="prefix")
    assert await wait("prefix", poll=0.02, timeout=5) == 0
    assert "marker=on" in run_log("prefix").read_text()


async def test_running_reflects_liveness() -> None:
    run = await launch(["/bin/sh", "-c", "sleep 1"], name="live")
    assert running("live") == run.pid
    assert await wait("live", poll=0.05, timeout=5) == 0
    await until_gone("live")
    assert running("live") is None


async def test_running_is_none_for_unknown_name() -> None:
    assert running("never-launched") is None


async def test_duplicate_live_name_raises_detach_error() -> None:
    run = await launch(["/bin/sh", "-c", "sleep 1"], name="dup")
    try:
        with pytest.raises(DetachError, match="already live"):
            await launch(["/bin/sh", "-c", "exit 0"], name="dup")
    finally:
        kill_run(run)
    await until_gone("dup")


async def test_relaunch_allowed_after_completion() -> None:
    await launch(["/bin/sh", "-c", "exit 4"], name="reuse")
    assert await wait("reuse", poll=0.02, timeout=5) == 4
    await until_gone("reuse")
    run2 = await launch(["/bin/sh", "-c", "exit 5"], name="reuse")
    assert run2.pid > 0
    await until_gone("reuse")
    assert "ATHOME-RUN-DONE name=reuse exit=5" in run_log("reuse").read_text()


async def test_wait_times_out_on_unfinished_run() -> None:
    run = await launch(["/bin/sh", "-c", "sleep 5"], name="slow")
    try:
        with pytest.raises(TimeoutError, match="did not finish"):
            await wait("slow", poll=0.05, timeout=0.3)
    finally:
        kill_run(run)


def test_cli_launch_wait_log_round_trip(tmp_path: Path) -> None:
    runner = CliRunner()
    launched = runner.invoke(cli, ["--detach", "--name", "clijob", "--", "/bin/sh", "-c", "exit 6"])
    assert launched.exit_code == 0
    assert "pid" in launched.output
    assert "log" in launched.output

    waited = runner.invoke(cli, ["wait", "clijob", "--poll", "0.02", "--timeout", "5"])
    assert waited.exit_code == 0
    assert "exit: 6" in waited.output

    logged = runner.invoke(cli, ["log", "clijob"])
    assert logged.exit_code == 0
    assert logged.output.strip() == str(run_log("clijob"))


def test_cli_wait_emits_json() -> None:
    runner = CliRunner()
    assert runner.invoke(cli, ["--detach", "--name", "cjson", "--", "/bin/sh", "-c", "exit 2"]).exit_code == 0
    result = runner.invoke(cli, ["wait", "cjson", "--poll", "0.02", "--timeout", "5", "--json"])
    assert result.exit_code == 0
    assert result.output.strip() == '{"name": "cjson", "exit": 2}'
