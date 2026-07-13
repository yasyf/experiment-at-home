from __future__ import annotations

import sys
from pathlib import Path

import click
import pytest
from click.testing import CliRunner
from loguru import logger

from athome.cli import AthomeGroup, coro, emit, json_option, main


def test_coro_runs_async_callback() -> None:
    @coro
    async def double(x: int) -> int:
        return x * 2

    assert double(21) == 42


@pytest.mark.parametrize(
    ("data", "as_json", "expected"),
    [
        pytest.param({"a": 1, "b": 2}, True, '{"a": 1, "b": 2}\n', id="json-mapping"),
        pytest.param({"a": 1, "b": 2}, False, "a: 1\nb: 2\n", id="human-mapping"),
        pytest.param(["x", "y"], False, "x\ny\n", id="human-sequence"),
        pytest.param(42, False, "42\n", id="human-scalar"),
        pytest.param({"p": Path("/tmp/x")}, True, '{"p": "/tmp/x"}\n', id="json-default-str"),
    ],
)
def test_emit(data: object, as_json: bool, expected: str, capsys: pytest.CaptureFixture[str]) -> None:
    emit(data, as_json=as_json)
    assert capsys.readouterr().out == expected


def test_json_option_is_flag() -> None:
    @click.command()
    @json_option
    def cmd(as_json: bool) -> None:
        emit({"ok": as_json}, as_json=as_json)

    result = CliRunner().invoke(cmd, ["--json"])
    assert result.exit_code == 0
    assert result.output == '{"ok": true}\n'


def test_main_lists_subcommands_without_import() -> None:
    modules = (
        "athome.cache",
        "athome.launchd",
        "athome.detach",
        "athome.sync",
        "athome.serve",
        "athome.llm.batch",
        "athome.ocr.profiles",
        "athome.bakeoff",
        "athome.hf",
        "athome.research.cli",
    )
    saved = {name: sys.modules.pop(name, None) for name in modules}
    try:
        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0
        for name in ("cache", "launchd", "run", "sync", "serve", "status", "batch", "ocr", "bakeoff", "hf", "research"):
            assert name in result.output
        for module in modules:
            assert module not in sys.modules
    finally:
        sys.modules.update({name: module for name, module in saved.items() if module is not None})


def test_main_list_commands_sorted() -> None:
    assert main.list_commands(click.Context(main)) == [
        "bakeoff",
        "batch",
        "cache",
        "hf",
        "launchd",
        "ocr",
        "research",
        "run",
        "serve",
        "status",
        "sync",
    ]


def test_lazy_group_defers_import_until_invoked() -> None:
    sys.modules.pop("tests.lazy_probe", None)
    group = AthomeGroup(name="t", lazy_subcommands={"probe": ("tests.lazy_probe:cli", "Probe.")})
    help_result = CliRunner().invoke(group, ["--help"])
    assert help_result.exit_code == 0
    assert "probe" in help_result.output
    assert "tests.lazy_probe" not in sys.modules
    run_result = CliRunner().invoke(group, ["probe"])
    assert run_result.exit_code == 0
    assert run_result.output.strip() == "probe ok"
    assert "tests.lazy_probe" in sys.modules


def test_athome_error_renders_stderr_and_exits_1() -> None:
    messages: list[str] = []
    sink = logger.add(messages.append, level="ERROR", format="{message}")
    group = AthomeGroup(name="t", lazy_subcommands={"boom": ("tests.lazy_probe:boom", "Boom.")})
    try:
        result = CliRunner().invoke(group, ["boom"])
    finally:
        logger.remove(sink)
    assert result.exit_code == 1
    assert len(messages) == 1
    assert messages[0].strip() == "kaboom"


def test_version_option_reports_package_version() -> None:
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.0.0" in result.output
