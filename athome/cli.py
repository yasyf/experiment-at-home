from __future__ import annotations

import functools
import importlib
import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import anyio
import click
from loguru import logger

from athome.errors import AthomeError

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine


def coro[**P, R](f: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, R]:
    """Wrap an async Click callback so it runs to completion under ``anyio.run``."""

    @functools.wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return anyio.run(functools.partial(f, *args, **kwargs))

    return wrapper


json_option = click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")


def emit(data: object, *, as_json: bool) -> None:
    """Render ``data`` to stdout: JSON when ``as_json``, else key/value lines or plain rows."""
    if as_json:
        click.echo(json.dumps(data, default=str))
        return
    match data:
        case Mapping():
            for key, value in data.items():
                click.echo(f"{key}: {value}")
        case str():
            click.echo(data)
        case Sequence():
            for row in data:
                emit(row, as_json=False)
        case _:
            click.echo(str(data))


class LazyGroup(click.Group):
    """Click group whose subcommands import lazily from ('module:attr', short_help) entries."""

    def __init__(self, *, lazy_subcommands: Mapping[str, tuple[str, str]], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.lazy_subcommands = dict(lazy_subcommands)

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted({*super().list_commands(ctx), *self.lazy_subcommands})

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        if (entry := self.lazy_subcommands.get(cmd_name)) is None:
            return super().get_command(ctx, cmd_name)
        module_name, _, attr = entry[0].partition(":")
        return getattr(importlib.import_module(module_name), attr)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        names = [name for name in self.list_commands(ctx) if (cmd := self.commands.get(name)) is None or not cmd.hidden]
        if not names:
            return
        limit = formatter.width - 6 - max(len(name) for name in names)
        rows = [
            (
                name,
                entry[1]
                if (entry := self.lazy_subcommands.get(name)) is not None
                else self.commands[name].get_short_help_str(limit),
            )
            for name in names
        ]
        with formatter.section("Commands"):
            formatter.write_dl(rows)


class AthomeGroup(LazyGroup):
    """Root group: catches AthomeError from subcommands → loguru error + exit(err.exit_code)."""

    def invoke(self, ctx: click.Context) -> object:
        try:
            return super().invoke(ctx)
        except AthomeError as err:
            logger.error("{}", err)
            ctx.exit(err.exit_code)


main = click.version_option(package_name="experiment-at-home")(
    AthomeGroup(
        name="athome",
        help="The plumbing every local AI experiment rebuilds, built once.",
        lazy_subcommands={
            "cache": ("athome.cache:cli", "Inspect the shared content-keyed cache."),
            "launchd": ("athome.launchd:cli", "List, inspect, and remove athome launchd agents."),
            "run": ("athome.detach:cli", "Launch and track detached overnight runs."),
            "sync": ("athome.sync:cli", "Mirror a tree with sha256 verification."),
        },
    )
)
