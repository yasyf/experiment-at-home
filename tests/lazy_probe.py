from __future__ import annotations

import click

from athome.errors import AthomeError


@click.command()
def cli() -> None:
    click.echo("probe ok")


@click.command()
def boom() -> None:
    raise AthomeError("kaboom")
