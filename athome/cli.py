from __future__ import annotations

import click
from loguru import logger


@click.group()
@click.version_option(package_name="experiment-at-home")
def main() -> None:
    """The plumbing every local AI experiment rebuilds, built once."""


@main.command()
def hello() -> None:
    """Print a greeting — the starter command."""
    logger.debug("hello invoked")
    click.echo("Hello from experiment-at-home!")
