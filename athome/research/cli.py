from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import anyio
import click

from athome import launchd
from athome.cli import coro, emit, json_option
from athome.research import loop, nightly
from athome.research.driver import ClaudeCodeDriver
from athome.research.spec import ExperimentSpec

INIT_TEMPLATE = """\
name = "{name}"
metric_command = ["python", "score.py"]
metric_key = "loss"
direction = "min"
mutable_paths = ["train.py"]
immutable_paths = ["score.py"]
metric_file = ".athome-metric.json"

[budget]
max_units = 20
max_wall_s = 28800.0
hard_kill_s = 900.0
"""

spec_argument = click.argument("spec_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
repo_option = click.option(
    "--repo",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="The git repository to run against (default: the spec's repository).",
)
mirror_cc_notes_option = click.option(
    "--mirror-cc-notes",
    is_flag=True,
    default=False,
    help="Mirror journal rows to the installed cc-notes service.",
)


async def resolve_repo(spec_path: Path, repo: Path | None) -> Path:
    return repo.resolve() if repo is not None else await nightly.repo_root(spec_path)


def summary_record(report: nightly.MorningReport) -> dict[str, object]:
    return {
        "experiment": report.experiment,
        "units": report.units,
        "kept": report.kept,
        "crashes": report.crashes,
        "infra_retries": report.infra_retries,
        "accounting_aborts": report.accounting_aborts,
        "best": asdict(report.best) if report.best is not None else None,
    }


@click.group("research")
def cli() -> None:
    """Run and inspect overnight autoresearch loops."""


@cli.command("init")
@click.argument("path", type=click.Path(dir_okay=False, path_type=Path), default="experiment.toml")
@click.option("--name", default="experiment", show_default=True, help="The experiment name written into the spec.")
@json_option
@coro
async def init_command(path: Path, name: str, *, as_json: bool) -> None:
    """Scaffold an ExperimentSpec TOML at PATH (default: experiment.toml)."""
    await anyio.Path(path).write_text(INIT_TEMPLATE.format(name=name))
    emit({"created": str(path)}, as_json=as_json)


@cli.command("run")
@spec_argument
@repo_option
@mirror_cc_notes_option
@json_option
@coro
async def run_command(spec_path: Path, repo: Path | None, *, mirror_cc_notes: bool, as_json: bool) -> None:
    """Drive the greedy keep/discard loop for SPEC_PATH with the Claude Code driver."""
    spec = ExperimentSpec.load(spec_path)
    result = await loop.run(
        spec,
        driver=ClaudeCodeDriver(spec),
        repo=await resolve_repo(spec_path, repo),
        mirror_cc_notes=mirror_cc_notes,
    )
    emit({"kept": result.kept, "best": asdict(result.best) if result.best is not None else None}, as_json=as_json)


@cli.command("status")
@spec_argument
@repo_option
@json_option
@coro
async def status_command(spec_path: Path, repo: Path | None, *, as_json: bool) -> None:
    """Show SPEC_PATH's loop progress from the journal: units, kept, crashes, and the best so far."""
    spec = ExperimentSpec.load(spec_path)
    emit(summary_record(await nightly.report(spec, repo=await resolve_repo(spec_path, repo))), as_json=as_json)


@cli.command("report")
@spec_argument
@repo_option
@json_option
@coro
async def report_command(spec_path: Path, repo: Path | None, *, as_json: bool) -> None:
    """Print SPEC_PATH's morning summary: the counts, the best kept row, and every journaled unit."""
    spec = ExperimentSpec.load(spec_path)
    report = await nightly.report(spec, repo=await resolve_repo(spec_path, repo))
    emit(summary_record(report) | {"rows": [asdict(row) for row in report.rows]}, as_json=as_json)


@cli.group("nightly")
def nightly_group() -> None:
    """Install the overnight launchd agent."""


@nightly_group.command("install")
@spec_argument
@click.option("--hour", type=int, default=2, show_default=True, help="Local hour the agent fires.")
@click.option("--minute", type=int, default=0, show_default=True, help="Minute the agent fires.")
@mirror_cc_notes_option
@json_option
@coro
async def nightly_install_command(
    spec_path: Path, hour: int, minute: int, *, mirror_cc_notes: bool, as_json: bool
) -> None:
    """Install a launchd agent running SPEC_PATH's loop overnight."""
    path = await nightly.install(
        spec_path,
        calendar=launchd.Calendar(hour=hour, minute=minute),
        mirror_cc_notes=mirror_cc_notes,
    )
    emit({"installed": str(path)}, as_json=as_json)
