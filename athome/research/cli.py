from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import anyio
import click

from athome import config, launchd
from athome.cli import coro, emit, json_option
from athome.research import loop, meta, nightly, watchdog
from athome.research.driver import ClaudeCodeDriver
from athome.research.meta import MetaSettings
from athome.research.policy import ProposalPolicy
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


@cli.command("watch")
@spec_argument
@repo_option
@json_option
@coro
async def watch_command(spec_path: Path, repo: Path | None, *, as_json: bool) -> None:
    """Check SPEC_PATH's live run for quiet journal and log progress."""
    spec = ExperimentSpec.load(spec_path)
    result = await watchdog.check(spec, repo=await resolve_repo(spec_path, repo))
    emit({"live": result.live, "alarm": result.alarm}, as_json=as_json)
    if result.alarm:
        raise SystemExit(1)


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


@nightly_group.command("install-watch")
@spec_argument
@json_option
@coro
async def nightly_install_watch_command(spec_path: Path, *, as_json: bool) -> None:
    """Install a launchd agent checking SPEC_PATH's quiet alarm every ten minutes."""
    emit({"installed": str(await watchdog.install(spec_path))}, as_json=as_json)


policy_argument = click.argument("policy_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
root_option = click.option(
    "--root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="The campaign state root (default: the [research.meta] settings root).",
)


def meta_root(root: Path | None) -> Path:
    return root.resolve() if root is not None else config.load(MetaSettings).root


@cli.group("meta")
def meta_group() -> None:
    """Run the autonomous campaign outer loop."""


@meta_group.command("run")
@policy_argument
@repo_option
@root_option
@click.option("--backend", default=None, help="spawnllm backend registry name (default: settings).")
@click.option("--tier", default=None, help="Abstract spawnllm model tier (default: settings).")
@mirror_cc_notes_option
@json_option
@coro
async def meta_run_command(
    policy_path: Path,
    repo: Path | None,
    root: Path | None,
    backend: str | None,
    tier: str | None,
    *,
    mirror_cc_notes: bool,
    as_json: bool,
) -> None:
    """Run POLICY_PATH's campaign rounds until a boundary halts them."""
    settings = config.load(MetaSettings)
    result = await meta.run_campaign(
        ProposalPolicy.load(policy_path),
        repo=await resolve_repo(policy_path, repo),
        root=meta_root(root),
        backend=backend or settings.backend,
        tier=tier or settings.tier,
        mirror_cc_notes=mirror_cc_notes,
    )
    emit({"completed": result.completed, "total_usd": result.total_usd, "halted": result.halted}, as_json=as_json)


@meta_group.command("stop")
@root_option
@click.option("--reason", default="operator stop", show_default=True, help="Recorded in the stop file and ledger.")
@json_option
@coro
async def meta_stop_command(root: Path | None, reason: str, *, as_json: bool) -> None:
    """Arm the kill switch: the runner halts at the next experiment boundary."""
    emit({"stop": str(await meta.request_stop(meta_root(root), reason=reason))}, as_json=as_json)


@meta_group.command("report")
@root_option
@json_option
@coro
async def meta_report_command(root: Path | None, *, as_json: bool) -> None:
    """Summarize the campaign ledger: totals, event counts, and the kill-switch state."""
    emit(await meta.campaign_report(meta_root(root)), as_json=as_json)


@meta_group.command("watch")
@click.option(
    "--repo",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="The git repository the campaign's experiments ran against (default: the working directory).",
)
@root_option
@json_option
@coro
async def meta_watch_command(repo: Path | None, root: Path | None, *, as_json: bool) -> None:
    """Scan the campaign registry for proposal processes orphaned by a harness kill."""
    result = await watchdog.check_campaign(meta_root(root), repo=(repo or Path.cwd()).resolve())
    emit(
        {
            "live": result.live,
            "orphans": [
                {
                    "run": orphan.record.run,
                    "experiment": orphan.record.experiment,
                    "pid": orphan.pid,
                    "alive": orphan.live,
                    "latch": str(orphan.latch),
                }
                for orphan in result.orphans
            ],
        },
        as_json=as_json,
    )


@meta_group.command("approve")
@click.argument("seq", type=int)
@root_option
@json_option
@coro
async def meta_approve_command(seq: int, root: Path | None, *, as_json: bool) -> None:
    """Digest-verify pending experiment SEQ and move it into the run queue."""
    emit({"queued": str(await meta.approve(meta_root(root), seq))}, as_json=as_json)


@meta_group.command("reject")
@click.argument("seq", type=int)
@click.option("--reason", required=True, help="Why the operator refused the proposal.")
@root_option
@json_option
@coro
async def meta_reject_command(seq: int, reason: str, root: Path | None, *, as_json: bool) -> None:
    """Remove pending experiment SEQ and ledger why it was refused."""
    await meta.reject(meta_root(root), seq, reason=reason)
    emit({"rejected": seq, "reason": reason}, as_json=as_json)


@meta_group.command("install")
@policy_argument
@click.option("--hour", type=int, default=2, show_default=True, help="Local hour the agent fires.")
@click.option("--minute", type=int, default=0, show_default=True, help="Minute the agent fires.")
@json_option
@coro
async def meta_install_command(policy_path: Path, hour: int, minute: int, *, as_json: bool) -> None:
    """Install a launchd agent running POLICY_PATH's campaign overnight."""
    emit(
        {"installed": str(await meta.install(policy_path, calendar=launchd.Calendar(hour=hour, minute=minute)))},
        as_json=as_json,
    )
