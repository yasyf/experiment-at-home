from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio
import pytest

from athome.detach import run_exitfile, run_log, run_pidfile
from athome.research import meta, procs, watchdog
from athome.research.driver import ClaudeCodeDriver
from athome.research.errors import AccountingIntegrityError, PreflightFailure
from athome.research.journal import Journal
from athome.research.loop import experiment_lock
from athome.research.loop import run as run_loop
from athome.research.procs import PROCS_NAME, ExperimentProcs, ProcessEntry, ProcessRegistry
from athome.research.spec import Budget, ProposalTimeout
from tests.test_research_driver import (
    FAKE_CLAUDE_COST,
    FAKE_CLAUDE_COST_THEN_HANGS,
    FAKE_CLAUDE_HANGS,
    fake_claude,
    make_spec,
    plain_checkout,
    toy_repo,
)
from tests.test_research_meta import forbidden_factory, make_campaign_policy, make_proposal
from tests.test_research_meta import toy_repo as campaign_repo
from tests.test_research_propose import scripted_backend

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from athome.research.spec import ExperimentSpec

DIGEST = "d" * 64


def make_registry(root: Path) -> ProcessRegistry:
    return ProcessRegistry(root / PROCS_NAME)


def run_pid_file(registry: ProcessRegistry, run: str) -> Path:
    return registry.path.parent / f"{run}.pid"


def table_with(*entries: tuple[int, str], pgid: int | None = None) -> Callable[[], dict[int, ProcessEntry]]:
    return lambda: {
        pid: ProcessEntry(pgid=pgid if pgid is not None else pid, command=f"/bin/sh -c claude; echo $? > {run}.exit")
        for pid, run in entries
    }


def registered(registry: ProcessRegistry, *, run: str = "run-a", pid: int | None = 4242) -> str:
    handle = registry.experiment("001-round1", seq=1, spec_digest=DIGEST)
    handle.register(
        run,
        log=registry.path.parent / f"{run}.log",
        pid_file=run_pid_file(registry, run),
        exit_file=registry.path.parent / f"{run}.exit",
    )
    if pid is not None:
        handle.bind(run, pid=pid, pgid=pid)
    return run


def test_registry_folds_register_bind_and_terminal_lines(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    handle = registry.experiment("001-round1", seq=1, spec_digest=DIGEST)
    handle.register("run-a", log=tmp_path / "a.log", pid_file=tmp_path / "a.pid", exit_file=tmp_path / "a.exit")
    handle.bind("run-a", pid=4242, pgid=4242)
    handle.finalize("run-a", outcome="accounted")

    [record] = registry.records()
    assert record.run == "run-a"
    assert record.experiment == "001-round1" and record.seq == 1 and record.spec_digest == DIGEST
    assert record.pid == 4242 and record.pgid == 4242
    assert record.outcome == "accounted"
    assert (record.log, record.pid_file, record.exit_file) == (
        tmp_path / "a.log",
        tmp_path / "a.pid",
        tmp_path / "a.exit",
    )


def test_registry_skips_a_torn_final_line_and_heals_on_append(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    registered(registry, run="run-a")
    with registry.path.open("ab") as sink:
        sink.write(b'{"op": "terminal", "run": "run-a", "outco')

    [record] = registry.records()
    assert record.outcome is None  # the torn terminal line never counts as accounted

    registered(registry, run="run-b")
    assert {record.run for record in registry.records()} == {"run-a", "run-b"}


async def test_scan_alarms_a_live_orphan_and_leaves_it_non_terminal(tmp_path: Path) -> None:
    repo = campaign_repo(tmp_path / "repo")
    registry = make_registry(tmp_path / "meta")
    registered(registry)

    [orphan] = await procs.scan(registry, repo=repo, alive=lambda pid: pid == 4242, table=table_with((4242, "run-a")))

    assert orphan.live and orphan.pid == 4242
    assert not orphan.latch.exists()
    [again] = await procs.scan(registry, repo=repo, alive=lambda pid: pid == 4242, table=table_with((4242, "run-a")))
    assert again.live  # still non-terminal: every later scan keeps refusing until reconciled


async def test_scan_latches_a_dead_orphan_and_marks_it_terminal(tmp_path: Path) -> None:
    repo = campaign_repo(tmp_path / "repo")
    registry = make_registry(tmp_path / "meta")
    run = registered(registry)

    [orphan] = await procs.scan(registry, repo=repo, alive=lambda pid: False, table=table_with())

    assert not orphan.live and orphan.pid == 4242
    latch = json.loads(orphan.latch.read_text())
    assert latch["run"] == run and "reconcile the spend" in latch["reason"]
    [record] = registry.records()
    assert record.outcome == "orphaned"

    assert await procs.scan(registry, repo=repo, alive=lambda pid: False, table=table_with()) == []
    [stale] = await procs.unreconciled(registry, repo=repo)
    assert stale.latch == orphan.latch
    orphan.latch.unlink()
    assert await procs.unreconciled(registry, repo=repo) == []


async def test_scan_resolves_the_pid_from_the_pid_file_when_the_bind_never_landed(tmp_path: Path) -> None:
    repo = campaign_repo(tmp_path / "repo")
    registry = make_registry(tmp_path / "meta")
    run = registered(registry, pid=None)
    run_pid_file(registry, run).write_text("4242")

    [orphan] = await procs.scan(registry, repo=repo, alive=lambda pid: pid == 4242, table=table_with((4242, run)))

    assert orphan.live and orphan.pid == 4242


async def test_scan_latches_a_never_bound_record_with_no_pid_file(tmp_path: Path) -> None:
    repo = campaign_repo(tmp_path / "repo")
    registry = make_registry(tmp_path / "meta")
    registered(registry, pid=None)

    [orphan] = await procs.scan(registry, repo=repo, alive=lambda pid: pytest.fail("no pid to probe"), table=table_with())

    assert not orphan.live and orphan.pid is None and orphan.latch.exists()


async def test_scan_latches_an_alive_pid_reused_by_an_unrelated_process(tmp_path: Path) -> None:
    repo = campaign_repo(tmp_path / "repo")
    registry = make_registry(tmp_path / "meta")
    run = registered(registry)

    [orphan] = await procs.scan(
        registry, repo=repo, alive=lambda pid: True, table=table_with((4242, "some-unrelated-daemon"))
    )

    assert not orphan.live and orphan.pid == 4242  # alive but not our process: PID reuse, never a LIVE block
    assert json.loads(orphan.latch.read_text())["run"] == run
    [record] = registry.records()
    assert record.outcome == "orphaned"


async def test_scan_latches_a_pgid_mismatch_as_pid_reuse(tmp_path: Path) -> None:
    repo = campaign_repo(tmp_path / "repo")
    registry = make_registry(tmp_path / "meta")
    registered(registry)

    [orphan] = await procs.scan(
        registry, repo=repo, alive=lambda pid: True, table=table_with((4242, "run-a"), pgid=999)
    )

    assert not orphan.live and orphan.latch.exists()
    [record] = registry.records()
    assert record.outcome == "orphaned"


async def test_scan_treats_an_alive_pid_as_live_when_the_table_is_unavailable(tmp_path: Path) -> None:
    repo = campaign_repo(tmp_path / "repo")
    registry = make_registry(tmp_path / "meta")
    registered(registry)

    [orphan] = await procs.scan(registry, repo=repo, alive=lambda pid: True, table=lambda: None)

    assert orphan.live  # identity unverifiable: refuse conservatively, never latch a possibly-live biller
    assert not orphan.latch.exists()


async def test_run_campaign_refuses_startup_until_the_orphan_latch_is_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = campaign_repo(tmp_path / "repo")
    root = tmp_path / "meta"
    registered(make_registry(root))
    monkeypatch.setattr(procs, "pid_alive", lambda pid: False)
    monkeypatch.setattr(procs, "process_table", table_with())
    backend = scripted_backend([])

    with pytest.raises(AccountingIntegrityError, match="unreconciled spend"):
        await meta.run_campaign(
            make_campaign_policy(), repo=repo, root=root, backend=backend, driver_factory=forbidden_factory
        )
    assert backend.calls == []
    latch = await procs.abort_latch(repo, "001-round1")
    assert latch.exists()

    with pytest.raises(AccountingIntegrityError, match="latched"):  # the latch still refuses the next startup
        await meta.run_campaign(
            make_campaign_policy(), repo=repo, root=root, backend=backend, driver_factory=forbidden_factory
        )

    latch.unlink()
    await meta.request_stop(root, reason="registry reconciled")
    result = await meta.run_campaign(
        make_campaign_policy(), repo=repo, root=root, backend=backend, driver_factory=forbidden_factory
    )
    assert result.halted == "stop requested: registry reconciled"


async def test_run_campaign_refuses_startup_while_a_live_orphan_bills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = campaign_repo(tmp_path / "repo")
    root = tmp_path / "meta"
    registered(make_registry(root))
    monkeypatch.setattr(procs, "pid_alive", lambda pid: True)
    monkeypatch.setattr(procs, "process_table", table_with((4242, "run-a")))

    for _ in range(2):  # non-terminal while live: every startup keeps refusing
        with pytest.raises(AccountingIntegrityError, match="LIVE pid 4242"):
            await meta.run_campaign(
                make_campaign_policy(),
                repo=repo,
                root=root,
                backend=scripted_backend([]),
                driver_factory=forbidden_factory,
            )
    assert not (await procs.abort_latch(repo, "001-round1")).exists()


async def test_run_campaign_default_factory_binds_the_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[ExperimentProcs | None] = []

    @dataclass(frozen=True, slots=True)
    class CapturingDriver:
        spec: ExperimentSpec
        procs: ExperimentProcs | None = None
        label: str = "capturing"

        async def preflight(self) -> None:
            captured.append(self.procs)
            raise PreflightFailure("captured; stop the round")

        async def propose(self, contract: str, workdir: Path, *, budget_usd: float | None) -> float:
            raise AssertionError("preflight always fails first")

        async def recover_cost(self) -> float:
            return 0.0

        def settle(self) -> None:
            return None

    monkeypatch.setattr(meta, "ClaudeCodeDriver", CapturingDriver)
    repo = campaign_repo(tmp_path / "repo")
    root = tmp_path / "meta"

    await meta.run_campaign(
        make_campaign_policy(max_consecutive_failures=1),
        repo=repo,
        root=root,
        backend=scripted_backend([make_proposal(1)]),
    )

    [handle] = captured
    assert isinstance(handle, ExperimentProcs)
    assert handle.experiment == "001-round1" and handle.seq == 1
    assert handle.registry.path == root / PROCS_NAME
    assert len(handle.spec_digest) == 64


async def test_claude_driver_registers_binds_and_marks_terminal_only_on_settle(tmp_path: Path) -> None:
    workdir = plain_checkout(toy_repo(tmp_path))
    registry = make_registry(tmp_path / "meta")
    driver = ClaudeCodeDriver(
        make_spec(budget=Budget(max_units=1)),
        command=fake_claude(tmp_path, FAKE_CLAUDE_COST),
        poll=0.02,
        timeout_s=10,
        procs=registry.experiment("toy", seq=1, spec_digest=DIGEST),
    )

    cost = await driver.propose("the generated contract", workdir, budget_usd=None)

    assert cost == 0.4207
    [record] = registry.records()
    assert record.experiment == "toy"
    assert record.outcome is None  # a kill in this window must leave the record for the orphan scan
    assert record.pid is not None and record.pgid == record.pid  # the detached run leads its own session
    assert (record.log, record.pid_file, record.exit_file) == (
        run_log(record.run),
        run_pidfile(record.run),
        run_exitfile(record.run),
    )

    driver.settle()

    [record] = registry.records()
    assert record.outcome == "accounted"
    driver.settle()  # idempotent: nothing pending appends nothing
    assert len(registry.events()) == 3


async def test_claude_driver_timeout_carried_spend_stays_pending_until_settle(tmp_path: Path) -> None:
    workdir = plain_checkout(toy_repo(tmp_path))
    registry = make_registry(tmp_path / "meta")
    driver = ClaudeCodeDriver(
        make_spec(budget=Budget(max_units=1)),
        command=fake_claude(tmp_path, FAKE_CLAUDE_COST_THEN_HANGS),
        poll=0.02,
        timeout_s=1.0,
        procs=registry.experiment("toy", seq=1, spec_digest=DIGEST),
    )

    with pytest.raises(ProposalTimeout) as excinfo:
        await driver.propose("the generated contract", workdir, budget_usd=None)

    assert excinfo.value.cost == 0.99
    [record] = registry.records()
    assert record.outcome is None  # the carried spend is not durable yet

    driver.settle()

    [record] = registry.records()
    assert record.outcome == "accounted"


async def test_claude_driver_recovered_spend_stays_pending_until_settle(tmp_path: Path) -> None:
    workdir = plain_checkout(toy_repo(tmp_path))
    registry = make_registry(tmp_path / "meta")
    driver = ClaudeCodeDriver(
        make_spec(budget=Budget(max_units=1)),
        command=fake_claude(tmp_path, FAKE_CLAUDE_COST_THEN_HANGS),
        poll=0.02,
        timeout_s=30,
        procs=registry.experiment("toy", seq=1, spec_digest=DIGEST),
    )

    with anyio.move_on_after(1.5):
        await driver.propose("the generated contract", workdir, budget_usd=None)

    assert await driver.recover_cost() == 0.99
    [record] = registry.records()
    assert record.outcome is None  # recovery hands spend to the caller; the durable write hasn't happened

    driver.settle()

    [record] = registry.records()
    assert record.outcome == "accounted"


async def test_loop_settles_the_registry_after_the_journal_row(tmp_path: Path) -> None:
    (repo_dir := tmp_path / "repo").mkdir()
    repo = toy_repo(repo_dir)
    registry = make_registry(tmp_path / "meta")
    spec = make_spec(budget=Budget(max_units=1))
    driver = ClaudeCodeDriver(
        spec,
        command=fake_claude(tmp_path, FAKE_CLAUDE_COST),
        poll=0.02,
        timeout_s=10,
        procs=registry.experiment("toy", seq=1, spec_digest=DIGEST),
    )

    with anyio.fail_after(30.0):
        await run_loop(spec, driver=driver, repo=repo)

    [record] = registry.records()
    assert record.outcome == "accounted"


async def test_loop_journal_failure_leaves_the_registry_non_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (repo_dir := tmp_path / "repo").mkdir()
    repo = toy_repo(repo_dir)
    registry = make_registry(tmp_path / "meta")
    spec = make_spec(budget=Budget(max_units=1))
    driver = ClaudeCodeDriver(
        spec,
        command=fake_claude(tmp_path, FAKE_CLAUDE_COST),
        poll=0.02,
        timeout_s=10,
        procs=registry.experiment("toy", seq=1, spec_digest=DIGEST),
    )

    async def fail_append(self: Journal, row: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Journal, "append", fail_append)
    with anyio.fail_after(30.0), pytest.raises(AccountingIntegrityError, match="could not append journal row"):
        await run_loop(spec, driver=driver, repo=repo)

    [record] = registry.records()
    assert record.outcome is None  # never terminal without the durable journal row; the orphan scan latches it


async def test_claude_driver_leaves_an_unaccounted_proposal_non_terminal(tmp_path: Path) -> None:
    workdir = plain_checkout(toy_repo(tmp_path))
    registry = make_registry(tmp_path / "meta")
    driver = ClaudeCodeDriver(
        make_spec(budget=Budget(max_units=1)),
        command=fake_claude(tmp_path, FAKE_CLAUDE_HANGS),
        poll=0.02,
        timeout_s=1.0,
        procs=registry.experiment("toy", seq=1, spec_digest=DIGEST),
    )

    with pytest.raises(AccountingIntegrityError):
        await driver.propose("the generated contract", workdir, budget_usd=None)

    [record] = registry.records()
    assert record.outcome is None  # unknown spend stays non-terminal for the orphan scan to latch


async def test_check_campaign_reports_live_and_leaves_the_registry_alone(tmp_path: Path) -> None:
    repo = campaign_repo(tmp_path / "repo")
    root = tmp_path / "meta"
    root.mkdir()
    registry = make_registry(root)
    registered(registry)

    async with experiment_lock(root / meta.LOCK_NAME):
        result = await watchdog.check_campaign(root, repo=repo)

    assert result.live and result.orphans == ()
    [record] = registry.records()
    assert record.outcome is None and not (await procs.abort_latch(repo, "001-round1")).exists()


async def test_check_campaign_alerts_and_latches_orphans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = campaign_repo(tmp_path / "repo")
    root = tmp_path / "meta"
    root.mkdir()
    registry = make_registry(root)
    registered(registry, run="run-live", pid=4242)
    registered(registry, run="run-dead", pid=9999)
    monkeypatch.setattr(procs, "pid_alive", lambda pid: pid == 4242)
    monkeypatch.setattr(procs, "process_table", table_with((4242, "run-live")))
    alerts: list[tuple[str, str, str]] = []

    async def fake_alert(_journal: Path, *, unit: str, detail: str, kind: str = "quiet_alarm") -> None:
        alerts.append((kind, unit, detail))

    monkeypatch.setattr(watchdog, "_alert", fake_alert)

    result = await watchdog.check_campaign(root, repo=repo)

    assert not result.live and {orphan.record.run for orphan in result.orphans} == {"run-live", "run-dead"}
    assert {kind for kind, _, _ in alerts} == {"orphan_alarm"}
    assert any("billing with no harness" in detail for _, _, detail in alerts)
    assert any("abort latch written" in detail for _, _, detail in alerts)
    assert (await procs.abort_latch(repo, "001-round1")).exists()
    outcomes = {record.run: record.outcome for record in registry.records()}
    assert outcomes == {"run-live": None, "run-dead": "orphaned"}
