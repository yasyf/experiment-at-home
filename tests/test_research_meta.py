from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import pytest
from click.testing import CliRunner

from athome import launchd
from athome.research import meta
from athome.research.cli import cli
from athome.research.driver import StubProposal
from athome.research.errors import PreflightFailure
from athome.research.meta import CampaignEvent, Ledger, LedgerRow, PoisonedLedger
from athome.research.policy import CampaignBudget, ExperimentTemplate, ProposalPolicy
from athome.research.propose import MAX_PROPOSAL_ATTEMPTS, Proposal
from athome.research.retro import RetroJournal, RetroVerdict
from athome.research.spec import Budget, ExperimentSpec
from tests.test_research_propose import scripted_backend

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from athome.research.driver import Driver

# Immutable evaluator: metric from train.py to the JSON channel, a LYING 999.0 to stdout.
SCORE_PY = textwrap.dedent(
    """
    import json, pathlib
    namespace = {}
    exec(pathlib.Path("train.py").read_text(), namespace)
    pathlib.Path(".athome-metric.json").write_text(json.dumps({"loss": namespace["LOSS"]}))
    print("loss=999.0")
    """
).strip()

POLICY_TOML = textwrap.dedent(
    """
    mode = "auto"

    [campaign]
    max_experiments = 4
    max_total_usd = 7.0
    max_wall_s = 100000.0
    max_consecutive_failures = 3

    [ceilings]
    max_units = 4
    max_wall_s = 600.0
    hard_kill_s = 120.0
    max_usd = 10.0

    [[templates]]
    name = "toy"
    metric_command = ["python", "score.py"]
    metric_key = "loss"
    direction = "min"
    immutable_paths = ["score.py"]
    mutable_allowlist = ["train.py"]
    """
).strip()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def toy_repo(root: Path, *, initial_loss: float = 1.0) -> Path:
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "toy@localhost")
    git(root, "config", "user.name", "toy")
    (root / "train.py").write_text(f"LOSS = {initial_loss}\n")
    (root / "score.py").write_text(SCORE_PY + "\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    return root


def make_campaign_policy(
    *,
    mode: str = "auto",
    max_experiments: int = 4,
    max_total_usd: float = 7.0,
    max_consecutive_failures: int = 3,
) -> ProposalPolicy:
    return ProposalPolicy(
        mode=mode,
        campaign=CampaignBudget(
            max_experiments=max_experiments,
            max_total_usd=max_total_usd,
            max_wall_s=100000.0,
            max_consecutive_failures=max_consecutive_failures,
        ),
        ceilings=Budget(max_units=4, max_wall_s=600.0, hard_kill_s=120.0, max_usd=10.0),
        templates=(
            ExperimentTemplate(
                name="toy",
                metric_command=(sys.executable, "score.py"),
                metric_key="loss",
                direction="min",
                immutable_paths=("score.py",),
                mutable_allowlist=("train.py",),
            ),
        ),
    )


def make_proposal(index: int, **overrides: object) -> Proposal:
    defaults: dict[str, object] = {
        "template": "toy",
        "name": f"round{index}",
        "mutable_paths": ("train.py",),
        "max_units": 1,
        "max_usd": 2.0,
        "max_wall_s": 300.0,
        "hard_kill_s": 60.0,
        "hypothesis": "A lower loss constant scores lower.",
    }
    return Proposal.model_validate(defaults | overrides)


def make_verdict(summary: str) -> RetroVerdict:
    return RetroVerdict(outcome="improved", summary=summary, evidence=("the loss fell",), next_steps=("go lower",))


def loss_proposal(loss: float) -> StubProposal:
    return StubProposal({"train.py": f"LOSS = {loss}\n"})


@dataclass(frozen=True, slots=True)
class PaidDriver:
    proposals: Iterator[StubProposal]
    contracts: list[str]
    cost: float = 2.0
    label: str = "paid-stub"

    async def preflight(self) -> None:
        return None

    async def propose(self, contract: str, workdir: Path) -> float:
        self.contracts.append(contract)
        for relative, content in next(self.proposals).files.items():
            target = anyio.Path(workdir) / relative
            await target.parent.mkdir(parents=True, exist_ok=True)
            await target.write_text(content)
        return self.cost

    async def recover_cost(self) -> float:
        return self.cost


@dataclass(frozen=True, slots=True)
class StopArmingDriver:
    inner: PaidDriver
    root: Path
    label: str = "stop-arming"

    async def preflight(self) -> None:
        await self.inner.preflight()

    async def propose(self, contract: str, workdir: Path) -> float:
        await meta.request_stop(self.root, reason="mid-campaign halt")
        return await self.inner.propose(contract, workdir)

    async def recover_cost(self) -> float:
        return await self.inner.recover_cost()


@dataclass(frozen=True, slots=True)
class BrokenPreflightDriver:
    label: str = "broken-preflight"

    async def preflight(self) -> None:
        raise PreflightFailure("scorer missing on this machine")

    async def propose(self, contract: str, workdir: Path) -> float:
        raise AssertionError("a failed preflight must never reach a proposal")

    async def recover_cost(self) -> float:
        return 0.0


@dataclass(frozen=True, slots=True)
class InfraDriver:
    label: str = "infra"

    async def preflight(self) -> None:
        return None

    async def propose(self, contract: str, workdir: Path) -> float:
        raise OSError("network unreachable")

    async def recover_cost(self) -> float:
        return 0.0


def paid_factory(losses: Sequence[float], contracts: list[str]) -> Callable[[ExperimentSpec], PaidDriver]:
    remaining = iter(losses)
    return lambda spec: PaidDriver(iter([loss_proposal(next(remaining))]), contracts)


def forbidden_factory(spec: ExperimentSpec) -> Driver:
    raise AssertionError("no driver may be constructed in this scenario")


def ledger_rows(root: Path) -> list[LedgerRow]:
    return Ledger.open(root / meta.LEDGER_NAME).rows()


def retro_names(root: Path) -> list[str]:
    return [record.experiment for record in RetroJournal.open(root / meta.RETROS_NAME, mirror_cc_notes=False).records()]


async def test_three_experiment_campaign_refuses_the_fourth_reservation(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path / "repo")
    root = tmp_path / "meta"
    contracts: list[str] = []
    backend = scripted_backend(
        [
            make_proposal(1),
            make_verdict("round one improved the loss"),
            make_proposal(2),
            make_verdict("round two improved the loss"),
            make_proposal(3),
            make_verdict("round three improved the loss"),
            make_proposal(4),
        ]
    )
    result = await meta.run_campaign(
        make_campaign_policy(),
        repo=repo,
        root=root,
        backend=backend,
        driver_factory=paid_factory([0.9, 0.8, 0.7], contracts),
    )

    assert result.completed == 3
    assert result.halted is not None and "reservation refused" in result.halted
    ledger = Ledger.open(root / meta.LEDGER_NAME)
    assert ledger.total_usd() == 6.0 <= 7.0
    completed = [row for row in ledger.rows() if row.event is CampaignEvent.COMPLETED]
    assert [row.seq for row in completed] == [1, 2, 3]
    assert [row.usd for row in completed] == [2.0, 2.0, 2.0]
    assert [row.extra["kept"] for row in completed] == [1, 1, 1]
    assert sum(row.event is CampaignEvent.RESERVED for row in ledger.rows()) == 3
    assert not ledger.open_reservations()
    proposed = [row.seq for row in ledger.rows() if row.event is CampaignEvent.PROPOSED]
    assert proposed == [1, 2, 3, 4]
    assert ledger.rows()[-1].event is CampaignEvent.STOPPED

    assert retro_names(root) == ["001-round1", "002-round2", "003-round3"]
    audit = root / meta.EXPERIMENTS_DIR / "001-round1"
    assert ExperimentSpec.load(audit / meta.SPEC_NAME).name == "001-round1"
    proposal_record = json.loads((audit / meta.PROPOSAL_NAME).read_text())
    assert proposal_record["seq"] == 1 and proposal_record["template"] == "toy"

    prompts = [call.prompt for call in backend.calls]
    assert len(prompts) == 7
    assert "001-round1: improved" in prompts[2] and "round one improved the loss" in prompts[2]
    assert "round two improved the loss" in prompts[4]


async def test_no_prompt_or_contract_carries_run_log_bytes(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path / "repo")
    root = tmp_path / "meta"
    contracts: list[str] = []
    backend = scripted_backend([make_proposal(1), make_verdict("quiet"), make_proposal(2)])
    await meta.run_campaign(
        make_campaign_policy(max_experiments=1),
        repo=repo,
        root=root,
        backend=backend,
        driver_factory=paid_factory([0.9], contracts),
    )

    assert contracts and all("999.0" not in contract for contract in contracts)
    assert backend.calls and all("999.0" not in call.prompt for call in backend.calls)
    assert all(
        "999.0" not in meta.render_retro(record)
        for record in RetroJournal.open(root / meta.RETROS_NAME, mirror_cc_notes=False).records()
    )


async def test_stop_file_halts_before_any_launch(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path / "repo")
    root = tmp_path / "meta"
    await meta.request_stop(root, reason="halt now")
    backend = scripted_backend([])

    result = await meta.run_campaign(
        make_campaign_policy(), repo=repo, root=root, backend=backend, driver_factory=forbidden_factory
    )

    assert result.halted == "stop requested: halt now"
    assert [row.event for row in ledger_rows(root)] == [CampaignEvent.STOPPED]
    assert backend.calls == []


async def test_stop_mid_campaign_halts_at_the_experiment_boundary(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path / "repo")
    root = tmp_path / "meta"
    contracts: list[str] = []
    losses = iter([0.9])
    backend = scripted_backend([make_proposal(1), make_verdict("one done")])

    result = await meta.run_campaign(
        make_campaign_policy(),
        repo=repo,
        root=root,
        backend=backend,
        driver_factory=lambda spec: StopArmingDriver(PaidDriver(iter([loss_proposal(next(losses))]), contracts), root),
    )

    assert result.completed == 1
    assert result.halted == "stop requested: mid-campaign halt"
    rows = ledger_rows(root)
    assert sum(row.event is CampaignEvent.STARTED for row in rows) == 1
    assert [row.seq for row in rows if row.event is CampaignEvent.PROPOSED] == [1]
    assert rows[-1].event is CampaignEvent.STOPPED
    assert len(backend.calls) == 2


async def test_out_of_policy_proposals_ledger_rejected_until_the_failure_cap(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path / "repo")
    root = tmp_path / "meta"
    bad = make_proposal(1, mutable_paths=("score.py",))
    backend = scripted_backend([bad] * (MAX_PROPOSAL_ATTEMPTS * 2))

    result = await meta.run_campaign(
        make_campaign_policy(max_consecutive_failures=2),
        repo=repo,
        root=root,
        backend=backend,
        driver_factory=forbidden_factory,
    )

    rejected = [row for row in ledger_rows(root) if row.event is CampaignEvent.REJECTED]
    assert [row.seq for row in rejected] == [1, 2]
    assert "outside the template allowlist" in rejected[0].reason
    assert result.halted is not None and "consecutive failed rounds" in result.halted
    round_two_prompt = backend.calls[MAX_PROPOSAL_ATTEMPTS].prompt
    assert "Prior rounds that failed" in round_two_prompt and "outside the template allowlist" in round_two_prompt


async def test_preflight_failure_releases_the_reservation_and_counts_toward_the_stop(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path / "repo")
    root = tmp_path / "meta"
    backend = scripted_backend([make_proposal(1)])

    result = await meta.run_campaign(
        make_campaign_policy(max_consecutive_failures=1),
        repo=repo,
        root=root,
        backend=backend,
        driver_factory=lambda spec: BrokenPreflightDriver(),
    )

    ledger = Ledger.open(root / meta.LEDGER_NAME)
    failed = [row for row in ledger.rows() if row.event is CampaignEvent.PREFLIGHT_FAILED]
    assert len(failed) == 1 and "scorer missing" in failed[0].reason
    assert not ledger.open_reservations()
    assert ledger.total_usd() == 0.0
    assert ledger.experiments_run() == 0
    assert result.halted is not None and "consecutive failed rounds" in result.halted


async def test_infra_abort_does_not_consume_a_candidate_slot_and_counts_toward_the_stop(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path / "repo")
    root = tmp_path / "meta"
    backend = scripted_backend([make_proposal(1)])

    result = await meta.run_campaign(
        make_campaign_policy(max_consecutive_failures=1),
        repo=repo,
        root=root,
        backend=backend,
        driver_factory=lambda spec: InfraDriver(),
    )

    ledger = Ledger.open(root / meta.LEDGER_NAME)
    aborted = [row for row in ledger.rows() if row.event is CampaignEvent.INFRA_ABORTED]
    assert len(aborted) == 1 and "infra retries" in aborted[0].reason
    assert ledger.experiments_run() == 0
    assert not ledger.open_reservations()
    assert result.halted is not None and "consecutive failed rounds" in result.halted
    assert retro_names(root) == []


async def test_gated_mode_never_runs_an_unapproved_spec(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path / "repo")
    root = tmp_path / "meta"
    backend = scripted_backend([make_proposal(1)])

    result = await meta.run_campaign(
        make_campaign_policy(mode="gated"), repo=repo, root=root, backend=backend, driver_factory=forbidden_factory
    )

    assert result.completed == 0 and result.halted is None
    pending = root / meta.PENDING_DIR / "001-round1.toml"
    assert ExperimentSpec.load(pending).name == "001-round1"
    assert [row.event for row in ledger_rows(root)] == [CampaignEvent.PROPOSED, CampaignEvent.PENDING]


async def test_gated_approve_digest_verifies_and_runs_the_identical_codepath(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path / "repo")
    root = tmp_path / "meta"
    policy = make_campaign_policy(mode="gated")
    await meta.run_campaign(
        policy, repo=repo, root=root, backend=scripted_backend([make_proposal(1)]), driver_factory=forbidden_factory
    )

    queued = await meta.approve(root, 1)
    assert queued == root / meta.QUEUE_DIR / "001-round1.toml"

    contracts: list[str] = []
    backend = scripted_backend([make_verdict("gated round one improved"), make_proposal(2)])
    result = await meta.run_campaign(
        policy, repo=repo, root=root, backend=backend, driver_factory=paid_factory([0.9], contracts)
    )

    assert result.completed == 1
    assert not queued.exists()
    assert (root / meta.PENDING_DIR / "002-round2.toml").exists()
    events = [row.event for row in ledger_rows(root)]
    assert events.count(CampaignEvent.APPROVED) == 1
    assert events.count(CampaignEvent.COMPLETED) == 1
    assert retro_names(root) == ["001-round1"]
    assert "gated round one improved" in backend.calls[1].prompt


async def test_approve_refuses_a_drifted_pending_spec(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path / "repo")
    root = tmp_path / "meta"
    await meta.run_campaign(
        make_campaign_policy(mode="gated"),
        repo=repo,
        root=root,
        backend=scripted_backend([make_proposal(1)]),
        driver_factory=forbidden_factory,
    )
    pending = root / meta.PENDING_DIR / "001-round1.toml"
    pending.write_text(pending.read_text().replace("max_usd = 2.0", "max_usd = 9000.0"))

    with pytest.raises(meta.CampaignError, match="digest"):
        await meta.approve(root, 1)
    assert pending.exists()


async def test_reject_removes_the_pending_spec_and_ledgers_why(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path / "repo")
    root = tmp_path / "meta"
    await meta.run_campaign(
        make_campaign_policy(mode="gated"),
        repo=repo,
        root=root,
        backend=scripted_backend([make_proposal(1)]),
        driver_factory=forbidden_factory,
    )

    await meta.reject(root, 1, reason="too costly for tonight")

    assert not (root / meta.PENDING_DIR / "001-round1.toml").exists()
    last = ledger_rows(root)[-1]
    assert last.event is CampaignEvent.REJECTED and last.reason == "operator rejected: too costly for tonight"
    with pytest.raises(meta.CampaignError, match="no pending spec"):
        await meta.reject(root, 1, reason="again")


async def test_ledger_recomputes_totals_seq_and_streak_from_disk(tmp_path: Path) -> None:
    path = tmp_path / meta.LEDGER_NAME
    ledger = Ledger.open(path)
    await ledger.append(LedgerRow(seq=1, event=CampaignEvent.RESERVED, usd=2.0, wall_s=300.0, reason=""))
    await ledger.append(LedgerRow(seq=1, event=CampaignEvent.COMPLETED, usd=1.5, wall_s=4.0, reason=""))
    await ledger.append(LedgerRow(seq=2, event=CampaignEvent.RESERVED, usd=2.0, wall_s=300.0, reason=""))
    await ledger.append(LedgerRow(seq=3, event=CampaignEvent.REJECTED, usd=0.0, wall_s=0.0, reason="bad paths"))

    resumed = Ledger.open(path)
    assert resumed.next_seq() == 4
    assert resumed.total_usd() == 1.5 + 2.0
    assert resumed.total_wall_s() == 4.0 + 300.0
    assert resumed.experiments_run() == 1
    assert resumed.consecutive_failures() == 1
    assert sorted(resumed.open_reservations()) == [2]


def test_ledger_rejects_a_torn_final_line(tmp_path: Path) -> None:
    path = tmp_path / meta.LEDGER_NAME
    intact = json.dumps(LedgerRow(seq=1, event=CampaignEvent.PROPOSED, usd=0.0, wall_s=0.0, reason="").to_record())
    path.write_text(intact + '\n{"seq": 2, "event": "reser')

    with pytest.raises(PoisonedLedger, match="unreadable or malformed"):
        Ledger.open(path)


def test_ledger_rejects_invalid_accounting(tmp_path: Path) -> None:
    path = tmp_path / meta.LEDGER_NAME
    record = LedgerRow(seq=1, event=CampaignEvent.COMPLETED, usd=0.0, wall_s=0.0, reason="").to_record()
    path.write_text(json.dumps(record | {"usd": -1.0}) + "\n")

    with pytest.raises(PoisonedLedger, match="invalid accounting"):
        Ledger.open(path)


async def test_reservation_requires_declared_worst_cases(tmp_path: Path) -> None:
    repo = toy_repo(tmp_path / "repo")
    root = tmp_path / "meta"
    policy = make_campaign_policy(mode="gated")
    await meta.run_campaign(
        policy,
        repo=repo,
        root=root,
        backend=scripted_backend([make_proposal(1, max_usd=None, max_wall_s=None)]),
        driver_factory=forbidden_factory,
    )
    await meta.approve(root, 1)

    result = await meta.run_campaign(
        policy, repo=repo, root=root, backend=scripted_backend([]), driver_factory=forbidden_factory
    )

    assert result.halted is not None and "declares no max_usd/max_wall_s" in result.halted
    assert not any(row.event is CampaignEvent.STARTED for row in ledger_rows(root))


def test_cli_meta_stop_and_report_are_wired(tmp_path: Path) -> None:
    root = tmp_path / "meta"
    runner = CliRunner()

    stop = runner.invoke(cli, ["meta", "stop", "--root", str(root), "--reason", "pause", "--json"])
    assert stop.exit_code == 0, stop.output
    assert json.loads(stop.output)["stop"] == str(root / meta.STOP_NAME)

    report = runner.invoke(cli, ["meta", "report", "--root", str(root), "--json"])
    assert report.exit_code == 0, report.output
    payload = json.loads(report.output)
    assert payload["stop"] == "pause"
    assert payload["next_seq"] == 1 and payload["total_usd"] == 0


def test_cli_meta_install_is_wired(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = toy_repo(tmp_path / "repo")
    policy_path = repo / "policy.toml"
    policy_path.write_text(POLICY_TOML + "\n")
    captured: dict[str, launchd.AgentSpec] = {}

    async def fake_install(agent: launchd.AgentSpec) -> Path:
        captured["agent"] = agent
        return tmp_path / "agent.plist"

    monkeypatch.setattr(launchd, "install", fake_install)
    result = CliRunner().invoke(cli, ["meta", "install", str(policy_path), "--hour", "3", "--json"])

    assert result.exit_code == 0, result.output
    agent = captured["agent"]
    assert agent.label == "com.athome.research.meta.policy"
    assert agent.command == ("athome", "research", "meta", "run", str(policy_path.resolve()))
    assert agent.schedule == launchd.Calendar(hour=3, minute=0)
    assert agent.working_dir == repo
