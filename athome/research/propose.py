"""The outer loop's LLM proposer: structured proposals admitted only by refusal.

A :class:`Proposal` has no slot for any executable field — the metric command and
key, direction, and immutable paths are copied verbatim from the operator template
it names. What it may choose (a name, allowlisted mutable paths, budget numbers) is
validated by exact membership and operator ceilings; every violation raises, and
nothing is clamped or normalized. Free text flows only into the contract's
``## Hypothesis`` section and the audit record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from athome.detach import NAME_RE
from athome.research.errors import ResearchError
from athome.research.judge import with_backoff
from athome.research.policy import ExperimentTemplate, ProposalPolicy
from athome.research.spec import Budget, ExperimentSpec

if TYPE_CHECKING:
    from spawnllm import LlmBackend, TModel

    from athome.registry import VersionInfo
    from athome.research.cells import Cell
    from athome.research.nightly import MorningReport

MAX_PROPOSAL_ATTEMPTS = 3
REJECTION_HEADER = "## Your previous proposal was rejected"


class ProposalViolation(ResearchError):
    """A proposal broke the policy: named and refused, never clamped or normalized."""


class Proposal(BaseModel):
    """The proposer's entire authority, as its structured-output schema.

    Executable fields (metric command and key, direction, immutable paths) have no
    slot here, so a proposal cannot even express them; :func:`validate_proposal`
    copies them verbatim from the chosen template. Extra fields are refused by the
    model itself.

    Attributes:
        template: The name of the operator template to instantiate.
        name: The experiment name; the harness prefixes the ledger sequence number.
        mutable_paths: The globs the candidate may edit; each must be an exact
            member of the template's allowlist.
        max_units: Work-unit budget, at most the operator ceiling.
        max_usd: Spend backstop; required in auto mode, at most the ceiling.
        max_wall_s: Wall-clock backstop; required in auto mode, at most the ceiling.
        hard_kill_s: Per-iteration hard-kill timeout, at most the ceiling.
        hypothesis: Free text rendered into the contract's ``## Hypothesis`` section
            and the audit record — never a shell, path, or key.
    """

    model_config = ConfigDict(extra="forbid")

    template: str
    name: str
    mutable_paths: tuple[str, ...]
    max_units: int
    max_usd: float | None = None
    max_wall_s: float | None = None
    hard_kill_s: float | None = None
    hypothesis: str


@dataclass(frozen=True, slots=True)
class ProposerContext:
    """The sanitized channel the proposer reads: harness-authored records, never run-log bytes.

    Attributes:
        retros: Rendered retro documents from prior experiments, oldest first.
        reports: Morning reports of prior runs (their stats, not their logs).
        leaderboard: The campaign's best cells.
        current: The registry's currently promoted version, if any.
        failures: Harness-written reasons for prior rejected or failed rounds.
    """

    retros: tuple[str, ...] = ()
    reports: tuple[MorningReport, ...] = ()
    leaderboard: tuple[Cell, ...] = ()
    current: VersionInfo | None = None
    failures: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProposalRound:
    """One admitted proposal and its audit trail.

    Attributes:
        spec: The validated spec, built through :class:`ExperimentSpec`.
        proposal: The raw proposal the spec was admitted from.
        attempts: Extract calls spent (1 plus up to ``MAX_PROPOSAL_ATTEMPTS - 1`` re-prompts).
    """

    spec: ExperimentSpec
    proposal: Proposal
    attempts: int


def bounded(label: str, proposed: float | None, ceiling: float | None) -> None:
    if proposed is None:
        return
    if ceiling is None:
        raise ProposalViolation(f"{label}={proposed} has no operator ceiling to bound it")
    if not 0 < proposed <= ceiling:
        raise ProposalViolation(f"{label}={proposed} must be positive and at most the operator ceiling {ceiling}")


def validate_proposal(proposal: Proposal, policy: ProposalPolicy, *, seq: int) -> ExperimentSpec:
    """Admit a proposal by refusal: every violation raises, and nothing is clamped.

    The admitted spec's executable fields are copied verbatim from the named
    template; ``mutable_paths`` must be exact members of the template's allowlist
    (string equality, never glob subsumption); every budget number must sit at or
    under its operator ceiling, with ``max_usd`` and ``max_wall_s`` mandatory in
    auto mode; and the name must fully match ``NAME_RE`` before the harness-supplied
    ledger ``seq`` prefixes it. Construction runs through
    :class:`~athome.research.spec.ExperimentSpec`, so its unbounded-glob rejection
    fires as defense in depth.

    Args:
        proposal: The proposer's structured output.
        policy: The frozen operator policy to validate against.
        seq: The ledger sequence number the experiment name is prefixed with.

    Raises:
        ProposalViolation: any policy violation, with the specific refusal reason.
    """
    templates = {template.name: template for template in policy.templates}
    if (template := templates.get(proposal.template)) is None:
        raise ProposalViolation(f"unknown template {proposal.template!r}; the policy offers {sorted(templates)}")
    if not NAME_RE.fullmatch(proposal.name):
        raise ProposalViolation(f"name {proposal.name!r} must fully match {NAME_RE.pattern}")
    if not proposal.mutable_paths:
        raise ProposalViolation("mutable_paths is empty: a proposal must name the files it may edit")
    if outside := [path for path in proposal.mutable_paths if path not in template.mutable_allowlist]:
        raise ProposalViolation(f"mutable_paths outside the template allowlist (exact membership only): {outside}")
    if policy.mode == "auto" and (proposal.max_usd is None or proposal.max_wall_s is None):
        raise ProposalViolation("auto mode requires max_usd and max_wall_s: reservations use declared caps")
    for label, proposed, ceiling in (
        ("max_units", proposal.max_units, policy.ceilings.max_units),
        ("max_wall_s", proposal.max_wall_s, policy.ceilings.max_wall_s),
        ("hard_kill_s", proposal.hard_kill_s, policy.ceilings.hard_kill_s),
        ("max_usd", proposal.max_usd, policy.ceilings.max_usd),
    ):
        bounded(label, proposed, ceiling)
    return ExperimentSpec(
        name=f"{seq:03d}-{proposal.name}",
        metric_command=template.metric_command,
        metric_key=template.metric_key,
        direction=template.direction,
        mutable_paths=proposal.mutable_paths,
        immutable_paths=template.immutable_paths,
        budget=Budget(
            max_units=proposal.max_units,
            max_wall_s=proposal.max_wall_s,
            hard_kill_s=proposal.hard_kill_s,
            max_usd=proposal.max_usd,
        ),
        metric_file=template.metric_file,
        hypothesis=proposal.hypothesis,
    )


def render_report(report: MorningReport) -> str:
    best = f"best kept metric {report.best.metric}" if report.best else "nothing kept"
    return f"- {report.experiment}: {report.units} units, {report.kept} kept, {report.crashes} crashed; {best}"


def render_cell(cell: Cell) -> str:
    return f"- {cell.experiment} unit {cell.unit} [{cell.arm}]: {cell.metric} ({cell.verdict})"


def render_template(template: ExperimentTemplate) -> str:
    return (
        f"### {template.name}\n"
        f"- goal: {template.direction} `{template.metric_key}` via `{' '.join(template.metric_command)}`\n"
        f"- mutable allowlist (pick exact entries): {list(template.mutable_allowlist)}"
    )


def numbered(items: tuple[str, ...]) -> str:
    return "\n\n".join(f"[{index}] {item}" for index, item in enumerate(items, start=1))


def build_prompt(policy: ProposalPolicy, context: ProposerContext) -> str:
    """Renders the proposer prompt: policy bounds plus the sanitized campaign history."""
    ceilings = policy.ceilings
    sections = [
        "# Propose the next experiment",
        "You are the research proposer. Pick one template, name the experiment, choose mutable_paths "
        "as exact entries of that template's allowlist, declare budgets at or under the ceilings, and "
        "state the hypothesis your experiment tests. Out-of-policy proposals are rejected, not repaired.",
        f"## Ceilings\nmax_units <= {ceilings.max_units}, max_wall_s <= {ceilings.max_wall_s}, "
        f"hard_kill_s <= {ceilings.hard_kill_s}, max_usd <= {ceilings.max_usd}"
        + (" (max_usd and max_wall_s are required)" if policy.mode == "auto" else ""),
        "## Templates\n" + "\n\n".join(render_template(template) for template in policy.templates),
    ]
    if context.current:
        sections.append(f"## Currently promoted\n{context.current.name} {context.current.version}")
    if context.reports:
        sections.append("## Prior runs\n" + "\n".join(render_report(report) for report in context.reports))
    if context.leaderboard:
        sections.append("## Leaderboard\n" + "\n".join(render_cell(cell) for cell in context.leaderboard))
    if context.retros:
        sections.append("## Retros of prior experiments\n" + numbered(context.retros))
    if context.failures:
        sections.append("## Prior rounds that failed\n" + "\n".join(f"- {reason}" for reason in context.failures))
    return "\n\n".join(sections)


def rejected_prompt(prompt: str, violation: ProposalViolation) -> str:
    return f"{prompt}\n\n{REJECTION_HEADER}\n{violation}\nFix exactly this violation and propose again."


async def propose(
    policy: ProposalPolicy,
    context: ProposerContext,
    *,
    backend: LlmBackend,
    seq: int,
    tier: TModel = "large",
    timeout: int = 240,
) -> ProposalRound:
    """Ask the bound backend for the next experiment and admit it through :func:`validate_proposal`.

    A validation failure is appended to the prompt and re-asked, at most
    ``MAX_PROPOSAL_ATTEMPTS`` extract calls in total; the last violation propagates
    when every attempt is refused. The backend is concretely bound — the call never
    reaches spawnllm's auto-select branch — and transient backend errors retry
    under :func:`~athome.research.judge.with_backoff`.

    Args:
        policy: The frozen operator policy proposals are validated against.
        context: The sanitized campaign history the proposer reads.
        backend: The concrete spawnllm backend the extract call runs on.
        seq: The ledger sequence number for the admitted experiment's name prefix.
        tier: The abstract spawnllm model tier to run at.
        timeout: Seconds before the backend process is killed.

    Raises:
        ProposalViolation: every attempt was refused; carries the last refusal reason.
        JudgeError: the backend call failed after its transient-error retries.
    """
    from spawnllm import extract

    async def ask(prompt: str, *, attempt: int) -> Proposal:
        return await with_backoff(
            lambda: extract(prompt, Proposal, backend=backend, model=tier, timeout=timeout),
            label=f"propose[{attempt}]",
        )

    prompt = build_prompt(policy, context)
    for attempt in range(1, MAX_PROPOSAL_ATTEMPTS):
        proposal = await ask(prompt, attempt=attempt)
        try:
            spec = validate_proposal(proposal, policy, seq=seq)
        except ProposalViolation as violation:
            prompt = rejected_prompt(prompt, violation)
            continue
        return ProposalRound(spec=spec, proposal=proposal, attempts=attempt)
    proposal = await ask(prompt, attempt=MAX_PROPOSAL_ATTEMPTS)
    return ProposalRound(
        spec=validate_proposal(proposal, policy, seq=seq), proposal=proposal, attempts=MAX_PROPOSAL_ATTEMPTS
    )
