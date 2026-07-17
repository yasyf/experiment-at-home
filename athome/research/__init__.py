"""The autoresearch harness: karpathy's greedy keep/discard loop primitives, athome-styled.

Isolation boundary and threat model: ``docs/design/research-security-model.md``.
"""

from __future__ import annotations

from athome.research.errors import ResearchError
from athome.research.gate import PromotionVerdict, blocking_invariants, bootstrap_ci_gate, monotone_gate
from athome.research.journal import Journal, JournalRow, Verdict
from athome.research.policy import CampaignBudget, ExperimentTemplate, PolicyViolation, ProposalPolicy
from athome.research.propose import (
    Proposal,
    ProposalRound,
    ProposalViolation,
    ProposerContext,
    propose,
    validate_proposal,
)
from athome.research.registry import RegistryError, VersionInfo, current, promote, register, versions
from athome.research.spec import (
    Budget,
    BudgetExhausted,
    Comparability,
    ExperimentSpec,
    ImmutableViolation,
    UnknownSpecField,
)
