"""The research-harness error root; every research module subclasses it."""

from __future__ import annotations

from athome.errors import ResearchError as ResearchError


class AccountingIntegrityError(ResearchError):
    """Proposal spend could not be recovered or trusted; ledger reconciliation is required."""
