"""The research-harness error root; every research module subclasses it."""

from __future__ import annotations

from athome.errors import AthomeError


class ResearchError(AthomeError):
    """Root of every athome research-harness error (a blocking preflight or loop failure)."""
