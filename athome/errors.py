from __future__ import annotations

from typing import ClassVar


class AthomeError(Exception):
    """Root of every athome error; the CLI renders these as clean stderr + exit 1."""

    exit_code: ClassVar[int] = 1


class ResearchError(AthomeError):
    """Root of every athome research-harness error (a blocking preflight or loop failure).

    Defined here, not in :mod:`athome.research.errors`, so that :mod:`athome.registry` — which
    the research harness imports — can subclass it without importing the research package back.
    :mod:`athome.research.errors` re-exports it as the harness's public name.
    """
