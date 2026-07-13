from __future__ import annotations

from typing import ClassVar


class AthomeError(Exception):
    """Root of every athome error; the CLI renders these as clean stderr + exit 1."""

    exit_code: ClassVar[int] = 1
