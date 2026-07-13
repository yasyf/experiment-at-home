from __future__ import annotations

from dataclasses import dataclass

from athome.errors import AthomeError


class SpendExceeded(AthomeError):
    """Raised when projected + recorded spend would cross a :class:`SpendGuard`'s ``max_usd``."""


@dataclass(slots=True)
class SpendGuard:
    """Aborts loudly when projected + running cost crosses ``max_usd``.

    Example:
        >>> guard = SpendGuard(max_usd=1.0)
        >>> guard.check(0.4)
        >>> guard.record(0.4)
    """

    max_usd: float
    spent: float = 0.0

    def check(self, projected: float) -> None:
        """Raise :class:`SpendExceeded` when ``spent + projected`` would exceed ``max_usd``."""
        if self.spent + projected > self.max_usd:
            raise SpendExceeded(f"projected ${self.spent + projected:.4f} exceeds cap ${self.max_usd:.4f}")

    def record(self, actual: float) -> None:
        """Add ``actual`` to the running spend total."""
        self.spent += actual
