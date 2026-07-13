from __future__ import annotations

from dataclasses import dataclass, field

from anyio import Lock

from athome.errors import AthomeError


class SpendExceeded(AthomeError):
    """Raised when projected + recorded spend would cross a :class:`SpendGuard`'s ``max_usd``."""


@dataclass(slots=True)
class SpendGuard:
    """Aborts loudly when projected + running cost crosses ``max_usd``.

    Reservations are atomic under an :class:`anyio.Lock`: :meth:`check` reserves the
    projected cost before the call so concurrent callers cannot all pass against the
    same balance, and :meth:`record` reconciles that reservation with the actual spend.

    Example:
        >>> guard = SpendGuard(max_usd=1.0)
        >>> await guard.check(0.4)            # reserve the projection
        >>> await guard.record(0.4, 0.35)     # release the reservation, commit the actual
    """

    max_usd: float
    spent: float = 0.0
    reserved: float = 0.0
    lock: Lock = field(default_factory=Lock)

    async def check(self, projected: float) -> None:
        """Reserve ``projected`` atomically, raising :class:`SpendExceeded` if it would cross ``max_usd``."""
        async with self.lock:
            if (total := self.spent + self.reserved + projected) > self.max_usd:
                raise SpendExceeded(f"projected ${total:.4f} exceeds cap ${self.max_usd:.4f}")
            self.reserved += projected

    async def record(self, reserved: float, actual: float) -> None:
        """Release the ``reserved`` projection and add ``actual`` to the running spend total, atomically."""
        async with self.lock:
            self.reserved -= reserved
            self.spent += actual
