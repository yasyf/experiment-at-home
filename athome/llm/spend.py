from __future__ import annotations

import math
from dataclasses import dataclass, field

from anyio import Lock

from athome.errors import AthomeError


class SpendExceeded(AthomeError):
    """Raised when projected + recorded spend would cross a :class:`SpendGuard`'s ``max_usd``."""


class InvalidBudget(AthomeError):
    """Raised when a :class:`SpendGuard` is built with a non-finite or negative ``max_usd``."""


@dataclass(slots=True)
class SpendGuard:
    """Aborts loudly when projected + running cost crosses ``max_usd``.

    Reservations are atomic under an :class:`anyio.Lock`: :meth:`check` reserves the
    projected cost before the call so concurrent callers cannot all pass against the same
    balance, and :meth:`record` reconciles that reservation with the actual spend.
    :meth:`check` always reserves — accounting is uniform — but refuses with
    :class:`SpendExceeded` only when ``max_usd`` is a finite cap.

    ``max_usd`` of ``None`` is the deliberate spelling of an unbounded envelope: it never
    refuses, yet still accumulates ``spent`` for reporting. ``None`` must only ever appear
    as an explicit argument, never as anyone's default — omission and float arithmetic can
    never produce it. A float ``max_usd`` must be finite and non-negative (``0.0`` is the
    legal spelling of spend-nothing); anything else raises :class:`InvalidBudget`.

    A single guard threaded through ``fit`` and then ``score`` composes both calls under
    one cap: ``fit``'s recorded actuals draw down what ``score`` may still reserve.

    A non-athome billable call debits the same envelope by pre-authorizing the projection
    and reconciling the actual:

        >>> await budget.check(projected)          # reserve before the external call
        >>> await budget.record(projected, actual)  # reconcile after it returns

    A call that fails instead reconciles with ``release(projected)``, freeing the
    reservation so the cap is not permanently consumed. For a truly unprojectable,
    already-incurred cost, ``record(0.0, actual)`` debits it post-hoc.

    Example:
        >>> guard = SpendGuard(max_usd=1.0)
        >>> await guard.check(0.4)            # reserve the projection
        >>> await guard.record(0.4, 0.35)     # release the reservation, commit the actual
    """

    max_usd: float | None
    spent: float = 0.0
    reserved: float = 0.0
    lock: Lock = field(default_factory=Lock)

    def __post_init__(self) -> None:
        if self.max_usd is not None and (not math.isfinite(self.max_usd) or self.max_usd < 0.0):
            raise InvalidBudget(f"max_usd must be finite and non-negative, got {self.max_usd!r}")

    async def check(self, projected: float) -> None:
        """Reserve ``projected`` atomically, raising :class:`SpendExceeded` if a finite ``max_usd`` would be crossed."""
        async with self.lock:
            if self.max_usd is not None and (total := self.spent + self.reserved + projected) > self.max_usd:
                raise SpendExceeded(f"projected ${total:.4f} exceeds cap ${self.max_usd:.4f}")
            self.reserved += projected

    async def record(self, reserved: float, actual: float) -> None:
        """Release the ``reserved`` projection and add ``actual`` to the running spend total, atomically."""
        async with self.lock:
            self.reserved -= reserved
            self.spent += actual

    async def release(self, reserved: float) -> None:
        """Release the ``reserved`` projection without recording spend, atomically.

        The failure/cancel counterpart to :meth:`record`: a call that never completes
        frees its reservation so the cap is not permanently consumed.
        """
        async with self.lock:
            self.reserved -= reserved
