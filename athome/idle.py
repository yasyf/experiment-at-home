"""A generic idle-unload lifecycle: single-flight load, reference-counted use, TTL reaper.

An :class:`IdleResource` wraps an expensive-to-hold value — a loaded model, a spawned
server — behind ``load``/``unload`` callables. Concurrent first uses collapse to one
load under a lock; every :meth:`use` counts as in-flight so the reaper never evicts a
value a caller still holds; and a background :meth:`run` loop unloads the value once it
has sat idle past ``ttl_s``. The value reference is dropped before ``unload`` awaits, so a
consumer whose eviction only frees once its value is unreferenced (MLX weight caches) sees
a clean drop.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import anyio
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

DEFAULT_REAP_INTERVAL_S = 30.0


@dataclass(slots=True)
class IdleResource[T]:
    """A lazily-loaded value that unloads itself once idle past its TTL.

    ``load`` is awaited at most once per loaded lifetime, under a lock, so concurrent first
    uses trigger exactly one load; a load that raises leaves the resource unloaded and the
    next :meth:`use` retries fresh. ``unload`` closes over the consumer's own state — the
    resource holds only the loaded ``T`` — and runs from the reaper (:meth:`sweep` /
    :meth:`run`) once no use is in flight and the idle window has elapsed, or immediately
    from :meth:`discard`.

    Attributes:
        load: Awaited to produce the value; called once per loaded lifetime.
        unload: Awaited to release the consumer's state after the value reference is dropped.
        ttl_s: Idle seconds a loaded value must sit unused before the reaper unloads it.

    Example:
        >>> resource = IdleResource(spawn_server, teardown_server, ttl_s=300.0)
        >>> async with resource.use() as server:
        ...     await server.query(prompt)
    """

    load: Callable[[], Awaitable[T]]
    unload: Callable[[], Awaitable[None]]
    ttl_s: float = field(kw_only=True)
    lock: anyio.Lock = field(default_factory=anyio.Lock, init=False)
    value: T | None = field(default=None, init=False)
    inflight: int = field(default=0, init=False)
    last_done: float = field(default=0.0, init=False)

    @property
    def loaded(self) -> bool:
        """Whether a value is currently held."""
        return self.value is not None

    @asynccontextmanager
    async def use(self) -> AsyncIterator[T]:
        """Yield the loaded value, loading it first if necessary and counting the use in flight.

        Loads under the lock on first use so concurrent callers share one load. The in-flight
        count is held for the body's duration, blocking the reaper; on exit it records the
        completion time against the monotonic clock so the idle window starts from the last
        release.
        """
        if (value := self.value) is None:
            async with self.lock:
                if (value := self.value) is None:
                    value = self.value = await self.load()
        self.inflight += 1
        try:
            yield value
        finally:
            self.inflight -= 1
            self.last_done = anyio.current_time()

    async def sweep(self, *, now: float | None = None) -> None:
        """Unload the value iff it is loaded, no use is in flight, and it has been idle past ``ttl_s``.

        The value reference is dropped before ``unload`` is awaited, with no backup local ref
        (a ref would pin MLX weights and defeat eviction). So a failed ``unload`` leaves the
        resource unloaded while the underlying resource is possibly still alive; the next
        ``load`` must tolerate that by re-adopting or reloading it.

        Args:
            now: The monotonic time to measure the idle window against; resolves the clock itself
                when omitted.
        """
        clock = anyio.current_time() if now is None else now
        if self.value is None or self.inflight > 0 or clock - self.last_done < self.ttl_s:
            return
        # Drop the value reference before awaiting unload: an MLX weight cache only frees once
        # its value object is unreferenced as the consumer's cache clear runs.
        self.value = None
        await self.unload()

    async def run(self, *, interval_s: float = DEFAULT_REAP_INTERVAL_S) -> None:
        """Reaper loop: every ``interval_s`` seconds, sweep the resource. Runs until cancelled.

        Each iteration isolates its own sweep: a ``sweep`` that raises — a failed ``unload``,
        most likely — is logged and the loop keeps running, so one transient failure never
        silently kills the reaper. Cancellation still ends the loop: anyio raises it as a
        ``BaseException``, which ``except Exception`` cannot catch.
        """
        while True:
            await anyio.sleep(interval_s)
            try:
                await self.sweep()
            except Exception:
                logger.exception("idle reaper sweep failed; continuing")

    async def discard(self) -> None:
        """Drop the value and await ``unload`` now, bypassing both the TTL and the in-flight count.

        For a value known bad: in-flight holders keep their own local references and fail on
        their own, so the reaper's guards are deliberately skipped here.
        """
        self.value = None
        await self.unload()
