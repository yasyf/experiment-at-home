"""Bounded structured concurrency helpers shared across athome subsystems."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import anyio

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence


async def gather_bounded[T](tasks: Sequence[Callable[[], Awaitable[T]]], *, concurrency: int) -> list[T]:
    """Run the task factories under a capacity limiter, preserving input order in the results.

    Example:
        >>> await gather_bounded([lambda: fetch(url) for url in urls], concurrency=8)
    """
    results: list[T | None] = [None] * len(tasks)
    limiter = anyio.CapacityLimiter(concurrency)

    async def one(index: int) -> None:
        async with limiter:
            results[index] = await tasks[index]()

    async with anyio.create_task_group() as group:
        for index in range(len(tasks)):
            group.start_soon(one, index)
    return cast("list[T]", results)
