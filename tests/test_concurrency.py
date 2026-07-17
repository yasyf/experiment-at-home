from __future__ import annotations

import anyio

from athome.concurrency import gather_bounded


async def test_gather_bounded_preserves_input_order() -> None:
    n = 5

    async def make(index: int) -> int:
        await anyio.sleep((n - index) * 0.01)
        return index

    assert await gather_bounded([lambda i=i: make(i) for i in range(n)], concurrency=n) == list(range(n))


async def test_gather_bounded_caps_concurrent_tasks() -> None:
    running = {"now": 0, "peak": 0}

    async def occupy() -> None:
        running["now"] += 1
        running["peak"] = max(running["peak"], running["now"])
        await anyio.sleep(0.01)
        running["now"] -= 1

    await gather_bounded([occupy for _ in range(10)], concurrency=3)
    assert running["peak"] <= 3
