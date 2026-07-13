from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import anyio

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from athome.ocr.types import Box, OcrToken, TokenOcr

DEDUP_X = 15.0
DEDUP_Y = 10.0
DISAGREEMENT_PENALTY = 0.5


def center(token: OcrToken) -> tuple[float, float]:
    return (token.box.x + token.box.width / 2, token.box.y + token.box.height / 2)


def near(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return abs(a[0] - b[0]) < DEDUP_X and abs(a[1] - b[1]) < DEDUP_Y


def normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


async def gather_pair[A, B](first: Awaitable[A], second: Awaitable[B]) -> tuple[A, B]:
    results: list[object] = [None, None]

    async def run(index: int, awaitable: Awaitable[object]) -> None:
        results[index] = await awaitable

    async with anyio.create_task_group() as group:
        group.start_soon(run, 0, first)
        group.start_soon(run, 1, second)
    return results[0], results[1]


def reconcile(token: OcrToken, supplement: tuple[OcrToken, ...]) -> OcrToken:
    matches = [other for other in supplement if near(center(token), center(other))]
    if not matches or all(normalized(other.text) == normalized(token.text) for other in matches):
        return token
    lowest = min(token.confidence, *(other.confidence for other in matches))
    return replace(token, confidence=lowest * DISAGREEMENT_PENALTY)


def cross_validate(base: tuple[OcrToken, ...], supplement: tuple[OcrToken, ...]) -> tuple[OcrToken, ...]:
    return (
        *(reconcile(token, supplement) for token in base),
        *(token for token in supplement if not any(near(center(token), center(other)) for other in base)),
    )


@dataclass(frozen=True, slots=True)
class EnsembleTokenOcr:
    """Unions a ``primary`` token OCR with a cross-validating ``supplement`` — generic, domain-free.

    On the full-frame pass (no ``region``, ``upscale`` 1.0) both engines read concurrently and their
    tokens union: a supplement token whose box centre is not already covered by a primary token is
    added, and a primary token that a supplement token *contradicts* (same region, different text) is
    kept but flagged low-confidence. Agreement and single-source reads pass through unchanged — the
    primary is authoritative, the supplement only adds or disputes. A region/upscale read (the
    targeted tiles) stays on the primary alone.

    Example:
        >>> EnsembleTokenOcr(PaddleOcr(pool), AppleVision())
    """

    primary: TokenOcr
    supplement: TokenOcr

    async def tokens(self, image: bytes, *, region: Box | None = None, upscale: float = 1.0) -> tuple[OcrToken, ...]:
        if region is not None or upscale != 1.0:
            return await self.primary.tokens(image, region=region, upscale=upscale)
        base, supplement = await gather_pair(self.primary.tokens(image), self.supplement.tokens(image))
        return cross_validate(base, supplement)
