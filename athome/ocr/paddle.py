from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from athome.ocr.types import Box, OcrToken

if TYPE_CHECKING:
    from athome.wire import Wire
    from athome.workers import WorkerPool

PADDLE_METHOD = "tokens"


def digest_hex(image: bytes) -> str:
    return hashlib.blake2b(image, digest_size=16).hexdigest()


def box_to_wire(region: Box | None) -> Wire:
    return None if region is None else {"x": region.x, "y": region.y, "width": region.width, "height": region.height}


def token_from_wire(raw: Wire) -> OcrToken:
    return OcrToken(text=raw["text"], box=Box(**raw["box"]), confidence=raw["confidence"])


@dataclass(frozen=True, slots=True)
class PaddleOcr:
    """Token OCR via PP-OCRv6, hosted in the ``athome-ocr-paddle`` sidecar dist over a worker pool.

    Each read leases a :class:`~athome.workers.PipeWorker` from ``pool`` — keyed by the image
    digest so a prefetched frame lands on its already-warm process — and calls the sidecar's
    ``tokens`` handler, decoding the wire reply into full-frame ``OcrToken``s. The sidecar is a
    separate PyPI dist (``uvx athome-ocr-paddle``), never in athome's own dependencies.

    Example:
        >>> pool = WorkerPool(WorkerSpec(("uvx", "athome-ocr-paddle")), size=4)
        >>> await PaddleOcr(pool).tokens(jpeg)
    """

    pool: WorkerPool

    async def tokens(self, image: bytes, *, region: Box | None = None, upscale: float = 1.0) -> tuple[OcrToken, ...]:
        async with self.pool.lease(digest_hex(image)) as worker:
            reply = await worker.call(PADDLE_METHOD, {"jpeg": image, "region": box_to_wire(region), "upscale": upscale})
        return tuple(token_from_wire(token) for token in reply)
