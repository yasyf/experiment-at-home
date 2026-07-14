from __future__ import annotations

import io
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from athome.ocr.paddle import box_to_wire, token_from_wire
from athome.ocr.types import Box, OcrToken
from athome.workers import PipeWorker, WorkerSpec, serve

if TYPE_CHECKING:
    from athome.wire import Wire
    from athome.workers import WorkerTransport

APPLE_RECOGNITION_LEVEL = "accurate"
APPLE_METHOD = "tokens"
APPLE_BOOTSTRAP = "from athome.ocr.apple import serve_apple; serve_apple()"
APPLE_SPEC = WorkerSpec((sys.executable, "-c", APPLE_BOOTSTRAP))


def to_box(bbox: tuple[float, float, float, float], upscale: float, offset: tuple[int, int]) -> Box:
    left, top, right, bottom = bbox
    dx, dy = offset
    # Round the true corners, never origin and size independently: consumers rebuild the far
    # edge as x + width, so a separately rounded size drifts that edge off the float box.
    x, y = round(left / upscale) + dx, round(top / upscale) + dy
    return Box(x=x, y=y, width=round(right / upscale) + dx - x, height=round(bottom / upscale) + dy - y)


def to_token(
    annotation: tuple[str, float, tuple[float, float, float, float]], upscale: float, offset: tuple[int, int]
) -> OcrToken:
    text, confidence, bbox = annotation
    # ocrmac hands back objc.pyobjc_unicode, a str subclass; OcrToken.text is an exact str.
    return OcrToken(text=str(text), box=to_box(bbox, upscale, offset), confidence=confidence)


def recognize_tokens(image: bytes, region: Box | None, upscale: float) -> tuple[OcrToken, ...]:
    from ocrmac import ocrmac
    from PIL import Image
    from PIL.Image import Resampling

    pil = Image.open(io.BytesIO(image))
    cropped = pil.crop((region.x, region.y, region.x + region.width, region.y + region.height)) if region else pil
    offset = (region.x, region.y) if region else (0, 0)
    if upscale != 1.0:
        cropped = cropped.resize((round(cropped.width * upscale), round(cropped.height * upscale)), Resampling.LANCZOS)
    return tuple(
        to_token(annotation, upscale, offset)
        for annotation in ocrmac.OCR(cropped, recognition_level=APPLE_RECOGNITION_LEVEL).recognize(px=True)
    )


def token_to_wire(token: OcrToken) -> Wire:
    return {"text": token.text, "box": box_to_wire(token.box), "confidence": token.confidence}


@dataclass(frozen=True, slots=True)
class AppleHandler:
    def tokens(self, payload: Wire) -> Wire:
        region = None if payload["region"] is None else Box(**payload["region"])
        return [token_to_wire(token) for token in recognize_tokens(payload["image"], region, payload["upscale"])]


def serve_apple() -> None:
    serve(AppleHandler())


@dataclass(frozen=True, slots=True)
class AppleVision:
    """Token OCR via the macOS Vision framework (``ocrmac``, the ``ocr`` extra).

    ``ocrmac`` is a GIL-bound Objective-C bridge, so it never runs in the free-threaded
    parent: each read is dispatched over ``worker`` — a :class:`~athome.workers.PipeWorker`
    subprocess that imports ``ocrmac`` in its own process — and the reply is decoded into
    line-level tokens with pixel boxes in full-frame, top-left-origin coordinates even when a
    ``region`` restricts recognition. ``upscale`` Lanczos-magnifies the crop to recover text too
    small for native-resolution OCR, then divides recovered coordinates back down.

    Example:
        >>> await AppleVision().tokens(jpeg, region=Box(0, 0, 200, 996))
    """

    worker: WorkerTransport = field(default_factory=lambda: PipeWorker(APPLE_SPEC))

    async def tokens(self, image: bytes, *, region: Box | None = None, upscale: float = 1.0) -> tuple[OcrToken, ...]:
        payload = {"image": image, "region": box_to_wire(region), "upscale": upscale}
        reply = await self.worker.call(APPLE_METHOD, payload)
        return tuple(token_from_wire(token) for token in reply)

    async def aclose(self) -> None:
        await self.worker.aclose()
