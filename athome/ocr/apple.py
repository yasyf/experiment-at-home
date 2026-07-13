from __future__ import annotations

import io
from dataclasses import dataclass

import anyio

from athome.ocr.types import Box, OcrToken

APPLE_RECOGNITION_LEVEL = "accurate"


def to_box(bbox: tuple[float, float, float, float], upscale: float, offset: tuple[int, int]) -> Box:
    left, top, right, bottom = bbox
    dx, dy = offset
    return Box(
        x=round(left / upscale) + dx,
        y=round(top / upscale) + dy,
        width=round((right - left) / upscale),
        height=round((bottom - top) / upscale),
    )


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
        OcrToken(text=text, box=to_box(bbox, upscale, offset), confidence=confidence)
        for text, confidence, bbox in ocrmac.OCR(cropped, recognition_level=APPLE_RECOGNITION_LEVEL).recognize(px=True)
    )


@dataclass(frozen=True, slots=True)
class AppleVision:
    """Token OCR via the macOS Vision framework (``ocrmac``, the ``ocr`` extra).

    The accurate recognizer runs in a worker thread — ``ocrmac`` is a blocking
    Objective-C bridge with no async API — and returns line-level tokens with pixel
    boxes in full-frame, top-left-origin coordinates even when a ``region`` restricts
    recognition. ``upscale`` Lanczos-magnifies the crop to recover text too small for
    native-resolution OCR, then divides recovered coordinates back down.

    Example:
        >>> await AppleVision().tokens(jpeg, region=Box(0, 0, 200, 996))
    """

    async def tokens(self, image: bytes, *, region: Box | None = None, upscale: float = 1.0) -> tuple[OcrToken, ...]:
        return await anyio.to_thread.run_sync(recognize_tokens, image, region, upscale)
