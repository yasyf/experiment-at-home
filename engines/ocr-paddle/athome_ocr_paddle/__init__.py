"""GIL-enabled PP-OCRv6 sidecar for athome's token-level OCR ensemble.

onnxruntime ships no free-threaded (``cp314t``) wheel, so PP-OCRv6 cannot import into
athome's 3.14t runtime. This is a separate ``athome-ocr-paddle`` dist on a GIL-enabled
3.13 interpreter: a persistent process that builds the RapidOCR PP-OCRv6 engine, then
serves length-prefixed wire frames over stdin/stdout via the vendored
:func:`athome_ocr_paddle.serve.serve` loop — a drop-in :class:`athome.workers.PipeWorker`
sidecar. It carries no ``athome`` import; tokens cross the boundary as wire dicts the
parent (``athome.ocr.paddle.PaddleOcr``) re-wraps as ``OcrToken``s. Run via
``uvx athome-ocr-paddle``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import TYPE_CHECKING

from athome_ocr_paddle.serve import serve

if TYPE_CHECKING:
    from athome_ocr_paddle.wire import Wire

PPOCR_DET_REPO = "PaddlePaddle/PP-OCRv6_tiny_det_onnx"
PPOCR_REC_REPO = "PaddlePaddle/PP-OCRv6_small_rec_onnx"
# Pinned HF snapshot commits: the build and every runtime container resolve the exact ONNX weights,
# never HEAD, and the pins are stamped into the fingerprint so the Modal relay fails loud on skew.
PPOCR_DET_REVISION = "2ba1506c0380b8f0b03dd142459aac66d4421f6c"
PPOCR_REC_REVISION = "b8f84f0b80c529de40b4fbb3544b84fa7233a513"
# Dense-text detection tuning carried from the PaddleOCR text-detection doc: min-side upscaling to a
# 960 floor plus a 1.8 unclip band for tightly stacked rows. Folded into the parity fingerprint.
PPOCR_DET_PARAMS = {"Det.limit_type": "min", "Det.limit_side_len": 960, "Det.unclip_ratio": 1.8}
VERSION_PACKAGES = ("onnxruntime", "rapidocr", "pillow", "numpy")
REC_KEYS_TMP_PREFIX = "rec_keys."
REC_KEYS_TMP_SUFFIX = ".tmp"
STALE_TMP_SECONDS = 3600.0


def sweep_rec_keys_tmps(rec_dir: str) -> None:
    import os
    import time

    cutoff = time.time() - STALE_TMP_SECONDS
    for name in os.listdir(rec_dir):
        if not (name.startswith(REC_KEYS_TMP_PREFIX) and name.endswith(REC_KEYS_TMP_SUFFIX)):
            continue
        sibling = os.path.join(rec_dir, name)
        try:
            if os.stat(sibling).st_mtime < cutoff:
                os.remove(sibling)
        except OSError:
            continue


def rec_dict_path(rec_dir: str) -> str:
    import os
    import tempfile

    import yaml

    with open(f"{rec_dir}/inference.yml") as handle:
        cfg = yaml.safe_load(handle)
    path = f"{rec_dir}/rec_keys.txt"
    if os.path.exists(path):
        return path
    sweep_rec_keys_tmps(rec_dir)
    fd, tmp = tempfile.mkstemp(dir=rec_dir, prefix=REC_KEYS_TMP_PREFIX, suffix=REC_KEYS_TMP_SUFFIX)
    with os.fdopen(fd, "w") as handle:
        handle.write("\n".join(str(char) for char in cfg["PostProcess"]["character_dict"]))
    os.replace(tmp, path)
    return path


def build_engine() -> object:
    from huggingface_hub import snapshot_download
    from rapidocr import EngineType, RapidOCR

    det = snapshot_download(PPOCR_DET_REPO, revision=PPOCR_DET_REVISION) + "/inference.onnx"
    rec_dir = snapshot_download(PPOCR_REC_REPO, revision=PPOCR_REC_REVISION)
    return RapidOCR(
        params={
            "EngineConfig.onnxruntime.use_coreml": False,
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.model_path": det,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.model_path": rec_dir + "/inference.onnx",
            "Rec.rec_keys_path": rec_dict_path(rec_dir),
            "Cls.engine_type": EngineType.ONNXRUNTIME,
            **PPOCR_DET_PARAMS,
        }
    )


def box_of(poly: object) -> tuple[float, float, float, float]:
    xs = [float(point[0]) for point in poly]
    ys = [float(point[1]) for point in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def to_wire_token(poly: object, text: object, score: object, *, upscale: float, dx: float, dy: float) -> Wire:
    left, top, right, bottom = (
        coord / upscale + delta for coord, delta in zip(box_of(poly), (dx, dy, dx, dy), strict=True)
    )
    return {
        "text": str(text),
        "confidence": float(score),
        "box": [round(left), round(top), round(right - left), round(bottom - top)],
    }


def recognize(engine: object, jpeg: bytes, region: Wire, upscale: float) -> Wire:
    import numpy as np
    from PIL import Image
    from PIL.Image import Resampling

    image = Image.open(io.BytesIO(jpeg)).convert("RGB")
    dx, dy, width, height = region or (0, 0, image.width, image.height)
    cropped = image.crop((dx, dy, dx + width, dy + height))
    if upscale > 1.0:
        cropped = cropped.resize((round(cropped.width * upscale), round(cropped.height * upscale)), Resampling.LANCZOS)
    result = engine(np.asarray(cropped))
    if getattr(result, "boxes", None) is None:
        return []
    return [
        to_wire_token(poly, text, score, upscale=upscale, dx=dx, dy=dy)
        for poly, text, score in zip(result.boxes, result.txts, result.scores, strict=True)
    ]


def warmup(engine: object) -> None:
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (200, 60), (255, 255, 255))
    ImageDraw.Draw(canvas).text((12, 20), "12,345", fill=(0, 0, 0))
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG")
    recognize(engine, buffer.getvalue(), None, 1.0)


def fingerprint() -> Wire:
    import importlib.metadata

    return {
        "versions": {name: importlib.metadata.version(name) for name in VERSION_PACKAGES},
        "det_params": PPOCR_DET_PARAMS,
        "revisions": {"det": PPOCR_DET_REVISION, "rec": PPOCR_REC_REVISION},
    }


@dataclass(frozen=True, slots=True)
class PaddleHandler:
    """Wire handler over a built PP-OCRv6 engine: one ``tokens`` method plus the parity fingerprint.

    Example:
        >>> serve(PaddleHandler(build_engine()))
    """

    engine: object

    def tokens(self, payload: Wire) -> Wire:
        jpeg, region, upscale = payload
        return recognize(self.engine, jpeg, region, upscale)

    def fingerprint(self) -> Wire:
        return fingerprint()


def main() -> None:
    engine = build_engine()
    warmup(engine)
    serve(PaddleHandler(engine))
