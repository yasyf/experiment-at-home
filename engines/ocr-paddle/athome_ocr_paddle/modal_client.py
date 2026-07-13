"""Wire-compatible sidecar whose inference is served by the Modal-hosted PP-OCRv6.

A drop-in for the local ``athome-ocr-paddle`` sidecar: it speaks the same vendored wire protocol
(:func:`athome_ocr_paddle.serve.serve`) over stdin/stdout, so an ``athome.workers.PipeWorker`` parent cannot tell
it apart. Instead of building a local RapidOCR engine, each request round-trips a ``(jpeg, region, upscale)`` tuple
to :class:`athome_ocr_paddle.modal_app.PaddleOcrRemote`. On startup it calls the remote ``fingerprint`` — forcing
client hydration so a missing token or undeployed app raises *here*, before the handshake — then asserts the
remote's onnxruntime/rapidocr/pillow/numpy versions, pinned model revisions, and det params match the local pins,
raising a :class:`RuntimeError` on any skew. No local fallback: if Modal is unreachable the sidecar dies, exactly
as a crashed local engine would. Run via ``athome-ocr-paddle-modal``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from athome_ocr_paddle import PPOCR_DET_PARAMS, PPOCR_DET_REVISION, PPOCR_REC_REVISION, VERSION_PACKAGES
from athome_ocr_paddle.serve import serve

if TYPE_CHECKING:
    from athome_ocr_paddle.wire import Wire

APP_ENV = "ATHOME_MODAL_OCR_APP"
DEFAULT_APP = "athome-ocr-paddle"
REMOTE_CLASS = "PaddleOcrRemote"


def parity_mismatches(remote: dict) -> list[str]:
    import importlib.metadata

    revisions = {"det": PPOCR_DET_REVISION, "rec": PPOCR_REC_REVISION}
    return (
        [
            f"{name}: local {local} != remote {remote['versions'].get(name)}"
            for name in VERSION_PACKAGES
            if (local := importlib.metadata.version(name)) != remote["versions"].get(name)
        ]
        + [
            f"revision[{model}]: local {rev} != remote {remote['revisions'].get(model)}"
            for model, rev in revisions.items()
            if rev != remote["revisions"].get(model)
        ]
        + (
            []
            if remote["det_params"] == PPOCR_DET_PARAMS
            else [f"det_params: local {PPOCR_DET_PARAMS} != remote {remote['det_params']}"]
        )
    )


@dataclass(frozen=True, slots=True)
class ModalPaddleHandler:
    """Wire handler that relays each ``tokens`` call to the Modal-hosted engine, serving its parity fingerprint."""

    remote: object
    finger: dict

    def tokens(self, payload: Wire) -> Wire:
        jpeg, region, upscale = payload
        return self.remote.recognize.remote(jpeg, region, upscale)

    def fingerprint(self) -> Wire:
        return self.finger


def main() -> None:
    import modal

    remote = modal.Cls.from_name(os.environ.get(APP_ENV, DEFAULT_APP), REMOTE_CLASS)()
    finger = remote.fingerprint.remote()
    if mismatches := parity_mismatches(finger):
        raise RuntimeError("modal ocr sidecar parity mismatch: " + "; ".join(mismatches))
    serve(ModalPaddleHandler(remote, finger))


if __name__ == "__main__":
    main()
