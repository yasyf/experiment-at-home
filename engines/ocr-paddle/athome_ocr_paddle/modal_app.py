"""Modal-hosted PP-OCRv6: the same RapidOCR engine the local sidecar builds, run as an autoscaling CPU service.

Every model file is baked into the image at build time (:func:`download_models`) so a container never writes at
runtime, and the engine is built + warmed once per container in :meth:`PaddleOcrRemote.build`. The class reuses
``build_engine`` / ``warmup`` / ``recognize`` from :mod:`athome_ocr_paddle` verbatim — the relay
(:mod:`athome_ocr_paddle.modal_client`) cross-checks :meth:`PaddleOcrRemote.fingerprint` against the local pins so
a version skew fails loudly before serving. Deploy from the ``engines/ocr-paddle`` project root::

    uv run modal deploy -m athome_ocr_paddle.modal_app
"""

from __future__ import annotations

import modal

from athome_ocr_paddle import PPOCR_DET_REPO, PPOCR_DET_REVISION, PPOCR_REC_REPO, PPOCR_REC_REVISION, fingerprint

HF_HUB_CACHE = "/models/hf"

app = modal.App("athome-ocr-paddle")


def download_models() -> None:
    from huggingface_hub import snapshot_download

    from athome_ocr_paddle import rec_dict_path

    snapshot_download(PPOCR_DET_REPO, revision=PPOCR_DET_REVISION)
    rec_dict_path(snapshot_download(PPOCR_REC_REPO, revision=PPOCR_REC_REVISION))


image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_sync()
    .env({"HF_HUB_CACHE": HF_HUB_CACHE})
    .add_local_python_source("athome_ocr_paddle", copy=True)
    .run_function(download_models)
    # Runtime containers resolve models purely from the baked cache — never the Hub — so a pinned-revision
    # cache miss crashes loudly instead of silently pulling a newer snapshot. Set after the build download.
    .env({"HF_HUB_OFFLINE": "1"})
)


@app.cls(
    image=image,
    cpu=4.0,
    memory=4096,
    max_containers=64,
    scaledown_window=120,
    timeout=60,
    retries=modal.Retries(max_retries=2, initial_delay=1.0),
)
class PaddleOcrRemote:
    """One PP-OCRv6 inference per container; the autoscaler provides width, never in-container concurrency."""

    @modal.enter()
    def build(self) -> None:
        from athome_ocr_paddle import build_engine, warmup

        self.engine = build_engine()
        warmup(self.engine)

    @modal.method()
    def recognize(self, jpeg: bytes, region: object, upscale: float) -> object:
        from athome_ocr_paddle import recognize

        return recognize(self.engine, jpeg, region, upscale)

    @modal.method()
    def fingerprint(self) -> dict:
        return fingerprint()
