# athome-ocr-paddle

The PP-OCRv6 detection+recognition sidecar for athome's token-level OCR ensemble.

onnxruntime publishes no free-threaded (`cp314t`) wheel, so this engine cannot import into athome's 3.14t
runtime. It runs as a separate uv project on a GIL-enabled 3.13 interpreter. athome spawns it via
`uvx athome-ocr-paddle` and round-trips frames over the vendored (stdlib-only) length-prefixed wire protocol on
stdin/stdout, so it is a drop-in `athome.workers.PipeWorker` sidecar. The package carries no `athome` import —
`athome_ocr_paddle/wire.py` is a file copy of `athome/wire.py` (the wire vendoring contract), and
`athome_ocr_paddle/serve.py` mirrors `athome.workers.serve`.

## Wire contract

The handler exposes two methods reached through the serve loop:

- `tokens(payload)` where `payload = (jpeg: bytes, region: [x, y, width, height] | None, upscale: float)` →
  `list[{"text": str, "confidence": float, "box": [x, y, width, height]}]` in original-image pixel coordinates.
- `fingerprint()` → `{"versions": {...}, "det_params": {...}, "revisions": {"det", "rec"}}`, emitted in the
  handshake and cross-checked by the Modal relay for parity.

## Modal backend

`athome_ocr_paddle/modal_app.py` hosts the identical RapidOCR engine as an autoscaling CPU service; deploy with
`uv run modal deploy -m athome_ocr_paddle.modal_app`. `athome_ocr_paddle/modal_client.py`
(`athome-ocr-paddle-modal`) is a wire-identical relay: it asserts the remote fingerprint matches the local pins
before serving and dies loudly if Modal is unreachable — no local fallback. The remote app name defaults to
`athome-ocr-paddle`, overridable via `ATHOME_MODAL_OCR_APP`.
