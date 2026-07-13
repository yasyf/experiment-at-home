from __future__ import annotations

import importlib.metadata
import io
import subprocess
import sys
from pathlib import Path

import pytest
from athome_ocr_paddle import (
    PPOCR_DET_PARAMS,
    PPOCR_DET_REVISION,
    PPOCR_REC_REVISION,
    VERSION_PACKAGES,
    PaddleHandler,
    box_of,
    fingerprint,
    recognize,
)
from athome_ocr_paddle import wire as paddle_wire
from athome_ocr_paddle.modal_client import parity_mismatches
from athome_ocr_paddle.serve import dispatch, handler_fingerprint
from athome_ocr_paddle.wire import WIRE_VERSION, encode

SERVE_SOURCE = (
    "from athome_ocr_paddle.serve import serve\n"
    "class Fake:\n"
    "    def tokens(self, payload):\n"
    "        jpeg, region, upscale = payload\n"
    "        return [{'text': 'ok', 'confidence': 1.0, 'box': [0, 0, int(upscale), len(jpeg)]}]\n"
    "    def fingerprint(self):\n"
    "        return {'engine': 'fake'}\n"
    "serve(Fake())\n"
)


class FakeResult:
    def __init__(self, boxes: object, txts: object, scores: object) -> None:
        self.boxes = boxes
        self.txts = txts
        self.scores = scores


class FakeEngine:
    def __init__(self, result: FakeResult) -> None:
        self.result = result

    def __call__(self, array: object) -> FakeResult:
        return self.result


def jpeg_bytes(width: int, height: int) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(buffer, format="JPEG")
    return buffer.getvalue()


def text_jpeg(text: str) -> bytes:
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (240, 80), (255, 255, 255))
    ImageDraw.Draw(canvas).text((16, 28), text, fill=(0, 0, 0))
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG")
    return buffer.getvalue()


def pipe_call(cmd: list[str], requests: list[tuple[str, object]]) -> tuple[object, list[object]]:
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        handshake = paddle_wire.read_frame(proc.stdout)
        replies = []
        for method, payload in requests:
            proc.stdin.write(encode({"method": method, "payload": payload}))
            proc.stdin.flush()
            replies.append(paddle_wire.read_frame(proc.stdout))
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)
    return handshake, replies


def test_box_of_bounds_polygon() -> None:
    assert box_of([(10, 5), (50, 5), (50, 20), (10, 20)]) == (10.0, 5.0, 50.0, 20.0)


def test_recognize_maps_full_image_tokens() -> None:
    engine = FakeEngine(FakeResult([[(10, 5), (50, 5), (50, 20), (10, 20)]], ["12,345"], [0.97]))
    assert recognize(engine, jpeg_bytes(120, 40), None, 1.0) == [
        {"text": "12,345", "confidence": 0.97, "box": [10, 5, 40, 15]}
    ]


def test_recognize_maps_region_and_upscale_back_to_original_coords() -> None:
    engine = FakeEngine(FakeResult([[(20, 10), (60, 10), (60, 30), (20, 30)]], ["px"], [0.5]))
    assert recognize(engine, jpeg_bytes(400, 200), [100, 50, 80, 30], 2.0) == [
        {"text": "px", "confidence": 0.5, "box": [110, 55, 20, 10]}
    ]


def test_recognize_returns_empty_when_no_boxes() -> None:
    assert recognize(FakeEngine(FakeResult(None, None, None)), jpeg_bytes(120, 40), None, 1.0) == []


def test_fingerprint_shape_and_pins() -> None:
    fp = fingerprint()
    assert set(fp) == {"versions", "det_params", "revisions"}
    assert fp["det_params"] == PPOCR_DET_PARAMS
    assert fp["revisions"] == {"det": PPOCR_DET_REVISION, "rec": PPOCR_REC_REVISION}
    assert set(fp["versions"]) == set(VERSION_PACKAGES)


def test_handler_tokens_dispatch_ok() -> None:
    handler = PaddleHandler(FakeEngine(FakeResult([[(0, 0), (4, 0), (4, 6), (0, 6)]], ["hi"], [0.8])))
    frame = {"method": "tokens", "payload": (jpeg_bytes(40, 20), None, 1.0)}
    assert dispatch(handler, frame) == {"ok": [{"text": "hi", "confidence": 0.8, "box": [0, 0, 4, 6]}]}


def test_handler_tokens_dispatch_surfaces_error() -> None:
    reply = dispatch(PaddleHandler(FakeEngine(FakeResult(None, None, None))), {"method": "tokens", "payload": None})
    assert "err" in reply
    assert "TypeError" in reply["err"]


def test_dispatch_rejects_malformed_frame() -> None:
    assert dispatch(object(), {"unexpected": 1}) == {"err": "malformed request frame: {'unexpected': 1}"}


def test_handler_fingerprint_passthrough() -> None:
    fp = handler_fingerprint(PaddleHandler(FakeEngine(FakeResult(None, None, None))))
    assert fp["revisions"] == {"det": PPOCR_DET_REVISION, "rec": PPOCR_REC_REVISION}


def test_serve_loop_roundtrips_over_subprocess() -> None:
    handshake, replies = pipe_call([sys.executable, "-c", SERVE_SOURCE], [("tokens", (b"abc", None, 3.0))])
    assert handshake == {"wire": WIRE_VERSION, "fingerprint": {"engine": "fake"}}
    assert replies == [{"ok": [{"text": "ok", "confidence": 1.0, "box": [0, 0, 3, 3]}]}]


def test_vendored_wire_is_byte_identical_to_athome() -> None:
    upstream = Path(__file__).resolve().parents[3] / "athome" / "wire.py"
    if not upstream.exists():
        pytest.skip("athome/wire.py not present outside the monorepo")
    assert Path(paddle_wire.__file__).read_bytes() == upstream.read_bytes()


def local_fingerprint() -> dict:
    return {
        "versions": {name: importlib.metadata.version(name) for name in VERSION_PACKAGES},
        "det_params": PPOCR_DET_PARAMS,
        "revisions": {"det": PPOCR_DET_REVISION, "rec": PPOCR_REC_REVISION},
    }


def test_parity_mismatches_empty_when_aligned() -> None:
    assert parity_mismatches(local_fingerprint()) == []


def test_parity_mismatches_flags_version_revision_and_params() -> None:
    skewed = local_fingerprint()
    skewed["versions"][VERSION_PACKAGES[0]] = "0.0.0-skew"
    skewed["revisions"]["det"] = "deadbeef"
    skewed["det_params"] = {"Det.limit_type": "max"}
    mismatches = parity_mismatches(skewed)
    assert any(VERSION_PACKAGES[0] in line and "0.0.0-skew" in line for line in mismatches)
    assert any("revision[det]" in line for line in mismatches)
    assert any("det_params" in line for line in mismatches)


def test_modal_app_declares_named_app() -> None:
    pytest.importorskip("modal")
    from athome_ocr_paddle import modal_app

    assert modal_app.app.name == "athome-ocr-paddle"
    assert modal_app.PaddleOcrRemote.__name__ == "PaddleOcrRemote"


@pytest.mark.live
def test_real_sidecar_subprocess_reads_text() -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("rapidocr")
    handshake, replies = pipe_call(
        [sys.executable, "-m", "athome_ocr_paddle"], [("tokens", (text_jpeg("12,345"), None, 3.0))]
    )
    assert handshake["wire"] == WIRE_VERSION
    tokens = replies[0]["ok"]
    assert any("345" in token["text"] for token in tokens)
