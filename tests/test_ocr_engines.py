from __future__ import annotations

import base64
import io
import json
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx
import pytest
from click.testing import CliRunner

from athome.config import load
from athome.ocr import vlm
from athome.ocr.apple import APPLE_SPEC, AppleVision, to_box, to_token, token_to_wire
from athome.ocr.ensemble import DISAGREEMENT_PENALTY, EnsembleTokenOcr, cross_validate, near
from athome.ocr.merge import LlmMerger, merge_prompt
from athome.ocr.paddle import PaddleOcr, box_to_wire, digest_hex, token_from_wire
from athome.ocr.profiles import (
    OcrError,
    OcrSettings,
    cli,
    document_from_bytes,
    document_to_bytes,
    documents_agree,
    layout_markdown,
    read,
)
from athome.ocr.types import Box, Document, OcrToken
from athome.ocr.vlm import image_media_type
from athome.serve import ServerHandle
from athome.wire import decode, encode

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


def token(text: str, x: int, confidence: float = 1.0, *, y: int = 0, width: int = 40, height: int = 10) -> OcrToken:
    return OcrToken(text=text, box=Box(x=x, y=y, width=width, height=height), confidence=confidence)


def words_tokens(text: str) -> tuple[OcrToken, ...]:
    return tuple(token(word, x=index * 50) for index, word in enumerate(text.split()))


@dataclass
class RecordingTokenOcr:
    result: tuple[OcrToken, ...]
    calls: list[tuple[Box | None, float]] = field(default_factory=list)

    async def tokens(self, image: bytes, *, region: Box | None = None, upscale: float = 1.0) -> tuple[OcrToken, ...]:
        self.calls.append((region, upscale))
        return self.result


@dataclass
class RaisingTokenOcr:
    async def tokens(self, image: bytes, *, region: Box | None = None, upscale: float = 1.0) -> tuple[OcrToken, ...]:
        raise AssertionError("supplement must not run on a region/upscale read")


@dataclass
class FakeTransport:
    reply: object
    calls: list[tuple[str, object]] = field(default_factory=list)

    async def call(self, method: str, payload: object) -> object:
        self.calls.append((method, payload))
        return self.reply

    async def aclose(self) -> None: ...


@dataclass
class FakePool:
    transport: FakeTransport
    keys: list[str | None] = field(default_factory=list)

    @asynccontextmanager
    async def lease(self, key: str | None = None) -> AsyncIterator[FakeTransport]:
        self.keys.append(key)
        yield self.transport


# ----- ensemble -----


async def test_ensemble_unions_and_keeps_agreement() -> None:
    primary = RecordingTokenOcr((token("hello", 0, 0.9), token("world", 100, 0.9)))
    supplement = RecordingTokenOcr((token("hello", 3, 0.95, y=1), token("extra", 200, 0.8)))
    result = await EnsembleTokenOcr(primary, supplement).tokens(b"img")
    assert [(t.text, t.confidence) for t in result] == [("hello", 0.9), ("world", 0.9), ("extra", 0.8)]


async def test_ensemble_flags_disagreement_low() -> None:
    primary = RecordingTokenOcr((token("100", 0, 0.9),))
    supplement = RecordingTokenOcr((token("700", 3, 0.8, y=1),))
    (only,) = await EnsembleTokenOcr(primary, supplement).tokens(b"img")
    assert only.text == "100"
    assert only.confidence == pytest.approx(min(0.9, 0.8) * DISAGREEMENT_PENALTY)


@pytest.mark.parametrize(
    ("region", "upscale"),
    [(Box(1, 2, 3, 4), 1.0), (None, 2.0)],
    ids=["region", "upscale"],
)
async def test_ensemble_targeted_read_is_primary_only(region: Box | None, upscale: float) -> None:
    primary = RecordingTokenOcr((token("only", 0),))
    result = await EnsembleTokenOcr(primary, RaisingTokenOcr()).tokens(b"img", region=region, upscale=upscale)
    assert result == (token("only", 0),)
    assert primary.calls == [(region, upscale)]


def test_cross_validate_adds_uncovered_supplement() -> None:
    base = (token("a", 0),)
    supplement = (token("a", 2), token("b", 300))
    assert [t.text for t in cross_validate(base, supplement)] == ["a", "b"]


def test_cross_validate_penalizes_contradiction_beside_agreement() -> None:
    base = (token("100", 0, 0.9),)
    supplement = (token("100", 3, 0.95, y=1), token("700", 5, 0.8, y=1))
    (only,) = cross_validate(base, supplement)
    assert only.text == "100"
    assert only.confidence == pytest.approx(min(0.9, 0.95, 0.8) * DISAGREEMENT_PENALTY)


def test_near_is_center_proximity() -> None:
    assert near((10.0, 10.0), (20.0, 15.0))
    assert not near((10.0, 10.0), (40.0, 10.0))


# ----- apple -----


class PyObjcUnicode(str):
    pass


@pytest.mark.parametrize(
    ("bbox", "upscale", "offset", "expected"),
    [
        ((10.0, 20.0, 30.0, 50.0), 1.0, (0, 0), Box(10, 20, 20, 30)),
        ((10.0, 20.0, 30.0, 50.0), 2.0, (5, 7), Box(10, 17, 10, 15)),
        ((0.0, 0.7, 100.0, 210.281), 1.0, (0, 0), Box(0, 1, 100, 209)),
    ],
    ids=["native", "upscaled-offset", "fractional-corners"],
)
def test_to_box_descales_and_offsets(
    bbox: tuple[float, float, float, float], upscale: float, offset: tuple[int, int], expected: Box
) -> None:
    assert to_box(bbox, upscale, offset) == expected


def test_to_box_far_edge_lands_on_the_rounded_float_corner() -> None:
    box = to_box((0.0, 0.7, 100.0, 210.281), 1.0, (0, 0))
    assert box.y + box.height == 210


def test_to_token_coerces_ocrmac_text_to_exact_str() -> None:
    tok = to_token((PyObjcUnicode("HELLO"), 0.9, (10.0, 20.0, 30.0, 50.0)), 1.0, (0, 0))
    assert tok == OcrToken(text="HELLO", box=Box(10, 20, 20, 30), confidence=0.9)
    assert type(tok.text) is str


def test_apple_token_crosses_the_wire() -> None:
    tok = to_token((PyObjcUnicode("HELLO"), 0.9, (10.0, 20.0, 30.0, 50.0)), 1.0, (0, 0))
    assert token_from_wire(decode(encode(token_to_wire(tok)))) == tok


async def test_apple_routes_tokens_through_worker() -> None:
    worker = FakeTransport([{"text": "HELLO", "box": {"x": 1, "y": 2, "width": 3, "height": 4}, "confidence": 0.9}])
    result = await AppleVision(worker=worker).tokens(b"png", region=Box(5, 6, 7, 8), upscale=2.0)  # type: ignore[arg-type]
    assert result == (OcrToken(text="HELLO", box=Box(1, 2, 3, 4), confidence=0.9),)
    assert worker.calls == [
        ("tokens", {"image": b"png", "region": {"x": 5, "y": 6, "width": 7, "height": 8}, "upscale": 2.0})
    ]


def test_apple_runs_out_of_process_without_importing_ocrmac() -> None:
    assert APPLE_SPEC.command[:2] == (sys.executable, "-c")
    assert "athome.ocr.apple" in APPLE_SPEC.command[2]
    assert "serve_apple" in APPLE_SPEC.command[2]
    assert "ocrmac" not in sys.modules


# ----- paddle -----


async def test_paddle_roundtrips_wire_tokens() -> None:
    transport = FakeTransport([{"text": "A", "box": {"x": 1, "y": 2, "width": 3, "height": 4}, "confidence": 0.9}])
    pool = FakePool(transport)
    result = await PaddleOcr(pool).tokens(b"jpeg")  # type: ignore[arg-type]
    assert result == (OcrToken(text="A", box=Box(1, 2, 3, 4), confidence=0.9),)
    assert transport.calls == [("tokens", {"jpeg": b"jpeg", "region": None, "upscale": 1.0})]
    assert pool.keys == [digest_hex(b"jpeg")]


async def test_paddle_encodes_region() -> None:
    transport = FakeTransport([])
    await PaddleOcr(FakePool(transport)).tokens(b"jpeg", region=Box(5, 6, 7, 8), upscale=2.0)  # type: ignore[arg-type]
    assert transport.calls[0][1] == {
        "jpeg": b"jpeg",
        "region": {"x": 5, "y": 6, "width": 7, "height": 8},
        "upscale": 2.0,
    }


def test_box_to_wire_none() -> None:
    assert box_to_wire(None) is None


# ----- vlm -----


async def fake_ensure(self: object, *, persistent: bool = False) -> ServerHandle:
    return ServerHandle(recipe="mlx-vlm", port=8401, pid=None, base_url="http://127.0.0.1:8401/v1", api_key="local")


async def test_vlm_read_posts_image_and_parses_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "# Heading\n\ntext"}}]})

    monkeypatch.setenv("ATHOME_SERVE_MLX_VLM_VERSION", "0.3.4")
    load.cache_clear()
    monkeypatch.setattr(vlm.ManagedServer, "ensure", fake_ensure)
    monkeypatch.setattr(vlm, "endpoint_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    document = await vlm.VlmOcr().read(b"\x89PNG\r\n\x1a\nbody")
    assert document.markdown == "# Heading\n\ntext"
    body = captured[0]
    assert body["model"] == "mlx-community/dots.ocr-4bit"
    content = body["messages"][0]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_image_media_type_sniffs() -> None:
    assert image_media_type(b"\x89PNG\r\n\x1a\n...") == "image/png"
    assert image_media_type(b"\xff\xd8\xff\xe0rest") == "image/jpeg"


def test_image_media_type_rejects_unknown() -> None:
    with pytest.raises(OcrError):
        image_media_type(b"not-an-image")


def test_vlm_model_rejects_non_vision_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATHOME_SERVE_LLAMA_SERVER_COMMAND", "llama-server -m m.gguf")
    load.cache_clear()
    with pytest.raises(OcrError):
        vlm.vlm_model("llama-server")


# ----- merge -----


async def test_merge_feeds_image_to_vision_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[str, list[dict[str, object]]]] = []

    async def fake_complete(recipe: str, messages: list[dict[str, object]]) -> str:
        captured.append((recipe, messages))
        return "MERGED"

    monkeypatch.setattr("athome.ocr.vlm.complete", fake_complete)
    image = b"\x89PNG\r\n\x1a\nPIXELS"
    document = await LlmMerger().merge(image, (Document("aaa"), Document("bbb")))
    assert document.markdown == "MERGED"
    recipe, messages = captured[0]
    assert recipe == "mlx-vlm"
    content = messages[0]["content"]
    text = next(part["text"] for part in content if part["type"] == "text")
    assert "aaa" in text and "bbb" in text
    image_uri = next(part["image_url"]["url"] for part in content if part["type"] == "image_url")
    assert base64.b64encode(image).decode() in image_uri


def test_merge_prompt_numbers_candidates() -> None:
    prompt = merge_prompt((Document("first"), Document("second")))
    assert "Reading 1" in prompt and "Reading 2" in prompt


# ----- profiles: layout / agreement / serialization -----


def test_layout_markdown_reading_order() -> None:
    tokens = (
        token("world", 200, y=0),
        token("hello", 0, y=0),
        token("next", 0, y=40),
    )
    assert layout_markdown(tokens) == "hello world\nnext"


@pytest.mark.parametrize(
    ("reference", "candidate", "expected"),
    [
        ("hello world foo", "hello world foo", True),
        ("hello world foo bar", "zzz qqq", False),
        ("", "anything", False),
        ("anything", "", False),
        ("", "", True),
    ],
    ids=["identical", "divergent", "reference-empty", "candidate-empty", "both-empty"],
)
def test_documents_agree(reference: str, candidate: str, expected: bool) -> None:
    assert documents_agree(Document(reference), Document(candidate)) is expected


def test_document_bytes_roundtrip() -> None:
    document = Document("# Title", (token("x", 3, 0.7),))
    assert document_from_bytes(document_to_bytes(document)) == document


# ----- profiles: quality dispatch (merge-only-on-disagreement) -----


def install_quality_engines(
    monkeypatch: pytest.MonkeyPatch, *, vlm_markdown: str, apple_text: str
) -> list[tuple[bytes, tuple[Document, ...]]]:
    merges: list[tuple[bytes, tuple[Document, ...]]] = []

    class FakeVlm:
        async def read(self, image: bytes) -> Document:
            return Document(vlm_markdown)

    class FakeApple:
        async def tokens(
            self, image: bytes, *, region: Box | None = None, upscale: float = 1.0
        ) -> tuple[OcrToken, ...]:
            return words_tokens(apple_text)

        async def aclose(self) -> None: ...

    class FakeMerger:
        async def merge(self, image: bytes, candidates: tuple[Document, ...]) -> Document:
            merges.append((image, candidates))
            return Document("MERGED")

    monkeypatch.setattr("athome.ocr.profiles.VlmOcr", FakeVlm)
    monkeypatch.setattr("athome.ocr.profiles.AppleVision", FakeApple)
    monkeypatch.setattr("athome.ocr.profiles.LlmMerger", FakeMerger)
    return merges


async def test_quality_returns_vlm_on_agreement(monkeypatch: pytest.MonkeyPatch) -> None:
    merges = install_quality_engines(monkeypatch, vlm_markdown="hello world foo", apple_text="hello world foo")
    document = await read(b"\x89PNG\r\n\x1a\nx", profile="quality")
    assert document.markdown == "hello world foo"
    assert merges == []


async def test_quality_merges_on_disagreement(monkeypatch: pytest.MonkeyPatch) -> None:
    merges = install_quality_engines(monkeypatch, vlm_markdown="hello world foo", apple_text="zzz qqq nnn")
    document = await read(b"\x89PNG\r\n\x1a\ny", profile="quality")
    assert document.markdown == "MERGED"
    assert len(merges) == 1


async def test_quality_merges_when_vlm_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    merges = install_quality_engines(monkeypatch, vlm_markdown="", apple_text="hello world foo")
    document = await read(b"\x89PNG\r\n\x1a\nz", profile="quality")
    assert document.markdown == "MERGED"
    assert len(merges) == 1


# ----- profiles: dispatch + cache keying -----


@pytest.fixture
def counting_profiles(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, int]]:
    counts = {"realtime": 0, "quality": 0}

    def make(name: str) -> object:
        async def profile_read(image: bytes) -> Document:
            counts[name] += 1
            return Document(f"{name}:{image.decode()}")

        return profile_read

    monkeypatch.setattr("athome.ocr.profiles.read_realtime", make("realtime"))
    monkeypatch.setattr("athome.ocr.profiles.read_quality", make("quality"))
    yield counts


async def test_read_caches_by_image_and_profile(counting_profiles: dict[str, int]) -> None:
    first = await read(b"A", profile="quality")
    second = await read(b"A", profile="quality")
    assert first.markdown == second.markdown == "quality:A"
    assert counting_profiles["quality"] == 1


async def test_read_profile_is_part_of_the_key(counting_profiles: dict[str, int]) -> None:
    await read(b"A", profile="quality")
    await read(b"A", profile="realtime")
    assert counting_profiles == {"realtime": 1, "quality": 1}


async def test_read_defaults_profile_from_settings(
    counting_profiles: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ATHOME_OCR_PROFILE", "realtime")
    load.cache_clear()
    assert load(OcrSettings).profile == "realtime"
    await read(b"Z")
    assert counting_profiles["realtime"] == 1


# ----- cli -----


def test_cli_prints_markdown(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    source = tmp_path / "page.png"  # type: ignore[operator]
    source.write_bytes(b"\x89PNG\r\n\x1a\nx")

    async def fake_read(image: bytes, *, profile: str | None = None) -> Document:
        return Document("# Result", (token("x", 1, 0.5),))

    monkeypatch.setattr("athome.ocr.profiles.read", fake_read)
    plain = CliRunner().invoke(cli, [str(source)])
    assert plain.exit_code == 0
    assert "# Result" in plain.output
    rich = CliRunner().invoke(cli, [str(source), "--json"])
    assert json.loads(rich.output)["markdown"] == "# Result"


# ----- live smokes (skipped in the mocked run) -----


@pytest.mark.live
async def test_live_apple_vision_reads_text() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (400, 120), "white")
    ImageDraw.Draw(image).text((10, 40), "HELLO 123", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    tokens = await AppleVision().tokens(buffer.getvalue())
    assert any("HELLO" in tok.text or "123" in tok.text for tok in tokens)


@pytest.mark.live
async def test_live_dots_ocr_reads_page() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (600, 200), "white")
    ImageDraw.Draw(image).text((20, 80), "Invoice Total 42.00", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    document = await vlm.VlmOcr().read(buffer.getvalue())
    assert document.markdown.strip()
