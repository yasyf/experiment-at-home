from __future__ import annotations

import array
import json
import sys
from typing import TYPE_CHECKING

import httpx
import pytest
from click.testing import CliRunner

# starlette + python-multipart ship in the `stt` extra; skip cleanly on an extra-less CI sync.
pytest.importorskip("starlette")
pytest.importorskip("multipart")

from athome import serve
from athome.stt import catalog
from athome.stt.catalog import VARIANTS
from athome.stt.cli import cli, transcript_payload
from athome.stt.server import (
    SttServer,
    SttServeSettings,
    duration_of,
    verbose_payload,
)
from athome.stt.types import Segment, SttError, Transcript, Word

if TYPE_CHECKING:
    from pathlib import Path

CANNED = Transcript(
    text="the quick brown fox",
    segments=(
        Segment(text="the quick", start=0.0, end=1.0),
        Segment(text="brown fox", start=1.0, end=2.0),
    ),
    words=(
        Word(text="the", start=0.0, end=0.3),
        Word(text="quick", start=0.3, end=1.0),
        Word(text="brown", start=1.0, end=1.5),
        Word(text="fox", start=1.5, end=2.0),
    ),
    load_ms=42.0,
)


class FakeResource:
    def __init__(self) -> None:
        self.loaded = False
        self.discarded = False

    async def run(self) -> None:
        import anyio

        await anyio.sleep_forever()

    async def discard(self) -> None:
        self.discarded = True


class FakeTranscriber:
    def __init__(self, *, transcript: Transcript | None = None, fail: Exception | None = None) -> None:
        self.transcript = transcript
        self.fail = fail
        self.resource = FakeResource()
        self.calls: list[object] = []

    async def transcribe(self, pcm: object) -> Transcript:
        self.calls.append(pcm)
        if self.fail is not None:
            raise self.fail
        assert self.transcript is not None
        return self.transcript


def build_server(
    monkeypatch: pytest.MonkeyPatch,
    transcript: Transcript | None = None,
    *,
    fail: Exception | None = None,
    variant: str | None = None,
) -> SttServer:
    from athome.stt import server as server_mod

    settings = SttServeSettings(variant=variant) if variant is not None else SttServeSettings()
    server = SttServer(settings)
    server.transcriber = FakeTranscriber(transcript=transcript, fail=fail)  # type: ignore[assignment]

    async def fake_decode(data: bytes) -> array.array:
        return array.array("f", [0.0, 0.0, 0.0])

    monkeypatch.setattr(server_mod, "decode", fake_decode)
    return server


def asgi_client(server: SttServer) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://stt")


def audio_files() -> dict[str, tuple[str, bytes, str]]:
    return {"file": ("clip.wav", b"RIFF\x00\x00\x00\x00WAVE", "audio/wav")}


# --- pure helpers ----------------------------------------------------------


def test_duration_of_prefers_segment_ends_then_words_then_zero() -> None:
    assert duration_of(CANNED) == 2.0
    words_only = Transcript(text="hi", segments=(), words=(Word(text="hi", start=0.0, end=0.9),), load_ms=1.0)
    assert duration_of(words_only) == 0.9
    assert duration_of(Transcript(text="", segments=(), words=(), load_ms=0.0)) == 0.0


def test_verbose_payload_shape() -> None:
    payload = verbose_payload(CANNED)
    assert payload["task"] == "transcribe"
    assert payload["duration"] == 2.0
    assert payload["text"] == "the quick brown fox"
    assert payload["segments"] == [
        {"id": 0, "start": 0.0, "end": 1.0, "text": "the quick"},
        {"id": 1, "start": 1.0, "end": 2.0, "text": "brown fox"},
    ]
    assert payload["words"][0] == {"word": "the", "start": 0.0, "end": 0.3}


# --- response shapes -------------------------------------------------------


async def test_transcription_json_is_the_default_format(monkeypatch: pytest.MonkeyPatch) -> None:
    server = build_server(monkeypatch, CANNED)
    async with asgi_client(server) as client:
        response = await client.post("/v1/audio/transcriptions", files=audio_files())
    assert response.status_code == 200
    assert response.json() == {"text": "the quick brown fox"}


async def test_transcription_verbose_json_carries_segments_and_words(monkeypatch: pytest.MonkeyPatch) -> None:
    server = build_server(monkeypatch, CANNED)
    async with asgi_client(server) as client:
        response = await client.post(
            "/v1/audio/transcriptions", files=audio_files(), data={"response_format": "verbose_json"}
        )
    assert response.status_code == 200
    assert response.json() == verbose_payload(CANNED)


async def test_transcription_text_returns_plain_body(monkeypatch: pytest.MonkeyPatch) -> None:
    server = build_server(monkeypatch, CANNED)
    async with asgi_client(server) as client:
        response = await client.post("/v1/audio/transcriptions", files=audio_files(), data={"response_format": "text"})
    assert response.status_code == 200
    assert response.text == "the quick brown fox"
    assert response.headers["content-type"].startswith("text/plain")


@pytest.mark.parametrize("bad_format", ["srt", "vtt", "diarized_json"])
async def test_unsupported_response_format_is_400_and_never_runs_the_model(
    monkeypatch: pytest.MonkeyPatch, bad_format: str
) -> None:
    server = build_server(monkeypatch, CANNED)
    async with asgi_client(server) as client:
        response = await client.post(
            "/v1/audio/transcriptions", files=audio_files(), data={"response_format": bad_format}
        )
    assert response.status_code == 400
    assert "not supported" in response.json()["error"]["message"]
    assert server.transcriber.calls == []  # rejected before decode/compute


async def test_model_form_field_is_accepted_and_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    server = build_server(monkeypatch, CANNED)
    async with asgi_client(server) as client:
        response = await client.post("/v1/audio/transcriptions", files=audio_files(), data={"model": "whisper-1"})
    assert response.status_code == 200
    assert response.json() == {"text": "the quick brown fox"}
    assert len(server.transcriber.calls) == 1  # ran, never inspected the model field


async def test_missing_file_part_is_400_not_500(monkeypatch: pytest.MonkeyPatch) -> None:
    server = build_server(monkeypatch, CANNED)
    async with asgi_client(server) as client:
        response = await client.post("/v1/audio/transcriptions", data={"model": "whisper-1"})
    assert response.status_code == 400
    assert "file" in response.json()["error"]["message"]
    assert server.transcriber.calls == []


async def test_a_non_upload_file_field_is_400(monkeypatch: pytest.MonkeyPatch) -> None:
    server = build_server(monkeypatch, CANNED)
    async with asgi_client(server) as client:
        response = await client.post("/v1/audio/transcriptions", data={"file": "not-an-upload"})
    assert response.status_code == 400
    assert "file" in response.json()["error"]["message"]


async def test_undecodable_audio_surfaces_as_400_not_500(monkeypatch: pytest.MonkeyPatch) -> None:
    from athome.stt import server as server_mod

    server = build_server(monkeypatch, CANNED)

    async def boom_decode(data: bytes) -> array.array:
        raise SttError("ffmpeg decode failed (rc=1): moov atom not found")

    monkeypatch.setattr(server_mod, "decode", boom_decode)
    async with asgi_client(server) as client:
        response = await client.post("/v1/audio/transcriptions", files=audio_files())
    assert response.status_code == 400
    assert "ffmpeg decode failed" in response.json()["error"]["message"]


# --- probe routes never wake the model -------------------------------------


async def test_models_lists_the_configured_variant_without_waking(monkeypatch: pytest.MonkeyPatch) -> None:
    server = build_server(monkeypatch, CANNED, variant="moonshine-tiny")
    async with asgi_client(server) as client:
        response = await client.get("/v1/models")
    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": [{"id": "moonshine-tiny", "object": "model"}]}
    assert server.transcriber.calls == []


async def test_health_reports_ok_without_waking(monkeypatch: pytest.MonkeyPatch) -> None:
    server = build_server(monkeypatch, CANNED, variant="moonshine-tiny")
    async with asgi_client(server) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "moonshine-tiny"}
    assert server.transcriber.calls == []


# --- lifespan reaps the model ----------------------------------------------


async def test_lifespan_discards_a_loaded_model_on_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    server = build_server(monkeypatch, CANNED)
    async with server._lifespan(server.app):
        server.transcriber.resource.loaded = True  # a request loaded the model
    assert server.transcriber.resource.discarded is True


async def test_lifespan_leaves_an_unloaded_model_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    server = build_server(monkeypatch, CANNED)
    async with server._lifespan(server.app):
        pass
    assert server.transcriber.resource.discarded is False


# --- serve.py recipe registration ------------------------------------------


def test_stt_is_an_enrolled_recipe() -> None:
    assert "stt" in serve.RECIPES


def test_settings_for_stt_loads_the_serve_stt_section() -> None:
    settings = serve.settings_for("stt")
    assert isinstance(settings, SttServeSettings)
    assert (settings.variant, settings.port, settings.idle_s) == ("parakeet-tdt-0.6b-v2", 8403, 300.0)


def test_command_for_stt_runs_athome_serve_stt() -> None:
    assert serve.command_for("stt") == (sys.executable, "-m", "athome", "serve", "stt")


def test_command_for_stt_refuses_a_model_or_port_override() -> None:
    with pytest.raises(serve.ServeError, match="no model or port override"):
        serve.command_for("stt", model="parakeet-unified-en-0.6b")
    with pytest.raises(serve.ServeError, match="no model or port override"):
        serve.command_for("stt", port=9999)


def test_configured_model_stt_is_the_variant() -> None:
    assert serve.configured_model("stt") == "parakeet-tdt-0.6b-v2"


def test_managed_server_stt_targets_the_configured_port() -> None:
    server = serve.ManagedServer("stt")
    assert server.served_port == 8403
    assert server.handle().base_url == "http://127.0.0.1:8403/v1"


def test_the_serve_group_registers_the_stt_command() -> None:
    result = CliRunner().invoke(serve.cli, ["stt", "--help"])
    assert result.exit_code == 0, result.output
    assert "--fd" in result.output


def test_serve_stt_binds_the_fd_when_given(monkeypatch: pytest.MonkeyPatch) -> None:
    from athome.stt import server as server_mod

    captured: dict[str, object] = {}

    def fake_run(app: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    server_mod.serve_stt(fd=7)
    assert captured == {"fd": 7, "log_level": "info", "access_log": False}


def test_serve_stt_binds_host_and_port_without_an_fd(monkeypatch: pytest.MonkeyPatch) -> None:
    from athome.stt import server as server_mod

    captured: dict[str, object] = {}

    def fake_run(app: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    server_mod.serve_stt()
    assert captured == {"host": "127.0.0.1", "port": 8403, "log_level": "info", "access_log": False}


# --- CLI: models + download ------------------------------------------------


def test_cli_models_lists_enrolled_variants() -> None:
    result = CliRunner().invoke(cli, ["models"])
    assert result.exit_code == 0, result.output
    assert "parakeet-tdt-0.6b-v2" in result.output
    assert "moonshine-tiny" in result.output


def test_cli_models_json_is_the_variant_list() -> None:
    result = CliRunner().invoke(cli, ["models", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == list(VARIANTS)


def test_cli_download_is_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "moonshine-tiny-Q8_0.gguf").write_bytes(b"gguf")
    calls: list[tuple[str, object]] = []

    async def fake_snapshot(repo: str, *, patterns: tuple[str, ...] | None = None) -> Path:
        calls.append((repo, patterns))
        return tmp_path

    monkeypatch.setattr(catalog.hf, "snapshot", fake_snapshot)
    runner = CliRunner()
    first = runner.invoke(cli, ["download", "moonshine-tiny", "--quant", "Q8_0"])
    second = runner.invoke(cli, ["download", "moonshine-tiny", "--quant", "Q8_0"])
    expected = str(tmp_path / "moonshine-tiny-Q8_0.gguf")
    assert (first.exit_code, second.exit_code) == (0, 0), (first.output, second.output)
    assert first.output.strip() == second.output.strip() == expected
    assert calls == [("handy-computer/moonshine-tiny-gguf", ("*Q8_0*.gguf",))] * 2


def test_cli_transcribe_prints_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from athome.stt import cli as cli_mod

    source = tmp_path / "clip.wav"
    source.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

    async def fake_decode(data: bytes) -> array.array:
        return array.array("f", [0.0, 0.0])

    class OneShot:
        def __init__(self, *a: object, **k: object) -> None: ...

        async def transcribe(self, pcm: object) -> Transcript:
            return CANNED

    monkeypatch.setattr(cli_mod, "decode", fake_decode)
    monkeypatch.setattr(cli_mod, "Transcriber", OneShot)
    result = CliRunner().invoke(cli, ["transcribe", str(source)])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "the quick brown fox"


def test_transcript_payload_shape() -> None:
    payload = transcript_payload(CANNED)
    assert payload["text"] == "the quick brown fox"
    assert payload["load_ms"] == 42.0
    assert payload["segments"][0] == {"start": 0.0, "end": 1.0, "text": "the quick"}
    assert payload["words"][0] == {"start": 0.0, "end": 0.3, "text": "the"}
