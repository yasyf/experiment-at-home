"""An OpenAI-compatible transcription server over the transcribe.cpp engine.

A tiny Starlette app fronts one lazily-loaded :class:`~athome.stt.engine.Transcriber`:
``POST /v1/audio/transcriptions`` decodes an uploaded clip and returns its transcript in the
requested ``response_format`` (``json`` / ``verbose_json`` / ``text``; ``srt``/``vtt`` are
refused), while ``GET /v1/models`` and ``GET /health`` answer from configuration alone so a
probe never wakes the model. The lifespan runs the engine's idle reaper, so the model unloads
after ``idle_s`` idle and reloads on the next request. Compute is serialized by the engine's own
lock — there is no second in-flight guard here — and the whole app is what the activator spawns
behind ``{LISTEN_FD}`` for wake-on-use lazy serving.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, ClassVar

from athome.config import SectionSettings, load
from athome.stt.catalog import DEFAULT_QUANT
from athome.stt.engine import Transcriber
from athome.stt.pcm import decode
from athome.stt.types import SttError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response

    from athome.stt.types import Transcript

# Audio parts routinely exceed starlette's 1 MiB per-part default; the engine handles unbounded
# length, so lift the multipart guard well clear of any HTTP-posted clip.
MAX_UPLOAD_BYTES = 1024**3
SUPPORTED_FORMATS: tuple[str, ...] = ("json", "verbose_json", "text")
DEFAULT_FORMAT = "json"


class SttServeSettings(SectionSettings):
    """The ``[serve.stt]`` section: the served variant, quant, bind address, and idle window.

    Env overrides derive from the section as ``ATHOME_SERVE_STT_<FIELD>`` (init kwargs > env >
    ``~/.athome/config.toml`` > defaults).

    Example:
        >>> load(SttServeSettings).port  # doctest: +SKIP
        8403
    """

    section: ClassVar[tuple[str, ...]] = ("serve", "stt")
    variant: str = "parakeet-tdt-0.6b-v2"
    quant: str = DEFAULT_QUANT
    host: str = "127.0.0.1"
    port: int = 8403
    idle_s: float = 300.0


def duration_of(transcript: Transcript) -> float:
    ends = [segment.end for segment in transcript.segments] or [word.end for word in transcript.words]
    return max(ends, default=0.0)


def verbose_payload(transcript: Transcript) -> dict[str, object]:
    return {
        "task": "transcribe",
        "duration": duration_of(transcript),
        "text": transcript.text,
        "segments": [
            {"id": index, "start": segment.start, "end": segment.end, "text": segment.text}
            for index, segment in enumerate(transcript.segments)
        ],
        "words": [{"word": word.text, "start": word.start, "end": word.end} for word in transcript.words],
    }


def error(message: str, *, status_code: int) -> Response:
    from starlette.responses import JSONResponse

    return JSONResponse({"error": {"message": message, "type": "invalid_request_error"}}, status_code=status_code)


def rendered(transcript: Transcript, response_format: str) -> Response:
    from starlette.responses import JSONResponse, PlainTextResponse

    match response_format:
        case "json":
            return JSONResponse({"text": transcript.text})
        case "verbose_json":
            return JSONResponse(verbose_payload(transcript))
        case "text":
            return PlainTextResponse(transcript.text)
    raise SttError(f"unrenderable response_format {response_format!r}")


class SttServer:
    """The transcription app: one :class:`~athome.stt.engine.Transcriber` behind an allowlist of routes.

    ``GET /v1/models`` and ``GET /health`` answer from :class:`SttServeSettings` without touching the
    model, so a health probe never loads weights; only ``POST /v1/audio/transcriptions`` decodes audio
    and runs the model, and the lifespan reaps it once idle.

    Example:
        >>> server = SttServer(load(SttServeSettings))  # doctest: +SKIP
        >>> uvicorn.run(server.app, host="127.0.0.1", port=8403)  # doctest: +SKIP
    """

    def __init__(self, settings: SttServeSettings) -> None:
        self.settings = settings
        self.transcriber = Transcriber(settings.variant, quant=settings.quant, idle_s=settings.idle_s)
        self._app: Starlette | None = None

    @property
    def app(self) -> Starlette:
        if self._app is None:
            self._app = self._build_app()
        return self._app

    def _build_app(self) -> Starlette:
        from starlette.applications import Starlette
        from starlette.routing import Route

        return Starlette(
            routes=[
                Route("/v1/audio/transcriptions", self.transcribe, methods=["POST"]),
                Route("/v1/models", self.models, methods=["GET"]),
                Route("/health", self.health, methods=["GET"]),
            ],
            lifespan=self._lifespan,
        )

    @asynccontextmanager
    async def _lifespan(self, app: Starlette) -> AsyncIterator[None]:
        import anyio

        async with anyio.create_task_group() as tg:
            tg.start_soon(self.transcriber.resource.run)
            try:
                yield
            finally:
                tg.cancel_scope.cancel()
        if self.transcriber.resource.loaded:
            await self.transcriber.resource.discard()

    def models_body(self) -> dict[str, object]:
        return {"object": "list", "data": [{"id": self.settings.variant, "object": "model"}]}

    async def models(self, request: Request) -> Response:
        """List the one configured model without waking it."""
        from starlette.responses import JSONResponse

        return JSONResponse(self.models_body())

    async def health(self, request: Request) -> Response:
        """Report liveness from configuration alone; never loads the model."""
        from starlette.responses import JSONResponse

        return JSONResponse({"status": "ok", "model": self.settings.variant})

    async def transcribe(self, request: Request) -> Response:
        """Decode the uploaded ``file`` and return its transcript in the requested ``response_format``."""
        from starlette.datastructures import UploadFile

        form = await request.form(max_part_size=MAX_UPLOAD_BYTES)
        response_format = str(form.get("response_format") or DEFAULT_FORMAT)
        if response_format not in SUPPORTED_FORMATS:
            return error(
                f"response_format {response_format!r} is not supported; use one of {', '.join(SUPPORTED_FORMATS)}",
                status_code=400,
            )
        if not isinstance(upload := form.get("file"), UploadFile):
            return error("a multipart 'file' part is required", status_code=400)
        try:
            transcript = await self.transcriber.transcribe(await decode(await upload.read()))
        except SttError as exc:
            return error(str(exc), status_code=400)
        return rendered(transcript, response_format)


def serve_stt(*, fd: int | None = None) -> None:
    """Run the STT server under uvicorn, binding the inherited ``fd`` or the configured host/port.

    Args:
        fd: A pre-bound listener fd to serve on (the activator's ``{LISTEN_FD}`` handoff); ``None``
            binds :class:`SttServeSettings`'s ``host``/``port`` directly.
    """
    import uvicorn

    settings = load(SttServeSettings)
    server = SttServer(settings)
    # access_log=False: /health and /v1/models probes hit every few seconds and add only noise.
    bind = {"fd": fd} if fd is not None else {"host": settings.host, "port": settings.port}
    uvicorn.run(server.app, log_level="info", access_log=False, **bind)
