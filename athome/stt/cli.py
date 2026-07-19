from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import click

from athome.cli import coro, emit, json_option
from athome.config import load
from athome.stt.catalog import VARIANTS, gguf_path
from athome.stt.engine import Transcriber
from athome.stt.pcm import decode
from athome.stt.server import SttServeSettings

if TYPE_CHECKING:
    from athome.stt.types import Transcript


def transcript_payload(transcript: Transcript) -> dict[str, object]:
    return {
        "text": transcript.text,
        "segments": [
            {"start": segment.start, "end": segment.end, "text": segment.text} for segment in transcript.segments
        ],
        "words": [{"start": word.start, "end": word.end, "text": word.text} for word in transcript.words],
        "load_ms": transcript.load_ms,
    }


@click.group("stt")
def cli() -> None:
    """Transcribe audio and manage transcribe.cpp GGUF weights."""


@cli.command("transcribe")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--variant", default=None, help="Enrolled variant to use (default: the [serve.stt] config value).")
@click.option("--quant", default=None, help="GGUF quantization tag (default: the [serve.stt] config value).")
@json_option
@coro
async def transcribe_command(source: Path, variant: str | None, quant: str | None, as_json: bool) -> None:
    """Transcribe SOURCE (any audio container) and print its text (or JSON)."""
    settings = load(SttServeSettings)
    stt = Transcriber(variant or settings.variant, quant=quant or settings.quant)
    transcript = await stt.transcribe(await decode(await anyio.Path(source).read_bytes()))
    emit(transcript_payload(transcript) if as_json else transcript.text, as_json=as_json)


@cli.command("models")
@json_option
def models_command(as_json: bool) -> None:
    """List the enrolled transcribe.cpp variants."""
    emit(list(VARIANTS), as_json=as_json)


@cli.command("download")
@click.argument("variant", required=False, default=None)
@click.option("--quant", default=None, help="GGUF quantization tag (default: the [serve.stt] config value).")
@json_option
@coro
async def download_command(variant: str | None, quant: str | None, as_json: bool) -> None:
    """Pre-fetch VARIANT's GGUF weights (default: the [serve.stt] config variant); idempotent."""
    settings = load(SttServeSettings)
    emit(str(await gguf_path(variant or settings.variant, quant or settings.quant)), as_json=as_json)
