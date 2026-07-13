from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import ClassVar, Literal

import anyio
import click

from athome.cache import Cache
from athome.cli import coro, emit, json_option
from athome.config import SectionSettings, load
from athome.errors import AthomeError
from athome.ocr.apple import AppleVision
from athome.ocr.ensemble import EnsembleTokenOcr
from athome.ocr.merge import LlmMerger
from athome.ocr.paddle import PaddleOcr
from athome.ocr.types import Box, Document, OcrToken
from athome.ocr.vlm import VlmOcr
from athome.workers import WorkerPool, WorkerSpec

type Profile = Literal["realtime", "quality"]

CACHE_NAMESPACE = "ocr"
CACHE_VERSION = 1
STACK_VERSION = "1"
PADDLE_SPEC = WorkerSpec(("uvx", "athome-ocr-paddle"))
PADDLE_POOL_SIZE = 4
LINE_TOLERANCE = 8.0
AGREEMENT_FLOOR = 0.6
PROFILE_CHOICE = click.Choice(("realtime", "quality"))


class OcrError(AthomeError):
    """Raised when an OCR read cannot be produced."""


class OcrSettings(SectionSettings):
    """The ``[ocr]`` section: the default read profile."""

    section: ClassVar[tuple[str, ...]] = ("ocr",)
    profile: Profile = "quality"


def image_digest(image: bytes) -> str:
    return hashlib.blake2b(image, digest_size=16).hexdigest()


def group_lines(tokens: tuple[OcrToken, ...]) -> list[list[OcrToken]]:
    lines: list[list[OcrToken]] = []
    for token in tokens:
        baseline = token.box.y + token.box.height / 2
        if lines and abs(baseline - (lines[-1][0].box.y + lines[-1][0].box.height / 2)) <= LINE_TOLERANCE:
            lines[-1].append(token)
        else:
            lines.append([token])
    return lines


def layout_markdown(tokens: tuple[OcrToken, ...]) -> str:
    ordered = sorted(tokens, key=lambda token: (token.box.y, token.box.x))
    lines = group_lines(tuple(ordered))
    return "\n".join(" ".join(token.text for token in sorted(line, key=lambda token: token.box.x)) for line in lines)


def words(text: str) -> set[str]:
    return {word for raw in text.split() if (word := raw.strip().casefold())}


def documents_agree(reference: Document, candidate: Document) -> bool:
    reference_words, candidate_words = words(reference.text), words(candidate.text)
    if not reference_words or not candidate_words:
        return True
    return len(reference_words & candidate_words) / len(reference_words | candidate_words) >= AGREEMENT_FLOOR


def document_to_bytes(document: Document) -> bytes:
    return json.dumps(asdict(document)).encode()


def document_from_bytes(data: bytes) -> Document:
    payload = json.loads(data)
    return Document(
        markdown=payload["markdown"],
        tokens=tuple(
            OcrToken(text=token["text"], box=Box(**token["box"]), confidence=token["confidence"])
            for token in payload["tokens"]
        ),
    )


async def read_realtime(image: bytes) -> Document:
    pool = WorkerPool(PADDLE_SPEC, size=PADDLE_POOL_SIZE)
    try:
        tokens = await EnsembleTokenOcr(PaddleOcr(pool), AppleVision()).tokens(image)
    finally:
        await pool.aclose()
    return Document(markdown=layout_markdown(tokens), tokens=tokens)


async def read_quality(image: bytes) -> Document:
    vlm_document = await VlmOcr().read(image)
    apple_tokens = await AppleVision().tokens(image)
    apple_document = Document(markdown=layout_markdown(apple_tokens), tokens=apple_tokens)
    if documents_agree(vlm_document, apple_document):
        return vlm_document
    return await LlmMerger().merge(image, (vlm_document, apple_document))


async def run_profile(image: bytes, profile: Profile) -> Document:
    match profile:
        case "realtime":
            return await read_realtime(image)
        case "quality":
            return await read_quality(image)


async def read(image: bytes, *, profile: Profile | None = None) -> Document:
    """Read an image into a Markdown ``Document``, dispatched by profile and blob-cached.

    ``realtime`` unions the PaddleOCR primary with an Apple Vision supplement into positioned
    tokens; ``quality`` reads the page with the VLM engine, cross-checks it against Apple Vision,
    and reconciles the two with :class:`LlmMerger` only when they disagree. Results are cached in
    the shared blob cache, keyed on the image digest, profile, and engine-stack version, so a
    repeat read of the same image and profile is a cache hit.

    Args:
        image: The page bytes (PNG/JPEG).
        profile: ``realtime`` or ``quality``; defaults to the ``[ocr]`` config profile.

    Returns:
        The recognized page as a ``Document``.
    """
    resolved = profile or load(OcrSettings).profile
    cache = Cache.open(CACHE_NAMESPACE, version=CACHE_VERSION)
    key = cache.key(image_digest(image), resolved, STACK_VERSION)
    if (cached := await cache.get_bytes(key)) is not None:
        return document_from_bytes(cached)
    document = await run_profile(image, resolved)
    await cache.put_bytes(key, document_to_bytes(document))
    return document


def document_payload(document: Document) -> dict[str, object]:
    return {"markdown": document.markdown, "tokens": [asdict(token) for token in document.tokens]}


@click.command(name="ocr")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--profile", type=PROFILE_CHOICE, default=None, help="Read profile (default: the [ocr] config value).")
@json_option
@coro
async def cli(source: Path, profile: Profile | None, as_json: bool) -> None:
    """Read an image with the OCR engines and print its Markdown (or JSON)."""
    document = await read(await anyio.Path(source).read_bytes(), profile=profile)
    emit(document_payload(document) if as_json else document.markdown, as_json=as_json)
