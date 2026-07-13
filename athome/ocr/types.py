from __future__ import annotations

import re
from dataclasses import dataclass
from functools import reduce
from typing import Protocol, runtime_checkable

FENCE = re.compile(r"^\s*(?:```|~~~)")
HEADING = re.compile(r"^#{1,6}\s+")
SEPARATOR = re.compile(r"^:?-+:?$")
EMPHASIS = (
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),
    (re.compile(r"__(.+?)__"), r"\1"),
    (re.compile(r"\*(.+?)\*"), r"\1"),
    (re.compile(r"_(.+?)_"), r"\1"),
    (re.compile(r"`(.+?)`"), r"\1"),
)


def strip_emphasis(text: str) -> str:
    return reduce(lambda acc, rule: rule[0].sub(rule[1], acc), EMPHASIS, text)


def strip_table_row(line: str) -> str:
    cells = [cell for cell in (part.strip() for part in line.split("|")) if cell]
    if not cells or all(SEPARATOR.match(cell) for cell in cells):
        return ""
    return strip_emphasis(" ".join(cells))


def strip_line(line: str) -> str:
    if "|" in line:
        return strip_table_row(line)
    return strip_emphasis(HEADING.sub("", line))


def strip_markdown(markdown: str) -> str:
    lines: list[str] = []
    in_fence = False
    for raw in markdown.splitlines():
        if FENCE.match(raw):
            in_fence = not in_fence
        elif in_fence:
            if raw.strip():
                lines.append(raw)
        elif line := strip_line(raw.strip()):
            lines.append(line)
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Box:
    """Pixel rectangle in top-left-origin image coordinates."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class OcrToken:
    """A single recognized token with its bounding box and recognizer confidence."""

    text: str
    box: Box
    confidence: float


@dataclass(frozen=True, slots=True)
class Document:
    """A page read by a document-OCR engine."""

    markdown: str
    tokens: tuple[OcrToken, ...] = ()

    @property
    def text(self) -> str:
        """The markdown stripped of formatting: headings, emphasis, and table pipes
        removed, cell text preserved, code fences unwrapped, into plain lines."""
        return strip_markdown(self.markdown)


@runtime_checkable
class TokenOcr(Protocol):
    """Token-level OCR: geometry + confidence per token (the ensemble substrate)."""

    async def tokens(
        self, image: bytes, *, region: Box | None = None, upscale: float = 1.0
    ) -> tuple[OcrToken, ...]: ...


@runtime_checkable
class DocOcr(Protocol):
    """Page-level document OCR (VLM engines): image in, structured markdown out."""

    async def read(self, image: bytes) -> Document: ...
