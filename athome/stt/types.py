from __future__ import annotations

from dataclasses import dataclass

from athome.errors import AthomeError


class SttError(AthomeError):
    """Raised when a transcription cannot be produced (bad input, an unknown variant, a decode failure)."""


@dataclass(frozen=True, slots=True)
class Word:
    """One recognized word with its span in **seconds** from the start of the audio."""

    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class Segment:
    """A contiguous run of speech with its span in **seconds** and the words it covers."""

    text: str
    start: float
    end: float
    words: tuple[Word, ...] = ()


@dataclass(frozen=True, slots=True)
class Transcript:
    """A fully materialized transcription.

    Every offset is in **seconds** (the native ``t0_ms``/``t1_ms`` are divided once, at the engine
    boundary). ``load_ms`` is the model's first-party cold/warm load timing for this run, carried
    through so callers can size their own cold-start budgets.

    Example:
        >>> transcript.text
        'hello world'
        >>> transcript.segments[0].start
        0.0
    """

    text: str
    segments: tuple[Segment, ...]
    words: tuple[Word, ...]
    load_ms: float
