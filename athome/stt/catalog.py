from __future__ import annotations

from typing import TYPE_CHECKING

from athome import hf
from athome.stt.types import SttError

if TYPE_CHECKING:
    from pathlib import Path

ORG = "handy-computer"
DEFAULT_QUANT = "Q8_0"

# Each name is both the HF repo stem (``<name>-gguf``) and the GGUF filename stem
# (``<name>-<quant>.gguf``); SHAs live in athome.hf.REVISIONS, capability in the binding.
VARIANTS: tuple[str, ...] = (
    "parakeet-tdt-0.6b-v2",
    "parakeet-unified-en-0.6b",
    "granite-speech-4.1-2b",
    "moonshine-tiny",
    "parakeet-tdt-0.6b-v3",
    "whisper-large-v3-turbo",
    "nemotron-speech-streaming-en-0.6b",
)


def repo_for(variant: str) -> str:
    """The ``handy-computer/<variant>-gguf`` HF repo id for an enrolled variant."""
    if variant not in VARIANTS:
        raise SttError(f"unknown STT variant {variant!r}; enrolled: {', '.join(VARIANTS)}")
    return f"{ORG}/{variant}-gguf"


async def gguf_path(variant: str, quant: str = DEFAULT_QUANT) -> Path:
    """Materialize one quant of ``variant`` from the HF cache and return its ``.gguf`` path.

    Downloads only the matching quant (``allow_patterns``) at the pinned revision, then resolves
    the exact ``<variant>-<quant>.gguf`` file inside the snapshot.

    Args:
        variant: An enrolled :data:`VARIANTS` name.
        quant: The GGUF quantization tag (e.g. ``"Q8_0"``, ``"F16"``).

    Returns:
        The local path of the ``.gguf`` weights file.

    Raises:
        SttError: ``variant`` is not enrolled, or no ``<variant>-<quant>.gguf`` exists in the repo.
    """
    snapshot = await hf.snapshot(repo_for(variant), patterns=(f"*{quant}*.gguf",))
    if not (path := snapshot / f"{variant}-{quant}.gguf").exists():
        raise SttError(f"no {quant} weights for {variant!r} in {repo_for(variant)} (looked for {path.name})")
    return path
