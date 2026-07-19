from __future__ import annotations

import array
import os
import shutil
import sys
import tempfile

import anyio
from anyio import to_thread

from athome.stt.types import SttError

RATE_HZ = 16_000

type Pcm = array.array | bytes | bytearray | memoryview


def f32_from_s16(data: bytes) -> array.array:
    """Convert little-endian signed-16-bit PCM to float32 samples in ``[-1.0, 1.0)``.

    This is the single s16 → f32 conversion point: the decode pipeline emits s16le mono 16 kHz,
    the native binding wants float32, and every caller routes through here rather than dividing
    inline.

    Args:
        data: Raw little-endian ``int16`` PCM bytes.

    Returns:
        A float32 :class:`array.array` of the same sample count.
    """
    samples = array.array("h")
    samples.frombytes(data)
    if sys.byteorder == "big":
        samples.byteswap()
    return array.array("f", (sample / 32768.0 for sample in samples))


def require_pcm(pcm: Pcm) -> Pcm:
    """Validate a float32 PCM buffer at the engine boundary, raising :class:`SttError` if unusable.

    The single validation seam before audio reaches the native binding, so a bad buffer surfaces
    as an athome error with a clear message instead of a deep ctypes ``InvalidArgument``.
    """
    match pcm:
        case array.array() if pcm.typecode != "f":
            raise SttError(f"PCM array must be float32 ('f'); got typecode {pcm.typecode!r}")
        case bytes() | bytearray() if len(pcm) % 4:
            raise SttError("raw PCM byte buffer length is not a whole number of float32 samples")
    if len(pcm) == 0:
        raise SttError("empty PCM buffer")
    return pcm


def scratch_path() -> str:
    fd, path = tempfile.mkstemp(prefix="athome-stt-")
    os.close(fd)
    return path


def ffmpeg_path() -> str:
    """Resolve an ffmpeg binary, preferring one on ``PATH`` and falling back to the static wheel.

    The ``static-ffmpeg`` fallback fetches a prebuilt binary on first use (cached thereafter), so
    a host with no system ffmpeg — the no-brew yclaw box — still decodes.
    """
    from static_ffmpeg.run import get_or_fetch_platform_executables_else_raise

    return shutil.which("ffmpeg") or get_or_fetch_platform_executables_else_raise()[0]


async def decode(data: bytes) -> array.array:
    """Decode any audio container to 16 kHz mono float32 PCM via an ffmpeg subprocess.

    ffmpeg reads a seekable temp copy of ``data`` (so trailing-``moov`` m4a/mp4 — an iMessage voice
    note — decodes) and writes ``f32le`` to stdout. This is server/CLI-only; the in-process engine
    wrapper takes already-decoded PCM.

    Args:
        data: The encoded audio bytes (wav, m4a, mp3, …).

    Returns:
        A float32 :class:`array.array` of 16 kHz mono samples.

    Raises:
        SttError: ffmpeg exited non-zero or produced no audio.
    """
    ffmpeg = await to_thread.run_sync(ffmpeg_path)
    path = await to_thread.run_sync(scratch_path)
    try:
        await anyio.Path(path).write_bytes(data)
        result = await anyio.run_process(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                path,
                "-ar",
                str(RATE_HZ),
                "-ac",
                "1",
                "-f",
                "f32le",
                "pipe:1",
            ],
            check=False,
        )
    finally:
        await anyio.Path(path).unlink(missing_ok=True)
    if result.returncode != 0:
        raise SttError(
            f"ffmpeg decode failed (rc={result.returncode}): {result.stderr.decode('utf-8', 'replace').strip()}"
        )
    samples = array.array("f")
    samples.frombytes(result.stdout)
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        raise SttError("ffmpeg produced no audio samples")
    return samples
