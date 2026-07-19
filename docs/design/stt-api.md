# athome STT API design (stt · serve.stt · activator wake_paths)

The binding public-surface contract for the `athome.stt` subpackage (v0.10.0).
Same rules as `core-api.md` and `phase-b-api.md` (frozen dataclasses+slots,
keyword-only options, async-only SDK, pydantic-settings per-module sections,
typed errors rooted at `AthomeError`, no defensive coding, Google docstrings on
public API only, imports clean on 3.14t — heavy deps behind the `stt` extra with
lazy imports).

The engine is [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp)
via its pure-ctypes binding — `py3-none-any`, imports on 3.14t, GIL released on
foreign calls, so no sidecar process is needed (unlike `engines/ocr-paddle`).
The binding is pre-1.0: both dists are exact-pinned and bumped in lockstep.

```toml
[project.optional-dependencies]
stt = [
  "transcribe-cpp==0.1.3",         # exact pin — pre-1.0 ABI
  "transcribe-cpp-native==0.1.3",  # bundled CPU+Metal dylib, exact pin
  "starlette", "uvicorn", "python-multipart",  # server
  "huggingface-hub",               # weights
  "static-ffmpeg",                 # decode fallback when no ffmpeg on PATH
]
```

Module → file map:

| Module | Files | CLI |
|---|---|---|
| stt types | `athome/stt/types.py` | — |
| stt pcm | `athome/stt/pcm.py` | — |
| stt catalog | `athome/stt/catalog.py` | — |
| stt engine | `athome/stt/engine.py` | — |
| stt server | `athome/stt/server.py` | `athome serve stt [--fd N]` |
| stt cli | `athome/stt/cli.py` | `athome stt transcribe\|models\|download` |

---

## Types — `athome/stt/types.py`

All offsets are **seconds** (float). The binding speaks milliseconds; the
ms-to-seconds division happens exactly once, at the engine boundary.

```python
class SttError(AthomeError): ...

@dataclass(frozen=True, slots=True)
class Word:
    text: str
    start: float
    end: float

@dataclass(frozen=True, slots=True)
class Segment:
    text: str
    start: float
    end: float
    words: tuple[Word, ...] = ()

@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    segments: tuple[Segment, ...]
    words: tuple[Word, ...]
    load_ms: float          # one-time weight upload, NOT process cold-start (see engine)
    language: str | None = None   # model-reported; None on the stream path
```

## PCM — `athome/stt/pcm.py`

The binding's input contract is 16 kHz mono float32 with no internal resampling.

```python
RATE_HZ = 16_000

def f32_from_s16(data: bytes) -> array.array: ...   # the one s16le→f32 conversion point
def require_pcm(pcm: Pcm) -> Pcm: ...               # rejects empty/wrong-typecode/odd-length buffers
async def decode(data: bytes) -> array.array: ...   # any container → 16k mono f32, ffmpeg subprocess
```

`decode` resolves `ffmpeg` PATH-first and falls back to the `static-ffmpeg`
wheel (the yclaw host installs no brew packages). It stages through a temp file
off the event loop and is server/CLI-only — in-process consumers pass PCM.

## Catalog — `athome/stt/catalog.py`

```python
ORG = "handy-computer"
DEFAULT_QUANT = "Q8_0"
VARIANTS: tuple[str, ...] = (
    "parakeet-tdt-0.6b-v2", "parakeet-unified-en-0.6b", "granite-speech-4.1-2b",
    "moonshine-tiny", "parakeet-tdt-0.6b-v3", "whisper-large-v3-turbo",
    "nemotron-speech-streaming-en-0.6b",
)

def repo_for(variant: str) -> str: ...                       # handy-computer/<variant>-gguf
async def gguf_path(variant: str, quant: str = DEFAULT_QUANT) -> Path: ...
```

Each variant name is both the HF repo stem and the GGUF filename stem; commit
SHAs live in `athome.hf.REVISIONS` (the repos carry no tags), and `gguf_path`
downloads only the matching quant via `hf.snapshot(patterns=)`. Capability is
the binding's alone: it raises (`NotImplementedByModel`, input-length caps)
when a variant can't do something, and the catalog stays a name table.

## Engine — `athome/stt/engine.py`

```python
class Transcriber:
    def __init__(self, variant: str, *, quant: str = DEFAULT_QUANT,
                 backend: Backend = "auto", idle_s: float = 300.0) -> None: ...
    async def transcribe(self, pcm: Pcm) -> Transcript: ...
    async def transcribe_batch(self, pcms: Sequence[Pcm]) -> list[Transcript | SttError]: ...
    async def stream(self, *, lookahead_ms: int) -> SttStream: ...

class SttStream:  # async context manager
    async def feed(self, pcm: Pcm) -> tuple[Segment, ...]: ...   # newly COMMITTED deltas only
    async def finalize(self) -> Transcript: ...
    async def aclose(self) -> None: ...                           # idempotent
```

The load-bearing contracts, in decreasing order of blood spilled learning them:

- **One in-flight compute per loaded model.** The 0.x binding corrupts on
  overlap. One `anyio.Semaphore(1, max_value=1)` serializes every run, batch,
  and stream feed. A semaphore, not a lock: a stream may open in one task and
  close in another, and lock release is task-affine while semaphore release is
  not; `max_value=1` makes an over-release raise instead of silently widening
  the lane. Each open `SttStream` also carries its own operation semaphore, so
  concurrent `feed`/`finalize` calls on one stream serialize instead of racing
  into the native layer.
- **Lazy load, idle unload.** `IdleResource[Model]` loads on first use and
  unloads after `idle_s`. An open stream holds a use-reference plus the compute
  lane for its whole lifetime — the reaper cannot unload mid-stream, and a stuck
  stream starves batch (bounded upstream by the activator's `upstream_timeout_s`).
- **Warmup at load.** `_load` runs a 0.5 s silent transcription before
  publishing the model: it captures `load_ms` for the stream path (streams have
  no `Result`) and moves the one-time Metal shader compile (~8 s on first touch,
  measured on an M4 Max) off the first real request. `Result.timings.load_ms`
  itself is ~175 ms and identical cold vs warm — the weight upload, not the
  cold-start. A warmup failure closes the model and re-raises; the model is
  never published half-warm.
- **Streamed timing is derived, not native.** The binding exposes only
  committed/tentative text plus `audio_committed_ms` watermarks on streams — no
  per-segment timestamps. `feed` emits one `Segment` per committed delta
  spanning `[previous watermark, new watermark]` in stream-absolute seconds,
  with monotonically non-decreasing starts; `finalize` emits the remaining tail
  without duplicating or dropping anything already returned. A watermark jump
  across several utterances yields one merged segment — by design.
- **Per-slot batch isolation.** An invalid buffer or a native failure becomes an
  `SttError` value in its slot; siblings still transcribe. An all-invalid batch
  never touches the native layer.
- **Fail-loud validation.** `require_pcm` raises `SttError` on empty or
  malformed buffers — including a zero-length stream feed, which leaves the
  stream open and usable. Native `TranscribeError`s map to `SttError` at every
  public boundary; nothing else is caught.

## Server — `athome/stt/server.py`

OpenAI-compatible shim, Starlette, one settings section:

```python
class SttServeSettings(SectionSettings):
    section = ("serve", "stt")
    variant: str = "parakeet-tdt-0.6b-v2"
    quant: str = DEFAULT_QUANT
    host: str = "127.0.0.1"
    port: int = 8403
    idle_s: float = 300.0

def serve_stt(*, fd: int | None = None) -> None: ...
```

- `POST /v1/audio/transcriptions` — multipart; the `model` form field is
  accepted and ignored (clients send `whisper-1`); `response_format` `json`
  (default) / `verbose_json` / `text`; `srt`/`vtt` return 400, as do a missing
  file, an empty file, and undecodable audio. `verbose_json` includes `language` only when
  the model reports one.
- `GET /v1/models`, `GET /health` — answer from settings without touching the
  model, so the activator's health probe never wakes a cold child.
- Lifespan runs the idle reaper; `--fd` maps the activator's pre-bound
  `{LISTEN_FD}` onto uvicorn, so the child serves the exact socket the wake
  arrived on. Single-flight is the engine's compute lane — no second guard.

The `serve.py` recipe `"stt"` sits beside rapid-mlx/mlx-vlm/llama-server, so
`ManagedServer("stt")` gets up/down/status/idle/launchd for free. Unlike the
other recipes it is always-configured (every setting has a default) and so
always appears in `probe_all()`.

## CLI — `athome/stt/cli.py`

```
athome stt transcribe FILE [--variant V] [--quant Q] [--json]
athome stt models [--json]
athome stt download [--variant V] [--quant Q] [--json]   # idempotent pre-fetch
```

`download` exists for deploy scripts (yclaw's `setup.sh` calls it so first boot
pulls weights and the first request only loads them).

## Activator — `wake_paths`

`ActivatorSettings` gains `wake_paths: tuple[str, ...]` defaulting to the
previous hardcoded constant, so existing deployments match byte-identically.
STT deployments override via env:

```sh
ATHOME_SERVE_ACTIVATOR_WAKE_PATHS='["/v1/audio/transcriptions"]'
```
