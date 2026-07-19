from __future__ import annotations

from athome.stt.catalog import VARIANTS, gguf_path
from athome.stt.engine import SttStream, Transcriber
from athome.stt.pcm import RATE_HZ, decode, f32_from_s16
from athome.stt.server import SttServeSettings, SttServer, serve_stt
from athome.stt.types import Segment, SttError, Transcript, Word
