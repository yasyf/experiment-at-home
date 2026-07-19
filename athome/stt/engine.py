from __future__ import annotations

import array
import functools
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

import anyio
from anyio import to_thread

from athome.idle import IdleResource
from athome.stt.catalog import DEFAULT_QUANT, gguf_path
from athome.stt.pcm import require_pcm
from athome.stt.types import Segment, SttError, Transcript, Word

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from transcribe_cpp import Backend, Model, Result, Session, Stream, StreamText, StreamUpdate
    from transcribe_cpp import Segment as NativeSegment
    from transcribe_cpp import Word as NativeWord

    from athome.stt.pcm import Pcm

# 0.5 s of silence run at load: reads the model's load_ms and forces the backend to
# compile its kernels here rather than on the first user request (the Metal cold stall).
WARMUP_SAMPLES = 8_000


async def load_model(path: str, backend: Backend) -> Model:
    from transcribe_cpp import Model

    return await to_thread.run_sync(functools.partial(Model, path, backend=backend))


def stream_family_for(model: Model, lookahead_ms: int):
    """Pick the streaming extension the model accepts, mapping ``lookahead_ms`` to its right context.

    Auto-detected per model (no host-hardcoding): the first family the model accepts wins, and a
    model that accepts none streams on its family default (no lookahead knob).
    """
    from transcribe_cpp import MoonshineStreamingOptions, ParakeetBufferedStreamOptions, VoxtralRealtimeStreamOptions

    candidates = (
        ParakeetBufferedStreamOptions(right_ms=lookahead_ms),
        MoonshineStreamingOptions(min_decode_interval_ms=lookahead_ms),
        VoxtralRealtimeStreamOptions(min_decode_interval_ms=lookahead_ms),
    )
    return next((family for family in candidates if model.accepts(family)), None)


def word_from(word: NativeWord) -> Word:
    return Word(text=word.text, start=word.t0_ms / 1000.0, end=word.t1_ms / 1000.0)


def segment_from(segment: NativeSegment, words: tuple[NativeWord, ...]) -> Segment:
    return Segment(
        text=segment.text,
        start=segment.t0_ms / 1000.0,
        end=segment.t1_ms / 1000.0,
        words=tuple(word_from(words[i]) for i in range(segment.first_word, segment.first_word + segment.n_words)),
    )


def validated_slot(pcm: Pcm) -> Pcm | SttError:
    try:
        return require_pcm(pcm)
    except SttError as invalid:
        return invalid


def transcript_from_result(result: Result) -> Transcript:
    """Map a native ``Result`` into a :class:`Transcript`, dividing ms → seconds once, here."""
    return Transcript(
        text=result.text,
        segments=tuple(segment_from(segment, result.words) for segment in result.segments),
        words=tuple(word_from(word) for word in result.words),
        load_ms=result.timings.load_ms,
    )


class Transcriber:
    """A lazily-loaded transcribe.cpp model with a single serialized compute lane.

    One ``anyio.Lock`` serializes every run, batch, and stream feed: the 0.x binding allows at most
    one in-flight compute per loaded model, and overlapping runs corrupt each other. The model loads
    on first use and unloads after ``idle_s`` idle via :class:`~athome.idle.IdleResource`; an open
    stream holds both a use-reference and the compute lock for its whole lifetime, so the reaper
    cannot unload mid-stream and a stuck stream starves batch (bounded upstream by the activator).

    Example:
        >>> stt = Transcriber("parakeet-tdt-0.6b-v2")
        >>> (await stt.transcribe(pcm)).text  # doctest: +SKIP
        'hello world'
    """

    def __init__(
        self, variant: str, *, quant: str = DEFAULT_QUANT, backend: Backend = "auto", idle_s: float = 300.0
    ) -> None:
        self.variant = variant
        self.quant = quant
        self.backend = backend
        self.resource: IdleResource[Model] = IdleResource(self._load, self._unload, ttl_s=idle_s)
        self.lock = anyio.Lock()
        self.model: Model | None = None
        self.load_ms = 0.0

    async def _load(self) -> Model:
        model = await load_model(str(await gguf_path(self.variant, self.quant)), self.backend)
        try:
            load_ms = await to_thread.run_sync(functools.partial(self._warmup, model))
        except Exception:
            await to_thread.run_sync(model.close)
            raise
        self.model = model
        self.load_ms = load_ms
        return model

    def _warmup(self, model: Model) -> float:
        from transcribe_cpp.errors import OutputTruncated

        with model.session() as session:
            try:
                return session.run(array.array("f", bytes(WARMUP_SAMPLES * 4))).timings.load_ms
            except OutputTruncated as truncated:
                # A short-utterance model (moonshine) loops on the silent warmup to its token cap;
                # the run still compiled the kernels, and the binding keeps timings on the partial.
                return truncated.partial_result.timings.load_ms

    async def _unload(self) -> None:
        if (model := self.model) is not None:
            self.model = None
            self.load_ms = 0.0
            await to_thread.run_sync(model.close)

    async def _compute[T](self, fn: Callable[[], T]) -> T:
        async with self.lock:
            return await to_thread.run_sync(fn)

    def _run(self, model: Model, pcm: Pcm) -> Result:
        with model.session() as session:
            return session.run(pcm, timestamps="auto")

    def _run_batch(self, model: Model, pcms: list[Pcm]) -> list:
        with model.session() as session:
            return session.run_batch(pcms, timestamps="auto", return_exceptions=True)

    async def transcribe(self, pcm: Pcm) -> Transcript:
        """Transcribe one float32 16 kHz mono PCM buffer into a :class:`Transcript`."""
        require_pcm(pcm)
        async with self.resource.use() as model:
            return transcript_from_result(await self._compute(functools.partial(self._run, model, pcm)))

    async def transcribe_batch(self, pcms: Sequence[Pcm]) -> list[Transcript | SttError]:
        """Transcribe several buffers in one dispatch; each slot is a Transcript or its own error.

        Order-preserving with per-slot isolation: a slot that fails validation or transcription
        surfaces as an :class:`SttError` in its position while the others return their transcripts.
        Only valid slots reach the native batch; a batch with no valid slot never wakes the model.
        """
        from transcribe_cpp.errors import TranscribeError

        slots = [validated_slot(pcm) for pcm in pcms]
        if not (valid := [slot for slot in slots if not isinstance(slot, SttError)]):
            return [slot for slot in slots if isinstance(slot, SttError)]
        async with self.resource.use() as model:
            results = await self._compute(functools.partial(self._run_batch, model, valid))
        transcripts = iter(
            SttError(str(item)) if isinstance(item, TranscribeError) else transcript_from_result(item)
            for item in results
        )
        return [slot if isinstance(slot, SttError) else next(transcripts) for slot in slots]

    async def stream(self, *, lookahead_ms: int) -> SttStream:
        """Open a streaming transcription that holds the compute lock for its whole lifetime."""
        stack = AsyncExitStack()
        model = await stack.enter_async_context(self.resource.use())
        await stack.enter_async_context(self.lock)
        try:
            session, native = await to_thread.run_sync(functools.partial(self._begin_stream, model, lookahead_ms))
        except BaseException:
            await stack.aclose()
            raise
        return SttStream(session, native, stack, self.load_ms)

    def _begin_stream(self, model: Model, lookahead_ms: int) -> tuple[Session, Stream]:
        session = model.session()
        native = session.stream(commit_policy="auto", family=stream_family_for(model, lookahead_ms))
        return session, native


class SttStream:
    """A live streaming transcription.

    Feed PCM chunks; each :meth:`feed` returns only the newly committed segment deltas (stream-absolute
    seconds, monotone starts). :meth:`finalize` flushes the tail and returns the whole
    :class:`Transcript` without duplicating or dropping any fed segment. Streamed segment timing is
    derived from the binding's ``audio_committed_ms`` watermark — the stream path exposes committed
    text, not native per-segment timestamps — so each committed delta spans from the previous
    watermark to the current one.
    """

    def __init__(self, session: Session, native: Stream, stack: AsyncExitStack, load_ms: float) -> None:
        self.session = session
        self.native = native
        self.stack = stack
        self.load_ms = load_ms
        self.committed = ""
        self.committed_ms = 0
        self.segments: list[Segment] = []
        self.closed = False

    def _feed_native(self, pcm: Pcm) -> tuple[StreamUpdate, StreamText | None]:
        update = self.native.feed(pcm)
        return update, self.native.text() if update.committed_changed else None

    def _commit(self, committed: str, watermark_ms: int) -> tuple[Segment, ...]:
        start, end = self.committed_ms / 1000.0, watermark_ms / 1000.0
        delta = committed[len(self.committed) :].strip()
        self.committed, self.committed_ms = committed, watermark_ms
        if not delta:
            return ()
        self.segments.append(segment := Segment(text=delta, start=start, end=end, words=()))
        return (segment,)

    async def feed(self, pcm: Pcm) -> tuple[Segment, ...]:
        """Feed a float32 16 kHz mono chunk; return only the newly committed segment deltas."""
        require_pcm(pcm)
        update, text = await to_thread.run_sync(functools.partial(self._feed_native, pcm))
        if text is None:
            return ()
        return self._commit(text.committed, update.audio_committed_ms)

    async def finalize(self) -> Transcript:
        """Flush the final hypothesis, close the stream, and return the full transcript."""
        update, text = await to_thread.run_sync(self._finalize_native)
        self._commit(text.committed, update.audio_committed_ms)
        await self.aclose()
        return Transcript(text=text.committed, segments=tuple(self.segments), words=(), load_ms=self.load_ms)

    def _finalize_native(self) -> tuple[StreamUpdate, StreamText]:
        update = self.native.finalize()
        return update, self.native.text()

    async def aclose(self) -> None:
        """Reset the native stream and release the compute lock and use-reference. Idempotent."""
        if self.closed:
            return
        self.closed = True
        await to_thread.run_sync(self._reset_native)
        await self.stack.aclose()

    def _reset_native(self) -> None:
        self.native.reset()
        self.session.close()
