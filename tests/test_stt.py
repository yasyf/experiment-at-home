from __future__ import annotations

import array
import tempfile
import threading
import time
import wave
from pathlib import Path

import anyio
import pytest

pytest.importorskip("transcribe_cpp")

# The binding's value dataclasses are pure data — use them REAL and fake only the compute boundary.
from transcribe_cpp import ParakeetBufferedStreamOptions, Result, StreamText, StreamUpdate, Timings
from transcribe_cpp import Segment as NativeSegment
from transcribe_cpp import Word as NativeWord
from transcribe_cpp.errors import InvalidArgument, NotImplementedByModel, OutputTruncated

from athome.stt import catalog
from athome.stt.catalog import DEFAULT_QUANT, VARIANTS, gguf_path, repo_for
from athome.stt.engine import Transcriber, transcript_from_result
from athome.stt.pcm import RATE_HZ, decode, f32_from_s16, require_pcm
from athome.stt.types import Segment, SttError, Transcript

FIXTURE = Path(__file__).parent / "fixtures" / "stt" / "hello.wav"


def load_fixture() -> array.array:
    with wave.open(str(FIXTURE)) as w:
        return f32_from_s16(w.readframes(w.getnframes()))


def pcm(n: int = 200) -> array.array:
    return array.array("f", [0.0] * n)


class ComputeTracker:
    """Counts concurrent in-flight compute across worker threads and records the peak."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def __enter__(self) -> ComputeTracker:
        with self.lock:
            self.current += 1
            self.peak = max(self.peak, self.current)
        return self

    def __exit__(self, *_exc: object) -> None:
        with self.lock:
            self.current -= 1


class FakeStream:
    def __init__(self, model: FakeModel) -> None:
        self.model = model
        self.script = list(model.script)
        self.final = model.final
        self.committed = ""
        self.index = -1
        self.reset_called = False

    def feed(self, _pcm: object) -> StreamUpdate:
        with self.model.tracker:
            time.sleep(self.model.delay)
            if (error := self.model.feed_error) is not None:
                raise error
            self.index += 1
            committed, watermark = self.script[self.index]
            changed = committed != self.committed
            self.committed = committed
            return StreamUpdate(
                result_changed=changed,
                is_final=False,
                revision=self.index,
                input_received_ms=watermark,
                audio_committed_ms=watermark,
                buffered_ms=0,
                committed_changed=changed,
                tentative_changed=False,
            )

    def finalize(self) -> StreamUpdate:
        with self.model.tracker:
            if (error := self.model.finalize_error) is not None:
                raise error
            self.committed, watermark = self.final
            return StreamUpdate(
                result_changed=True,
                is_final=True,
                revision=self.index + 1,
                input_received_ms=watermark,
                audio_committed_ms=watermark,
                buffered_ms=0,
                committed_changed=True,
                tentative_changed=False,
            )

    def text(self) -> StreamText:
        return StreamText(full=self.committed, committed=self.committed, tentative="")

    def reset(self) -> None:
        self.reset_called = True


class FakeSession:
    def __init__(self, model: FakeModel) -> None:
        self.model = model

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _timings(self) -> Timings:
        return Timings(load_ms=self.model.load_ms, mel_ms=0.0, encode_ms=0.0, decode_ms=0.0)

    def _result(self) -> Result:
        return Result(
            text=self.model.text,
            language="en",
            timestamp_kind="word",
            segments=self.model.segments,
            words=self.model.words,
            tokens=(),
            timings=self._timings(),
        )

    def run(self, _pcm: object, *, timestamps: str = "auto") -> Result:
        with self.model.tracker:
            time.sleep(self.model.delay)
            return self._result()

    def run_batch(self, pcms: list, *, timestamps: str = "auto", return_exceptions: bool = False) -> list:
        with self.model.tracker:
            self.model.batch_sizes.append(len(pcms))
            time.sleep(self.model.delay)
            return list(self.model.batch)

    def stream(self, **_kwargs: object) -> FakeStream:
        return FakeStream(self.model)

    def close(self) -> None:
        self.model.session_closes += 1


class FakeModel:
    def __init__(
        self,
        *,
        text: str = "",
        segments: tuple = (),
        words: tuple = (),
        batch: list | None = None,
        script: list | None = None,
        final: tuple = ("", 0),
        tracker: ComputeTracker | None = None,
        delay: float = 0.0,
        load_ms: float = 42.0,
        closed: list | None = None,
        feed_error: Exception | None = None,
        finalize_error: Exception | None = None,
    ) -> None:
        self.text = text
        self.segments = segments
        self.words = words
        self.batch = batch or []
        self.script = script or []
        self.final = final
        self.tracker = tracker or ComputeTracker()
        self.delay = delay
        self.load_ms = load_ms
        self.closed = closed if closed is not None else []
        self.feed_error = feed_error
        self.finalize_error = finalize_error
        self.batch_sizes: list[int] = []
        self.session_closes = 0

    def session(self) -> FakeSession:
        return FakeSession(self)

    def accepts(self, family: object) -> bool:
        return isinstance(family, ParakeetBufferedStreamOptions)

    def close(self) -> None:
        self.closed.append(True)


def wire(monkeypatch: pytest.MonkeyPatch, *models: FakeModel) -> None:
    remaining = iter(models)

    async def fake_gguf_path(variant: str, quant: str = DEFAULT_QUANT) -> Path:
        return Path("/fake/model.gguf")

    async def fake_load_model(path: str, backend: str) -> FakeModel:
        return next(remaining)

    monkeypatch.setattr("athome.stt.engine.gguf_path", fake_gguf_path)
    monkeypatch.setattr("athome.stt.engine.load_model", fake_load_model)


# --- pcm primitives --------------------------------------------------------


def test_f32_from_s16_scales_int16_to_unit_range() -> None:
    data = array.array("h", [0, 16384, -32768, 32767]).tobytes()
    out = f32_from_s16(data)
    assert out[0] == 0.0
    assert out[1] == 0.5
    assert out[2] == -1.0
    assert out[3] == pytest.approx(0.99997, abs=1e-4)


def test_require_pcm_accepts_float_array_and_rejects_bad_buffers() -> None:
    good = array.array("f", [0.1, 0.2])
    assert require_pcm(good) is good
    with pytest.raises(SttError, match="float32"):
        require_pcm(array.array("h", [1, 2]))
    with pytest.raises(SttError, match="whole number of float32"):
        require_pcm(b"\x00\x00\x00")
    with pytest.raises(SttError, match="empty"):
        require_pcm(array.array("f", []))


async def test_decode_stages_its_temp_file_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    loop_thread = threading.current_thread()
    staging_threads: list[threading.Thread] = []
    real_mkstemp = tempfile.mkstemp

    def recording_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        staging_threads.append(threading.current_thread())
        return real_mkstemp(*args, **kwargs)

    class FakeCompleted:
        returncode = 0
        stdout = array.array("f", [0.25]).tobytes()
        stderr = b""

    async def fake_run_process(cmd: list[str], *, check: bool = False) -> FakeCompleted:
        return FakeCompleted()

    monkeypatch.setattr("athome.stt.pcm.tempfile.mkstemp", recording_mkstemp)
    monkeypatch.setattr("athome.stt.pcm.ffmpeg_path", lambda: "ffmpeg")
    monkeypatch.setattr("athome.stt.pcm.anyio.run_process", fake_run_process)

    samples = await decode(b"fake-container-bytes")
    assert samples.tolist() == [0.25]
    assert staging_threads and all(thread is not loop_thread for thread in staging_threads)


# --- catalog ---------------------------------------------------------------


def test_repo_for_derives_the_handy_computer_repo() -> None:
    assert repo_for("parakeet-tdt-0.6b-v2") == "handy-computer/parakeet-tdt-0.6b-v2-gguf"


def test_repo_for_unknown_variant_raises() -> None:
    with pytest.raises(SttError, match="unknown STT variant"):
        repo_for("not-a-model")


async def test_gguf_path_filters_by_quant_and_resolves_the_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "parakeet-tdt-0.6b-v2-Q8_0.gguf").write_bytes(b"gguf")
    calls: dict[str, object] = {}

    async def fake_snapshot(repo: str, *, patterns: tuple[str, ...] | None = None) -> Path:
        calls["repo"], calls["patterns"] = repo, patterns
        return tmp_path

    monkeypatch.setattr(catalog.hf, "snapshot", fake_snapshot)
    path = await gguf_path("parakeet-tdt-0.6b-v2", "Q8_0")
    assert path == tmp_path / "parakeet-tdt-0.6b-v2-Q8_0.gguf"
    assert calls == {"repo": "handy-computer/parakeet-tdt-0.6b-v2-gguf", "patterns": ("*Q8_0*.gguf",)}


async def test_gguf_path_missing_file_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def fake_snapshot(repo: str, *, patterns: tuple[str, ...] | None = None) -> Path:
        return tmp_path

    monkeypatch.setattr(catalog.hf, "snapshot", fake_snapshot)
    with pytest.raises(SttError, match="no Q8_0 weights"):
        await gguf_path("moonshine-tiny", "Q8_0")


def test_catalog_variants_are_the_enrolled_set() -> None:
    assert "moonshine-tiny" in VARIANTS
    assert "parakeet-unified-en-0.6b" in VARIANTS


# --- ms -> seconds exactness -----------------------------------------------


def test_transcript_from_result_divides_ms_to_seconds_once() -> None:
    seg = NativeSegment(text="hi", t0_ms=1234, t1_ms=5678, first_word=0, n_words=1, first_token=0, n_tokens=1)
    word = NativeWord(text="hi", t0_ms=1234, t1_ms=5678, seg_index=0, first_token=0, n_tokens=1)
    result = Result(
        text="hi",
        language="en",
        timestamp_kind="word",
        segments=(seg,),
        words=(word,),
        tokens=(),
        timings=Timings(load_ms=9.0, mel_ms=0, encode_ms=0, decode_ms=0),
    )
    transcript = transcript_from_result(result)
    assert transcript.segments[0].start == 1.234
    assert transcript.segments[0].end == 5.678
    assert transcript.segments[0].words[0].start == 1.234
    assert transcript.words[0].end == 5.678
    assert transcript.load_ms == 9.0
    assert transcript.language == "en"  # the native language tag rides along


async def test_transcribe_maps_seconds_and_carries_load_ms(monkeypatch: pytest.MonkeyPatch) -> None:
    seg = NativeSegment(text="hi", t0_ms=500, t1_ms=1500, first_word=0, n_words=0, first_token=0, n_tokens=0)
    wire(monkeypatch, FakeModel(text="hi", segments=(seg,), load_ms=123.0))
    transcript = await Transcriber("x").transcribe(pcm())
    assert transcript.text == "hi"
    assert transcript.segments[0].start == 0.5
    assert transcript.load_ms == 123.0
    assert transcript.language == "en"  # round-trips from the native Result


# --- warmup tolerates a truncated silent run -------------------------------


def test_warmup_recovers_load_ms_from_a_truncated_silent_run() -> None:
    # A short-utterance model loops on the silent warmup and raises OutputTruncated (on Metal); the
    # partial result still carries load_ms, so the load must not fail.
    partial = Result(
        text="♪",
        language="en",
        timestamp_kind="segment",
        segments=(),
        words=(),
        tokens=(),
        timings=Timings(load_ms=17.0, mel_ms=0, encode_ms=0, decode_ms=0),
    )

    class TruncatingSession:
        def __enter__(self) -> TruncatingSession:
            return self

        def __exit__(self, *_exc: object) -> None:
            pass

        def run(self, _pcm: object, *, timestamps: str = "auto") -> Result:
            truncated = OutputTruncated("output truncated: decode hit the cap")
            truncated.partial_result = partial
            raise truncated

    class TruncatingModel:
        def session(self) -> TruncatingSession:
            return TruncatingSession()

    assert Transcriber("x")._warmup(TruncatingModel()) == 17.0


class ExplodingWarmupSession:
    def __enter__(self) -> ExplodingWarmupSession:
        return self

    def __exit__(self, *_exc: object) -> None:
        pass

    def run(self, _pcm: object, *, timestamps: str = "auto") -> Result:
        raise RuntimeError("metal kernel compile failed")


class ExplodingWarmupModel(FakeModel):
    def session(self) -> ExplodingWarmupSession:
        return ExplodingWarmupSession()


async def test_warmup_failure_closes_the_model_and_leaves_the_next_load_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad, good = ExplodingWarmupModel(), FakeModel(text="recovered")
    wire(monkeypatch, bad, good)
    stt = Transcriber("x")

    with pytest.raises(RuntimeError, match="kernel compile"):
        await stt.transcribe(pcm())
    assert bad.closed == [True]  # the natively loaded model was not orphaned
    assert stt.model is None

    assert (await stt.transcribe(pcm())).text == "recovered"


# --- compute serialization -------------------------------------------------


async def test_at_most_one_compute_in_flight_under_interleaving(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = ComputeTracker()
    wire(monkeypatch, FakeModel(tracker=tracker, delay=0.02))
    stt = Transcriber("x")
    async with anyio.create_task_group() as tg:
        for _ in range(6):
            tg.start_soon(stt.transcribe, pcm())
    assert tracker.peak == 1


async def test_open_stream_blocks_concurrent_compute(monkeypatch: pytest.MonkeyPatch) -> None:
    wire(monkeypatch, FakeModel(script=[("", 0)], final=("done", 1000)))
    stt = Transcriber("x")
    stream = await stt.stream(lookahead_ms=1000)
    done = anyio.Event()

    async def transcribe_once() -> None:
        await stt.transcribe(pcm())
        done.set()

    async with anyio.create_task_group() as tg:
        tg.start_soon(transcribe_once)
        await anyio.sleep(0.05)
        assert not done.is_set()  # the open stream holds the compute lock
        await stream.finalize()  # releases the lock
        with anyio.fail_after(2):
            await done.wait()
    assert done.is_set()


async def test_stream_closed_from_another_task_releases_the_compute_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # anyio.Lock release is task-affine; a stream opened here and closed from a sibling task must
    # not poison the transcriber (semaphore release is legal cross-task).
    wire(monkeypatch, FakeModel(text="after", script=[("", 0)], final=("done", 1000)))
    stt = Transcriber("x")
    stream = await stt.stream(lookahead_ms=1000)

    async with anyio.create_task_group() as tg:
        tg.start_soon(stream.aclose)
    assert stream.closed is True
    with anyio.fail_after(2):
        assert (await stt.transcribe(pcm())).text == "after"


async def test_concurrent_feeds_never_overlap_in_the_native_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = ComputeTracker()
    wire(monkeypatch, FakeModel(script=[("a", 100), ("ab", 200)], final=("ab", 300), tracker=tracker, delay=0.02))
    stt = Transcriber("x")
    stream = await stt.stream(lookahead_ms=1000)

    async with anyio.create_task_group() as tg:
        tg.start_soon(stream.feed, pcm())
        tg.start_soon(stream.feed, pcm())
    assert tracker.peak == 1  # the per-stream semaphore serializes concurrent feeds
    await stream.finalize()


# --- idle unload vs. an open stream ----------------------------------------


async def test_open_stream_blocks_idle_sweep_until_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []
    wire(monkeypatch, FakeModel(script=[("", 0)], final=("done", 1000), closed=closed))
    stt = Transcriber("x", idle_s=0.0)
    stream = await stt.stream(lookahead_ms=1000)

    await stt.resource.sweep(now=10**9)
    assert stt.resource.loaded is True  # inflight use() from the open stream blocks the reaper
    assert closed == []

    await stream.finalize()
    await stt.resource.sweep(now=10**9)
    assert stt.resource.loaded is False
    assert closed == [True]  # the model was closed on unload


# --- batch per-slot isolation ----------------------------------------------


async def test_batch_isolates_per_slot_errors_and_preserves_order(monkeypatch: pytest.MonkeyPatch) -> None:
    ok = Result(
        text="ok",
        language="en",
        timestamp_kind="segment",
        segments=(),
        words=(),
        tokens=(),
        timings=Timings(load_ms=1.0, mel_ms=0, encode_ms=0, decode_ms=0),
    )
    failed = InvalidArgument("utterance 1 in batch: bad audio")
    failed.utterance_index = 1
    wire(monkeypatch, FakeModel(batch=[ok, failed, ok]))
    out = await Transcriber("x").transcribe_batch([pcm(), pcm(), pcm()])
    assert [type(item).__name__ for item in out] == ["Transcript", "SttError", "Transcript"]
    assert out[0].text == "ok" and out[2].text == "ok"
    assert isinstance(out[1], SttError) and "bad audio" in str(out[1])


def canned_result(text: str) -> Result:
    return Result(
        text=text,
        language="en",
        timestamp_kind="segment",
        segments=(),
        words=(),
        tokens=(),
        timings=Timings(load_ms=1.0, mel_ms=0, encode_ms=0, decode_ms=0),
    )


async def test_batch_validates_per_slot_and_dispatches_only_the_valid_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel(batch=[canned_result("first"), canned_result("second")])
    wire(monkeypatch, model)
    out = await Transcriber("x").transcribe_batch([pcm(), array.array("f", []), pcm()])
    assert [type(item).__name__ for item in out] == ["Transcript", "SttError", "Transcript"]
    assert (out[0].text, out[2].text) == ("first", "second")
    assert isinstance(out[1], SttError) and "empty" in str(out[1])
    assert model.batch_sizes == [2]  # only the valid slots reached the native batch


async def test_all_invalid_batch_never_touches_the_native_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel(batch=[canned_result("never")])
    wire(monkeypatch, model)
    stt = Transcriber("x")
    out = await stt.transcribe_batch([array.array("f", []), array.array("h", [1, 2])])
    assert [isinstance(item, SttError) for item in out] == [True, True]
    assert "empty" in str(out[0]) and "float32" in str(out[1])
    assert stt.resource.loaded is False  # the model was never loaded
    assert model.batch_sizes == []  # and run_batch never ran


# --- public-boundary error mapping -----------------------------------------


class TruncatingRunSession:
    def __init__(self, model: FakeModel) -> None:
        self.model = model

    def __enter__(self) -> TruncatingRunSession:
        return self

    def __exit__(self, *_exc: object) -> None:
        pass

    def run(self, _pcm: object, *, timestamps: str = "auto") -> Result:
        truncated = OutputTruncated("output truncated: decode hit the token cap")
        truncated.partial_result = canned_result("partial")
        raise truncated


class TruncatingRunModel(FakeModel):
    def session(self) -> TruncatingRunSession:
        return TruncatingRunSession(self)


async def test_transcribe_maps_a_native_error_to_stt_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # The silent warmup tolerates the truncation (partial_result carries load_ms); the user-facing
    # run must surface it as SttError, matching transcribe_batch's mapping, not leak the binding type.
    wire(monkeypatch, TruncatingRunModel())
    with pytest.raises(SttError, match="token cap"):
        await Transcriber("x").transcribe(pcm())


# --- streaming committed-delta semantics -----------------------------------


async def test_feed_returns_only_committed_deltas_with_monotone_absolute_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = [("", 0), ("hello", 1000), ("hello", 1000), ("hello world", 2000)]
    wire(monkeypatch, FakeModel(script=script, final=("hello world foo", 3000)))
    stt = Transcriber("x")
    stream = await stt.stream(lookahead_ms=1000)

    emitted: list[Segment] = []
    for _ in script:
        emitted.extend(await stream.feed(pcm()))

    assert [s.text for s in emitted] == ["hello", "world"]  # no-change feeds emit nothing
    assert [s.start for s in emitted] == [0.0, 1.0]  # stream-absolute, monotone
    assert [s.end for s in emitted] == [1.0, 2.0]

    transcript = await stream.finalize()
    assert transcript.text == "hello world foo"
    # finalize adds only the tail, never duplicating or dropping a fed segment
    assert [s.text for s in transcript.segments] == ["hello", "world", "foo"]
    assert [s.start for s in transcript.segments] == sorted(s.start for s in transcript.segments)
    assert " ".join(s.text for s in transcript.segments) == transcript.text


# --- stream lifecycle -------------------------------------------------------


class StreamRefusingSession(FakeSession):
    def stream(self, **_kwargs: object) -> FakeStream:
        raise NotImplementedByModel("model has no streaming family")


class StreamRefusingModel(FakeModel):
    def session(self) -> FakeSession:
        return StreamRefusingSession(self)


async def test_stream_is_an_async_context_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    wire(monkeypatch, FakeModel(text="after", script=[("hi", 500)], final=("hi", 500)))
    stt = Transcriber("x")
    async with await stt.stream(lookahead_ms=1000) as stream:
        assert [s.text for s in await stream.feed(pcm())] == ["hi"]
    assert stream.closed is True
    with anyio.fail_after(2):
        assert (await stt.transcribe(pcm())).text == "after"


async def test_stream_setup_failure_closes_the_orphaned_session(monkeypatch: pytest.MonkeyPatch) -> None:
    model = StreamRefusingModel()
    wire(monkeypatch, model)
    stt = Transcriber("x")
    with pytest.raises(NotImplementedByModel):
        await stt.stream(lookahead_ms=1000)
    assert model.session_closes == 2  # the warmup session, then the orphaned stream session
    with anyio.fail_after(2):
        await stt.transcribe(pcm())  # the compute lane was released by the setup cleanup


async def test_native_feed_failure_closes_the_stream_and_frees_the_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = InvalidArgument("native feed rejected the buffer")
    wire(monkeypatch, FakeModel(text="after", script=[("", 0)], final=("", 0), feed_error=error))
    stt = Transcriber("x")
    stream = await stt.stream(lookahead_ms=1000)
    with anyio.fail_after(2), pytest.raises(SttError, match="rejected the buffer"):
        await stream.feed(pcm())
    assert stream.closed is True
    with anyio.fail_after(2):
        assert (await stt.transcribe(pcm())).text == "after"


async def test_native_finalize_failure_closes_the_stream_and_frees_the_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = InvalidArgument("native finalize failed")
    wire(monkeypatch, FakeModel(text="after", script=[("", 0)], final=("", 0), finalize_error=error))
    stt = Transcriber("x")
    stream = await stt.stream(lookahead_ms=1000)
    with anyio.fail_after(2), pytest.raises(SttError, match="finalize failed"):
        await stream.finalize()
    assert stream.closed is True
    with anyio.fail_after(2):
        assert (await stt.transcribe(pcm())).text == "after"


async def test_invalid_chunk_leaves_the_stream_open_and_usable(monkeypatch: pytest.MonkeyPatch) -> None:
    # Validation failure is fail-loud input rejection, not stream death: no native call happened.
    wire(monkeypatch, FakeModel(script=[("hello", 1000)], final=("hello", 1000)))
    stt = Transcriber("x")
    stream = await stt.stream(lookahead_ms=1000)
    with pytest.raises(SttError, match="float32"):
        await stream.feed(array.array("h", [1, 2]))
    assert stream.closed is False
    assert [s.text for s in await stream.feed(pcm())] == ["hello"]
    await stream.finalize()


# --- real models (dev-only; --run-live) ------------------------------------


@pytest.mark.live
async def test_live_moonshine_tiny_batch_transcribes_the_fixture() -> None:
    stt = Transcriber("moonshine-tiny", quant="Q8_0", backend="cpu", idle_s=5)
    out = await stt.transcribe_batch([load_fixture()])
    assert len(out) == 1
    assert isinstance(out[0], Transcript)
    assert "fox" in out[0].text.lower()
    assert out[0].load_ms > 0.0


@pytest.mark.live
async def test_live_moonshine_tiny_metal_batch_survives_the_silent_warmup() -> None:
    # Regression: on the default auto (Metal) backend moonshine truncates the silent warmup; the
    # first request must transcribe rather than 500 on the load.
    stt = Transcriber("moonshine-tiny", quant="Q8_0", idle_s=5)
    transcript = await stt.transcribe(load_fixture())
    assert "fox" in transcript.text.lower()
    assert transcript.load_ms > 0.0


@pytest.mark.live
async def test_live_parakeet_unified_streaming_is_monotone_and_absolute() -> None:
    stt = Transcriber("parakeet-unified-en-0.6b", quant="Q8_0", idle_s=5)
    audio = load_fixture()
    stream = await stt.stream(lookahead_ms=1040)
    starts: list[float] = []
    chunk = RATE_HZ // 2
    for i in range(0, len(audio), chunk):
        for seg in await stream.feed(audio[i : i + chunk]):
            starts.append(seg.start)
    transcript = await stream.finalize()
    assert starts == sorted(starts)  # monotone stream-absolute starts
    assert all(start >= 0.0 for start in starts)
    assert "fox" in transcript.text.lower()
    joined = " ".join(seg.text for seg in transcript.segments)
    assert " ".join(joined.split()) == " ".join(transcript.text.split())  # no drop / no duplication


@pytest.mark.live
async def test_live_parakeet_unified_metal_batch_transcribes() -> None:
    stt = Transcriber("parakeet-unified-en-0.6b", quant="Q8_0", idle_s=5)
    transcript = await stt.transcribe(load_fixture())
    assert "fox" in transcript.text.lower()
    assert transcript.words[0].start >= 0.0
    assert transcript.load_ms > 0.0
