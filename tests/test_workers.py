from __future__ import annotations

import copyreg
import os
import pickle
import signal
import sys
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import pytest

from athome import workers
from athome.wire import LENGTH_PREFIX, WIRE_VERSION, WireError, decode, encode, read_frame, validate
from athome.workers import (
    MAX_FRAME_BYTES,
    HandshakeMismatch,
    PipeWorker,
    WorkerCrashed,
    WorkerError,
    WorkerPool,
    WorkerSpec,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

HANDLER_SOURCE = """
import os, sys
from athome.workers import serve

class Handler:
    def echo(self, payload):
        return payload
    def boom(self, payload):
        raise ValueError("handler blew up: " + repr(payload))
    def fingerprint(self, payload=None):
        return {"pid": os.getpid()}

serve(Handler())
"""

BAD_WIRE_SOURCE = """
import sys, pickle
body = pickle.dumps({"wire": 999, "fingerprint": {}}, protocol=5)
sys.stdout.buffer.write(len(body).to_bytes(4, "big") + body)
sys.stdout.buffer.flush()
"""

NON_WIRE_SOURCE = """
import sys, pickle
def write(obj):
    body = pickle.dumps(obj, protocol=5)
    sys.stdout.buffer.write(len(body).to_bytes(4, "big") + body); sys.stdout.buffer.flush()
write({"wire": 1, "fingerprint": {}})
sys.stdin.buffer.read(int.from_bytes(sys.stdin.buffer.read(4), "big"))
write({1, 2, 3})
"""

STUBBORN_NON_WIRE_SOURCE = """
import sys, pickle, time
def write(obj):
    body = pickle.dumps(obj, protocol=5)
    sys.stdout.buffer.write(len(body).to_bytes(4, "big") + body); sys.stdout.buffer.flush()
write({"wire": 1, "fingerprint": {}})
sys.stdin.buffer.read(int.from_bytes(sys.stdin.buffer.read(4), "big"))
write({1, 2, 3})
time.sleep(120)
"""

CRASH_SOURCE = """
import sys, pickle
body = pickle.dumps({"wire": 1, "fingerprint": {}}, protocol=5)
sys.stdout.buffer.write(len(body).to_bytes(4, "big") + body); sys.stdout.buffer.flush()
sys.stdin.buffer.read(4)
sys.stderr.write("fatal: engine exploded\\n"); sys.stderr.flush()
sys.exit(7)
"""

HANDSHAKE_CRASH_SOURCE = """
import sys
sys.stderr.write("fatal: died before handshake\\n"); sys.stderr.flush()
sys.exit(9)
"""

DELAY_SOURCE = """
import time
from athome.workers import serve

class Handler:
    def slow_echo(self, payload):
        time.sleep(payload["delay"])
        return payload["value"]

serve(Handler())
"""

GRANDCHILD_SOURCE = """
import os, time
from athome.workers import serve

class Handler:
    def pid(self, payload):
        return os.getpid()
    def hang(self, payload):
        time.sleep(300)

serve(Handler())
"""

# Spawns the real worker as an inheriting child and waits, the way `uv run` forks without exec.
SUPERVISOR_SOURCE = (
    f"import subprocess, sys; sys.exit(subprocess.Popen([sys.executable, '-c', {GRANDCHILD_SOURCE!r}]).wait())"
)


def wire_side_effect(path: str) -> str:
    Path(path).write_text("pwned")
    return path


class ReduceGadget:
    def __init__(self, path: str) -> None:
        self.path = path

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (wire_side_effect, (self.path,))


class Weird(str):
    pass


class WeirdInt(int):
    pass


class WeirdFloat(float):
    pass


class OversizedPrefixStream:
    def __init__(self, size: int) -> None:
        self.prefix = size.to_bytes(LENGTH_PREFIX, "big")
        self.body_reads = 0

    async def receive_exactly(self, nbytes: int) -> bytes:
        if nbytes == LENGTH_PREFIX:
            return self.prefix
        self.body_reads += 1
        raise AssertionError("body must not be read for an oversized frame")


class SyncOversizedPrefixStream:
    def __init__(self, size: int) -> None:
        self.prefix = size.to_bytes(LENGTH_PREFIX, "big")
        self.offset = 0
        self.body_reads = 0

    def read(self, nbytes: int) -> bytes:
        if self.offset < LENGTH_PREFIX:
            chunk = self.prefix[self.offset : self.offset + nbytes]
            self.offset += len(chunk)
            return chunk
        self.body_reads += 1
        raise AssertionError("body must not be read for an oversized frame")


@asynccontextmanager
async def running(source: str) -> AsyncIterator[PipeWorker]:
    worker = PipeWorker(WorkerSpec((sys.executable, "-c", source)))
    try:
        yield worker
    finally:
        await worker.aclose()


def test_wire_round_trip_is_identity() -> None:
    for value in (None, True, 7, 3.5, "s", b"bytes", [1, [2]], (1, "a"), {"k": [b"v", None]}):
        assert decode(encode(value)) == value


def test_encode_normalizes_primitive_subclasses() -> None:
    payload = decode(encode({Weird("k"): Weird("HELLO"), "n": WeirdInt(3), "f": WeirdFloat(0.5), "l": [Weird("x")]}))
    assert payload == {"k": "HELLO", "n": 3, "f": 0.5, "l": ["x"]}
    assert [type(key) for key in payload] == [str, str, str, str]
    assert type(payload["k"]) is str
    assert type(payload["n"]) is int
    assert type(payload["f"]) is float
    assert type(payload["l"][0]) is str


def test_encode_keeps_bool_out_of_the_int_coercion() -> None:
    payload = decode(encode({"on": True, "off": False, "n": 1}))
    assert [type(value) for value in payload.values()] == [bool, bool, int]


def test_validate_rejects_non_wire() -> None:
    with pytest.raises(WireError):
        validate({1, 2, 3})
    with pytest.raises(WireError):
        validate(object())
    with pytest.raises(WireError):
        validate({1: "non-str-key"})
    with pytest.raises(WireError):
        encode({1, 2, 3})


def test_decode_rejects_non_wire_frame() -> None:
    body = pickle.dumps({1, 2, 3}, protocol=5)
    with pytest.raises(WireError):
        decode(len(body).to_bytes(4, "big") + body)


async def test_round_trip() -> None:
    async with running(HANDLER_SOURCE) as worker:
        assert await worker.call("echo", "hello") == "hello"
        assert await worker.call("echo", None) is None
        assert await worker.call("echo", 42) == 42
        assert await worker.call("echo", (1, "a")) == (1, "a")
        assert await worker.call("echo", [1, 2, {"k": b"bytes"}]) == [1, 2, {"k": b"bytes"}]


async def test_error_propagates_as_worker_error() -> None:
    async with running(HANDLER_SOURCE) as worker:
        with pytest.raises(WorkerError) as excinfo:
            await worker.call("boom", "payload-x")
        assert "handler blew up: 'payload-x'" in str(excinfo.value)
        assert "ValueError" in str(excinfo.value)


async def test_handshake_mismatch_raises_before_first_call_returns() -> None:
    async with running(BAD_WIRE_SOURCE) as worker:
        with pytest.raises(HandshakeMismatch) as excinfo:
            await worker.call("echo", 1)
        assert excinfo.value.got == 999
        assert excinfo.value.expected == WIRE_VERSION


async def test_wire_error_on_non_wire_reply() -> None:
    async with running(NON_WIRE_SOURCE) as worker:
        with pytest.raises(WireError):
            await worker.call("echo", "x")


async def test_cleanup_killpg_failure_never_masks_the_wire_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def deny_killpg(pgid: int, sig: int) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "killpg", deny_killpg)
    async with running(NON_WIRE_SOURCE) as worker:
        with pytest.raises(WireError):  # the poisoning path's own killpg failure stays suppressed
            await worker.call("echo", "x")


async def test_unkillable_worker_bounds_the_reap_wait_and_the_wire_error_still_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_killpg = os.killpg
    denied: list[int] = []

    def deny_killpg(pgid: int, sig: int) -> None:
        denied.append(pgid)
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "killpg", deny_killpg)
    monkeypatch.setattr(workers, "REAP_WAIT_TIMEOUT", 0.2)
    monkeypatch.setattr(workers, "STDERR_JOIN_TIMEOUT", 0.1)
    worker = PipeWorker(WorkerSpec((sys.executable, "-c", STUBBORN_NON_WIRE_SOURCE)))
    try:
        with anyio.fail_after(5.0):  # the bounded wait, not a hang, must deliver the error
            with pytest.raises(WireError):
                await worker.call("echo", "x")
    finally:
        for pgid in denied:  # the deliberately-leaked worker must not outlive the test
            with suppress(OSError):
                real_killpg(pgid, signal.SIGKILL)


async def test_worker_crash_raises_worker_crashed_with_returncode_and_stderr() -> None:
    async with running(CRASH_SOURCE) as worker:
        with pytest.raises(WorkerCrashed) as excinfo:
            await worker.call("echo", "x")
        assert excinfo.value.returncode == 7
        assert "engine exploded" in excinfo.value.stderr_tail


async def test_lock_serializes_concurrent_calls() -> None:
    payloads = {index: bytes([index]) * 50_000 for index in range(16)}
    results: dict[int, object] = {}
    async with running(HANDLER_SOURCE) as worker:

        async def one(index: int) -> None:
            results[index] = await worker.call("echo", payloads[index])

        async with anyio.create_task_group() as group:
            for index in payloads:
                group.start_soon(one, index)
    assert results == payloads


async def test_pool_lease_affinity_after_prefetch() -> None:
    pool = WorkerPool(WorkerSpec((sys.executable, "-c", HANDLER_SOURCE)), size=2)
    try:
        async with pool.lease() as held:
            await pool.prefetch("A", "echo", 1)
            warmed = pool.affinity["A"]
            assert warmed is not held
        async with pool.lease("A") as leased:
            assert leased is warmed
            assert leased is not held
            assert (await leased.call("fingerprint", None))["pid"] == warmed.fingerprint["pid"]
    finally:
        await pool.aclose()


def test_decode_refuses_reduce_gadget(tmp_path: Path) -> None:
    marker = tmp_path / "pwned.txt"
    body = pickle.dumps(ReduceGadget(str(marker)), protocol=5)
    with pytest.raises(WireError):
        decode(len(body).to_bytes(LENGTH_PREFIX, "big") + body)
    assert not marker.exists()


async def test_next_frame_rejects_oversized_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = PipeWorker(WorkerSpec((sys.executable, "-c", "")))
    stream = OversizedPrefixStream(MAX_FRAME_BYTES + 1)
    monkeypatch.setattr(worker, "stdout", stream)
    with pytest.raises(WireError):
        await worker.next_frame()
    assert stream.body_reads == 0


async def test_handshake_mismatch_tears_worker_down() -> None:
    async with running(BAD_WIRE_SOURCE) as worker:
        with pytest.raises(HandshakeMismatch):
            await worker.call("echo", 1)
        assert worker.process is None
        assert worker.stdout is None
        assert worker.stderr_thread is None


async def test_pool_acquire_returns_permit_on_cancel() -> None:
    pool = WorkerPool(WorkerSpec((sys.executable, "-c", HANDLER_SOURCE)), size=2)
    try:
        await pool.guard.acquire()
        started = anyio.Event()

        async def stalled() -> None:
            started.set()
            await pool.acquire(None)

        async with anyio.create_task_group() as group:
            group.start_soon(stalled)
            await started.wait()
            await anyio.sleep(0.05)
            group.cancel_scope.cancel()
        pool.guard.release()
        with anyio.fail_after(2):
            first = await pool.acquire(None)
            second = await pool.acquire(None)
        assert {first, second} == set(pool.workers)
    finally:
        await pool.aclose()


def test_decode_refuses_ext_gadget(tmp_path: Path) -> None:
    marker = tmp_path / "ext-pwned.txt"
    code = 0x00E11EAF
    module, name = wire_side_effect.__module__, wire_side_effect.__qualname__
    copyreg.add_extension(module, name, code)
    copyreg._extension_cache[code] = wire_side_effect
    try:
        body = pickle.dumps(ReduceGadget(str(marker)), protocol=5)
        with pytest.raises(WireError):
            decode(len(body).to_bytes(LENGTH_PREFIX, "big") + body)
    finally:
        copyreg.remove_extension(module, name, code)
        copyreg._extension_cache.pop(code, None)
    assert not marker.exists()


def test_read_frame_rejects_oversized_prefix() -> None:
    stream = SyncOversizedPrefixStream(MAX_FRAME_BYTES + 1)
    with pytest.raises(WireError):
        read_frame(stream)
    assert stream.body_reads == 0


async def test_handshake_crash_tears_worker_down() -> None:
    async with running(HANDSHAKE_CRASH_SOURCE) as worker:
        with pytest.raises(WorkerCrashed) as excinfo:
            await worker.call("echo", 1)
        assert excinfo.value.returncode == 9
        assert worker.process is None
        assert worker.stdout is None
        assert worker.stderr_thread is None


async def test_call_cancel_poisons_worker_against_stale_reply() -> None:
    async with running(DELAY_SOURCE) as worker:
        assert await worker.call("slow_echo", {"delay": 0.0, "value": "WARM"}) == "WARM"
        with anyio.move_on_after(0.15):
            await worker.call("slow_echo", {"delay": 0.5, "value": "STALE"})
        assert worker.process is None
        assert worker.stdout is None
        assert worker.stderr_thread is None
        assert await worker.call("slow_echo", {"delay": 0.0, "value": "FRESH"}) == "FRESH"


async def test_poison_kills_supervisor_wrapped_grandchild() -> None:
    async def reaped(pid: int) -> bool:
        for _ in range(100):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            await anyio.sleep(0.02)
        return False

    worker = PipeWorker(WorkerSpec((sys.executable, "-c", SUPERVISOR_SOURCE)))
    grandchild = await worker.call("pid", None)
    try:
        started = time.monotonic()
        with anyio.move_on_after(0.1):
            await worker.call("hang", None)
        elapsed = time.monotonic() - started
        assert worker.process is None
        assert elapsed < 1.0
        assert await reaped(grandchild)
    finally:
        with suppress(ProcessLookupError):
            os.kill(grandchild, signal.SIGKILL)
        await worker.aclose()


async def test_lease_release_survives_cancel_under_contention() -> None:
    pool = WorkerPool(WorkerSpec((sys.executable, "-c", HANDLER_SOURCE)), size=2)
    entered = anyio.Event()
    gate = anyio.Event()
    scopes: list[anyio.CancelScope] = []

    async def leaser() -> None:
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            async with pool.lease():
                entered.set()
                await gate.wait()

    try:
        async with anyio.create_task_group() as group:
            group.start_soon(leaser)
            await entered.wait()
            await pool.guard.acquire()
            gate.set()
            await anyio.sleep(0.05)
            scopes[0].cancel()
            await anyio.sleep(0.05)
            pool.guard.release()
        with anyio.fail_after(2):
            first = await pool.acquire(None)
            second = await pool.acquire(None)
        assert {first, second} == set(pool.workers)
    finally:
        await pool.aclose()
