from __future__ import annotations

import os
import sys
import threading
import traceback
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import anyio
from anyio import Lock
from anyio.streams.buffered import BufferedByteReceiveStream

from athome.config import base_environ
from athome.errors import AthomeError
from athome.wire import LENGTH_PREFIX, WIRE_VERSION, decode, encode, read_frame, validate, write_frame

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from anyio.abc import Process

    from athome.wire import Wire

STDERR_CHUNK = 4096
STDERR_TAIL_CHUNKS = 16
ACLOSE_TIMEOUT = 5.0
STDERR_JOIN_TIMEOUT = 2.0


class WorkerError(AthomeError):
    """A sidecar reported an error through the wire, or its process failed."""


class WorkerCrashed(WorkerError):
    """A sidecar process exited before replying; carries its return code and stderr tail."""

    def __init__(self, returncode: int | None, stderr_tail: str) -> None:
        self.returncode = returncode
        self.stderr_tail = stderr_tail
        super().__init__(f"worker exited with code {returncode}\n{stderr_tail}".rstrip())


class HandshakeMismatch(WorkerError):
    """A sidecar announced a wire version the parent does not speak."""

    def __init__(self, expected: int, got: Wire) -> None:
        self.expected = expected
        self.got = got
        super().__init__(f"worker announced wire {got!r}, parent speaks {expected}")


class WorkerTransport(Protocol):
    """The async surface a sidecar exposes: a wire method call, plus shutdown."""

    async def call(self, method: str, payload: Wire) -> Wire: ...
    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    """How to spawn a sidecar: the command, an environment overlay, and a working directory.

    Example:
        >>> WorkerSpec(("uvx", "--python", "3.13", "athome-ocr-paddle"))
    """

    command: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()
    cwd: Path | None = None


@dataclass(slots=True, eq=False)
class PipeWorker:
    """A lazily-spawned sidecar subprocess speaking length-prefixed wire frames over stdio.

    The child is spawned on the first :meth:`call`; its first frame is a handshake carrying the
    wire version and a fingerprint. A wire-version mismatch raises :class:`HandshakeMismatch`
    before that first call returns. Round-trips are serialized by one lock — a single stdin/stdout
    pipe pair carries one request/reply at a time, and interleaved writes deadlock the pipe. The
    child's stderr is teed to the parent's stderr and its tail retained for crash diagnostics.

    Example:
        >>> worker = PipeWorker(WorkerSpec(("uvx", "athome-ocr-paddle")))
        >>> await worker.call("read", image)
    """

    spec: WorkerSpec
    lock: Lock = field(default_factory=Lock)
    process: Process | None = None
    stdout: BufferedByteReceiveStream | None = None
    fingerprint: dict[str, Wire] = field(default_factory=dict)
    stderr_tail: deque[bytes] = field(default_factory=lambda: deque(maxlen=STDERR_TAIL_CHUNKS))
    stderr_thread: threading.Thread | None = None

    async def call(self, method: str, payload: Wire) -> Wire:
        async with self.lock:
            await self.ensure_spawned()
            await self.process.stdin.send(encode({"method": method, "payload": payload}))
            match await self.next_frame():
                case {"ok": result}:
                    return result
                case {"err": str() as message}:
                    raise WorkerError(message)
                case frame:
                    raise WorkerError(f"malformed reply frame: {frame!r}")

    async def aclose(self) -> None:
        if self.process is None:
            return
        process, self.process = self.process, None
        await process.stdin.aclose()
        with anyio.move_on_after(ACLOSE_TIMEOUT):
            await process.wait()
        if process.returncode is None:
            process.kill()
            await process.wait()
        if self.stderr_thread is not None:
            await anyio.to_thread.run_sync(self.stderr_thread.join, STDERR_JOIN_TIMEOUT)
        self.stdout = None
        self.stderr_thread = None

    async def ensure_spawned(self) -> None:
        if self.process is not None:
            return
        read_fd, write_fd = os.pipe()
        self.process = await anyio.open_process(
            self.spec.command, stderr=write_fd, cwd=self.spec.cwd, env=self.child_env()
        )
        os.close(write_fd)
        self.stdout = BufferedByteReceiveStream(self.process.stdout)
        self.stderr_thread = threading.Thread(target=self.pump_stderr, args=(read_fd,), daemon=True)
        self.stderr_thread.start()
        await self.handshake()

    async def handshake(self) -> None:
        match await self.next_frame():
            case {"wire": int() as version, "fingerprint": dict() as fingerprint} if version == WIRE_VERSION:
                self.fingerprint = fingerprint
            case {"wire": version}:
                raise HandshakeMismatch(WIRE_VERSION, version)
            case frame:
                raise HandshakeMismatch(WIRE_VERSION, frame)

    async def next_frame(self) -> Wire:
        try:
            prefix = await self.stdout.receive_exactly(LENGTH_PREFIX)
            body = await self.stdout.receive_exactly(int.from_bytes(prefix, "big"))
        except anyio.IncompleteRead:
            raise await self.crashed() from None
        return decode(prefix + body)

    async def crashed(self) -> WorkerCrashed:
        await self.process.wait()
        await anyio.to_thread.run_sync(self.stderr_thread.join, STDERR_JOIN_TIMEOUT)
        return WorkerCrashed(self.process.returncode, b"".join(self.stderr_tail).decode("utf-8", "replace"))

    def child_env(self) -> dict[str, str] | None:
        return base_environ() | dict(self.spec.env) if self.spec.env else None

    def pump_stderr(self, read_fd: int) -> None:
        with os.fdopen(read_fd, "rb", buffering=0) as stream:
            while chunk := stream.read(STDERR_CHUNK):
                sys.stderr.buffer.write(chunk)
                sys.stderr.buffer.flush()
                self.stderr_tail.append(chunk)


class WorkerPool:
    """A fixed pool of :class:`PipeWorker` sidecars with digest-keyed prefetch and lease affinity.

    Each :meth:`lease` hands out one worker for exclusive use. A lease keyed to a string prefers
    the worker that :meth:`prefetch`-ed that key while it is free, falling back to any free worker.
    :meth:`prefetch` warms a free worker with one call and records the affinity, so the follow-up
    keyed lease lands on the already-warm process.

    Example:
        >>> pool = WorkerPool(WorkerSpec(("uvx", "athome-ocr-paddle")), size=4)
        >>> await pool.prefetch(digest, "read", image)
        >>> async with pool.lease(digest) as worker:
        ...     await worker.call("read", image)
    """

    def __init__(self, spec: WorkerSpec, *, size: int) -> None:
        self.spec = spec
        self.size = size
        self.workers = tuple(PipeWorker(spec) for _ in range(size))
        self.free = list(self.workers)
        self.affinity: dict[str, PipeWorker] = {}
        self.available = anyio.Semaphore(size)
        self.guard = Lock()

    async def acquire(self, key: str | None) -> PipeWorker:
        await self.available.acquire()
        async with self.guard:
            if key is not None and (worker := self.affinity.get(key)) is not None and worker in self.free:
                self.free.remove(worker)
                return worker
            return self.free.pop()

    async def release(self, worker: PipeWorker) -> None:
        async with self.guard:
            self.free.append(worker)
        self.available.release()

    @asynccontextmanager
    async def lease(self, key: str | None = None) -> AsyncIterator[WorkerTransport]:
        worker = await self.acquire(key)
        try:
            yield worker
        finally:
            await self.release(worker)

    async def prefetch(self, key: str, method: str, payload: Wire) -> None:
        worker = await self.acquire(None)
        try:
            await worker.call(method, payload)
            async with self.guard:
                self.affinity[key] = worker
        finally:
            await self.release(worker)

    async def aclose(self) -> None:
        async with anyio.create_task_group() as group:
            for worker in self.workers:
                group.start_soon(worker.aclose)


def handler_fingerprint(handler: object) -> Wire:
    provider = getattr(handler, "fingerprint", None)
    return provider() if callable(provider) else {}


def dispatch(handler: object, frame: Wire) -> Wire:
    match frame:
        case {"method": str() as method, "payload": payload}:
            try:
                return {"ok": validate(getattr(handler, method)(payload))}
            except Exception:
                return {"err": traceback.format_exc()}
        case _:
            return {"err": f"malformed request frame: {frame!r}"}


def serve(handler: object) -> None:
    """Run a sidecar's frame loop until the parent closes the pipe.

    Emits the handshake frame (wire version plus ``handler.fingerprint()`` when defined), then
    reads request frames and dispatches each to the named handler method, replying ``{"ok": ...}``
    on success or ``{"err": <traceback>}`` on any exception. Returns when stdin reaches EOF.

    Example:
        >>> class Echo:
        ...     def echo(self, payload): return payload
        >>> serve(Echo())
    """
    write_frame(sys.stdout.buffer, {"wire": WIRE_VERSION, "fingerprint": handler_fingerprint(handler)})
    while True:
        try:
            frame = read_frame(sys.stdin.buffer)
        except EOFError:
            return
        write_frame(sys.stdout.buffer, dispatch(handler, frame))
