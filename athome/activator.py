"""A probe-safe idle-unload proxy: a reusable front over :class:`~athome.idle.IdleResource`.

The activator binds a public address and lazily manages one OpenAI-compatible child
server on a listener socket it binds itself and hands the child at spawn (``{LISTEN_FD}``),
so nothing else can squat the child's port between restarts. Only an explicit route
allowlist is served: probes (``/health``, ``/v1/models``) are answered locally while the
child is down so they never wake it, and the wake routes (chat/completions/messages) spawn
the child under :class:`~athome.idle.IdleResource`'s single-flight load, then reverse-proxy
with unbuffered streaming bounded by ``wake_concurrency`` and ``upstream_timeout_s``. Every
other path is a local 404 — never proxied, never a wake. A failed spawn refuses wakes for
``spawn_cooldown_s`` instead of respawn-thrashing, and the maintenance loop unloads the
child once it has sat idle past ``idle_s`` with zero requests in flight — SIGTERM first,
never SIGKILL, so the child's graceful shutdown (prefix-cache save; avoiding a wired-Metal
teardown pathology on MLX) runs.
"""

from __future__ import annotations

import json
import shlex
import socket
import subprocess
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import anyio
import httpx
from loguru import logger

from athome.cache import atomic_write_text
from athome.config import AthomeSettings, SectionSettings, load
from athome.errors import AthomeError
from athome.idle import IdleResource

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable
    from pathlib import Path

    from anyio.abc import Process
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response

CHILD_HOST = "127.0.0.1"
CHILD_LISTEN_BACKLOG = 128
WAKE_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/messages")
MAINTENANCE_INTERVAL_S = 30.0
CHILD_STOP_TIMEOUT_S = 120.0
CHILD_STOP_POLL_S = 1.0
HEALTH_POLL_S = 1.0
HEALTH_ATTEMPT_TIMEOUT_S = 2.0
EMPTY_MODELS: dict[str, object] = {"object": "list", "data": []}
HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
    }
)


class ActivatorError(AthomeError):
    """Root of every activator error surfaced through the proxy."""


class ChildStartError(ActivatorError):
    """The child failed to become healthy after a spawn; wakes surface it as 503."""


class ChildStartTimeout(ChildStartError):
    def __init__(self, timeout: float) -> None:
        super().__init__(f"child not healthy after {timeout:.0f}s")


class ChildExitedDuringStartup(ChildStartError):
    def __init__(self, returncode: int) -> None:
        super().__init__(f"child exited rc={returncode} during startup")


class ChildStartCooldown(ChildStartError):
    def __init__(self, remaining: float) -> None:
        super().__init__(f"spawn cooling down for another {remaining:.0f}s after a failed start")


class ActivatorSettings(SectionSettings):
    """The ``[serve.activator]`` section: the child command and the proxy's bind + policy knobs.

    Env overrides derive from the section as ``ATHOME_SERVE_ACTIVATOR_<FIELD>`` (init kwargs >
    env > ``~/.athome/config.toml`` > defaults). ``command`` is the child's argv template; the
    literal token ``{LISTEN_FD}`` is substituted with the inherited listener fd number so the
    child binds the socket the activator already holds.

    Example:
        >>> settings = load(ActivatorSettings)  # doctest: +SKIP
        >>> Activator(settings).app  # doctest: +SKIP
    """

    section: ClassVar[tuple[str, ...]] = ("serve", "activator")
    command: str
    host: str = "127.0.0.1"
    port: int = 8000
    child_port: int = 18000
    idle_s: float = 1800.0
    start_timeout_s: float = 150.0
    wake_concurrency: int = 8
    upstream_timeout_s: float = 600.0
    spawn_cooldown_s: float = 30.0


@dataclass(frozen=True, slots=True)
class ChildHandle:
    process: Process
    models: dict[str, object]


def strip_hop_headers(headers: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    drop = set(HOP_HEADERS)
    for name, value in (pairs := list(headers)):
        if name.lower() == "connection":
            drop.update(token.strip().lower() for token in value.split(",") if token.strip())
    return [(name, value) for name, value in pairs if name.lower() not in drop]


class Activator:
    """The single-policy idle-unload proxy: one :class:`~athome.idle.IdleResource`, an allowlist app.

    All proxy-specific behavior — the cooldown gate, the pre-bound listener fd handoff, the
    probe/wake routes, the streaming relay whose inflight count spans the whole response body —
    lives in this class's own callables and routes, never a second lifecycle. The resource wraps
    :meth:`_spawn`/:meth:`_stop`, so concurrent first wakes collapse to one spawn and the reaper
    never unloads a child a stream still holds.

    Example:
        >>> activator = Activator(load(ActivatorSettings))  # doctest: +SKIP
        >>> uvicorn.run(activator.app, host="127.0.0.1", port=8000)  # doctest: +SKIP
    """

    def __init__(self, settings: ActivatorSettings) -> None:
        self.settings = settings
        self.resource: IdleResource[ChildHandle] = IdleResource(self._spawn, self._stop, ttl_s=settings.idle_s)
        self.upstream = httpx.AsyncClient(
            base_url=f"http://{CHILD_HOST}:{settings.child_port}",
            timeout=httpx.Timeout(connect=5.0, read=None, write=None, pool=None),
            # ambient HTTP_PROXY/ALL_PROXY must never re-route loopback prompt traffic.
            trust_env=False,
        )
        self.wake_slots = anyio.Semaphore(settings.wake_concurrency)
        self.process: Process | None = None
        self.listen_sock: socket.socket | None = None
        self.cooldown_until = 0.0
        self._app: Starlette | None = None

    @property
    def app(self) -> Starlette:
        if self._app is None:
            self._app = self._build_app()
        return self._app

    def _build_app(self) -> Starlette:
        from starlette.applications import Starlette
        from starlette.routing import Route

        return Starlette(
            routes=[
                Route("/health", self.probe, methods=["GET"]),
                Route("/v1/models", self.probe, methods=["GET"]),
                *(Route(path, self.wake, methods=["POST"]) for path in WAKE_PATHS),
            ],
            lifespan=self._lifespan,
        )

    @asynccontextmanager
    async def _lifespan(self, app: Starlette) -> AsyncIterator[None]:
        # Bind eagerly at startup: a squatted child port fails the activator loudly here, not per-wake.
        self._listener()
        logger.info(
            "activator up on {}:{} (child listener {}:{}, idle {:.0f}s)",
            self.settings.host,
            self.settings.port,
            CHILD_HOST,
            self.settings.child_port,
            self.settings.idle_s,
        )
        async with anyio.create_task_group() as tg:
            tg.start_soon(self._maintain)
            try:
                yield
            finally:
                tg.cancel_scope.cancel()
        await self._stop()
        if self.listen_sock is not None:
            self.listen_sock.close()
            self.listen_sock = None
        await self.upstream.aclose()
        logger.info("activator exiting")

    def child_up(self) -> bool:
        return self.process is not None and self.process.returncode is None

    def _models_path(self) -> Path:
        return load(AthomeSettings).cache_root / "activator" / "models.json"

    def _replay_models(self) -> dict[str, object]:
        path = self._models_path()
        return json.loads(path.read_text()) if path.exists() else EMPTY_MODELS

    def _listener(self) -> socket.socket:
        # Bound once at lifespan startup and held for the activator's lifetime; the child inherits this fd
        # on every (re)spawn, so nothing else can squat the child port between restarts.
        if self.listen_sock is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((CHILD_HOST, self.settings.child_port))
            sock.listen(CHILD_LISTEN_BACKLOG)
            self.listen_sock = sock
        return self.listen_sock

    async def _launch_child(self) -> Process:
        fd = self._listener().fileno()
        argv = shlex.split(self.settings.command.replace("{LISTEN_FD}", str(fd)))
        # Inherit the activator's stdout/stderr (launchd log): anyio's default PIPE, undrained, deadlocks a
        # chatty child once it fills. DEVNULL stdin so the child never reads it.
        return await anyio.open_process(argv, pass_fds=(fd,), stdin=subprocess.DEVNULL, stdout=None, stderr=None)

    async def _spawn(self) -> ChildHandle:
        if (remaining := self.cooldown_until - anyio.current_time()) > 0:
            raise ChildStartCooldown(remaining)
        self.process = process = await self._launch_child()
        logger.info("spawned child pid={}: {}", process.pid, self.settings.command)
        try:
            models = await self._wait_healthy(process)
        except ChildStartError:
            self.cooldown_until = anyio.current_time() + self.settings.spawn_cooldown_s
            await self._stop()
            raise
        await self._persist_models(models)
        return ChildHandle(process=process, models=models)

    async def _wait_healthy(self, process: Process) -> dict[str, object]:
        deadline = anyio.current_time() + self.settings.start_timeout_s
        while anyio.current_time() < deadline:
            if process.returncode is not None:
                raise ChildExitedDuringStartup(process.returncode)
            try:
                # Each attempt needs its own read timeout: connects sit in the listener backlog until
                # the child accepts, so the GET only succeeds once the child actually serves.
                response = await self.upstream.get("/v1/models", timeout=HEALTH_ATTEMPT_TIMEOUT_S)
            except httpx.TransportError:
                pass
            else:
                if response.status_code == 200:
                    return response.json()
            await anyio.sleep(HEALTH_POLL_S)
        raise ChildStartTimeout(self.settings.start_timeout_s)

    async def _persist_models(self, models: dict[str, object]) -> None:
        # Atomic: a mid-write kill must never leave truncated JSON that _replay_models' json.loads then 500s on.
        await atomic_write_text(anyio.Path(self._models_path()), json.dumps(models))

    async def _stop(self) -> None:
        if (process := self.process) is None:
            return
        if process.returncode is None:
            # SIGTERM, never SIGKILL first: graceful shutdown saves the prefix cache and avoids a
            # ~20GB wired-Metal teardown pathology on MLX.
            logger.info("stopping child pid={} (SIGTERM)", process.pid)
            process.terminate()
        deadline = anyio.current_time() + CHILD_STOP_TIMEOUT_S
        while process.returncode is None:
            if anyio.current_time() >= deadline:
                logger.error(
                    "child pid={} ignored SIGTERM for {:.0f}s; SIGKILL as last resort",
                    process.pid,
                    CHILD_STOP_TIMEOUT_S,
                )
                process.kill()
                break
            await anyio.sleep(CHILD_STOP_POLL_S)
        await process.wait()
        logger.info("child pid={} exited rc={}", process.pid, process.returncode)
        # Only null our own child: a concurrent _spawn (not mutually excluded with sweep/discard) may have
        # already installed a fresh child while we awaited wait(); nulling it would strand the live process.
        if self.process is process:
            self.process = None

    async def reap_crashed(self) -> None:
        if self.resource.loaded and self.process is not None and self.process.returncode is not None:
            logger.warning("child pid={} crashed rc={}; discarding", self.process.pid, self.process.returncode)
            await self.resource.discard()

    async def _maintain(self) -> None:
        # The reap must precede each sweep — a child that crashed underneath us has to be discarded
        # before the idle sweep, hence this loop rather than IdleResource.run.
        while True:
            await anyio.sleep(MAINTENANCE_INTERVAL_S)
            try:
                await self.reap_crashed()
                await self.resource.sweep()
            except Exception:
                logger.exception("activator maintenance tick failed; continuing")

    async def probe(self, request: Request) -> Response:
        from starlette.responses import JSONResponse

        if self.child_up():
            return await self._proxy(request)
        if request.url.path == "/v1/models":
            return JSONResponse(self._replay_models())
        return JSONResponse({"status": "ok", "model": "idle"})

    async def wake(self, request: Request) -> Response:
        from starlette.responses import JSONResponse

        async with AsyncExitStack() as guard:
            await guard.enter_async_context(self.wake_slots)
            await self.reap_crashed()
            try:
                # use() loads (single-flight spawn) then counts this request in flight; that count is
                # handed to the streamed response below so it spans the entire body.
                await guard.enter_async_context(self.resource.use())
            except ChildStartError as exc:
                logger.error("wake failed: {}", exc)
                return JSONResponse({"error": str(exc)}, status_code=503)
            return await self._proxy(request, stack=guard.pop_all())

    async def _proxy(self, request: Request, *, stack: AsyncExitStack | None = None) -> Response:
        from starlette.background import BackgroundTask
        from starlette.responses import JSONResponse, StreamingResponse

        if stack is None:
            stack = AsyncExitStack()
        # build_request runs inside the guard too: a non-ASCII request header makes httpx raise before the
        # send, and an un-guarded raise here would leak the handed stack (pinning inflight + a wake slot).
        try:
            deadline = anyio.current_time() + self.settings.upstream_timeout_s
            upstream_request = self.upstream.build_request(
                request.method,
                httpx.URL(path=request.url.path, query=request.url.query.encode()),
                headers=strip_hop_headers((k.decode("latin-1"), v.decode("latin-1")) for k, v in request.headers.raw),
                content=request.stream(),
            )
            with anyio.fail_after(deadline - anyio.current_time()):
                response = await self.upstream.send(upstream_request, stream=True)
        except TimeoutError:
            logger.error("upstream gave no response within {:.0f}s; dropped", self.settings.upstream_timeout_s)
            await stack.aclose()
            return JSONResponse({"error": "upstream timeout"}, status_code=504)
        except BaseException:
            await stack.aclose()
            raise
        stack.push_async_callback(response.aclose)
        return StreamingResponse(
            self._relay(response, stack, deadline),
            status_code=response.status_code,
            headers=dict(strip_hop_headers(response.headers.items())),
            # Runs after the body drains AND on client disconnect (Starlette cancels the stream,
            # absorbs it in its own scope, then runs background), releasing the inflight count once.
            background=BackgroundTask(stack.aclose),
        )

    async def _relay(self, response: httpx.Response, stack: AsyncExitStack, deadline: float) -> AsyncIterator[bytes]:
        try:
            chunks = response.aiter_raw()
            while True:
                try:
                    with anyio.fail_after(deadline - anyio.current_time()):
                        chunk = await anext(chunks)
                except StopAsyncIteration:
                    break
                except TimeoutError:
                    logger.error("upstream stream exceeded {:.0f}s; truncating", self.settings.upstream_timeout_s)
                    break
                yield chunk
        finally:
            # Belt for the one case background cannot reach — send() raising mid-body: the generator's
            # finalization closes the same idempotent stack, so the inflight count never leaks.
            await stack.aclose()


def serve_activator(*, host: str | None = None) -> None:
    """Run the activator proxy under uvicorn, binding ``host`` (or the configured host) and its port.

    Args:
        host: Bind address overriding ``ActivatorSettings.host``; ``None`` uses the configured host.
    """
    import uvicorn

    settings = load(ActivatorSettings)
    activator = Activator(settings)
    # access_log=False: probes hit every few seconds and their request lines can leak prompt metadata.
    uvicorn.run(activator.app, host=host or settings.host, port=settings.port, log_level="info", access_log=False)
