from __future__ import annotations

import json
import socket
import subprocess
from typing import TYPE_CHECKING

import anyio
import httpx
import pytest
from click.testing import CliRunner

# starlette is the `activator` extra; skip this module cleanly on CI jobs that sync without extras.
pytest.importorskip("starlette")

from athome import serve
from athome.activator import (
    CHILD_HOST,
    CHILD_STOP_TIMEOUT_S,
    Activator,
    ActivatorSettings,
    ChildStartCooldown,
    serve_activator,
    strip_hop_headers,
)
from athome.config import load

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

MODELS_BODY = {"object": "list", "data": [{"id": "served-model", "object": "model"}]}


async def default_body() -> AsyncIterator[bytes]:
    yield b"ok"


class FakeChild:
    def __init__(self, *, returncode: int | None = None) -> None:
        self.pid = 4321
        self.returncode = returncode
        self.signals: list[str] = []

    def terminate(self) -> None:
        self.signals.append("TERM")

    def kill(self) -> None:
        self.signals.append("KILL")
        self.returncode = -9

    async def wait(self) -> int | None:
        return self.returncode


def build(
    monkeypatch: pytest.MonkeyPatch,
    *,
    spawned: list[FakeChild],
    child_factory: Callable[[], FakeChild] | None = None,
    body: Callable[[], AsyncIterator[bytes]] | None = None,
    seen_requests: list[httpx.Request] | None = None,
    healthy: bool = True,
    wake_paths: tuple[str, ...] | None = None,
) -> Activator:
    overrides = {"wake_paths": wake_paths} if wake_paths is not None else {}
    settings = ActivatorSettings(command="child {LISTEN_FD}", child_port=18999, **overrides)
    activator = Activator(settings)

    async def fake_launch() -> FakeChild:
        child = (child_factory or FakeChild)()
        spawned.append(child)
        return child

    monkeypatch.setattr(activator, "_launch_child", fake_launch)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen_requests is not None:
            seen_requests.append(request)
        if request.url.path == "/v1/models" and request.method == "GET":
            return httpx.Response(200 if healthy else 503, json=MODELS_BODY)
        return httpx.Response(200, content=(body or default_body)())

    activator.upstream = httpx.AsyncClient(base_url="http://child", transport=httpx.MockTransport(handler))
    return activator


def asgi_client(activator: Activator) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=activator.app), base_url="http://proxy")


def test_env_overrides_flow_through_the_section_machinery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATHOME_SERVE_ACTIVATOR_COMMAND", "rapid-mlx --listen-fd {LISTEN_FD}")
    monkeypatch.setenv("ATHOME_SERVE_ACTIVATOR_PORT", "9001")
    monkeypatch.setenv("ATHOME_SERVE_ACTIVATOR_CHILD_PORT", "19001")
    monkeypatch.setenv("ATHOME_SERVE_ACTIVATOR_IDLE_S", "42.5")
    monkeypatch.setenv("ATHOME_SERVE_ACTIVATOR_WAKE_CONCURRENCY", "3")
    monkeypatch.setenv("ATHOME_SERVE_ACTIVATOR_SPAWN_COOLDOWN_S", "7")
    load.cache_clear()

    settings = load(ActivatorSettings)

    assert settings.command == "rapid-mlx --listen-fd {LISTEN_FD}"
    assert settings.port == 9001
    assert settings.child_port == 19001
    assert settings.idle_s == 42.5
    assert settings.wake_concurrency == 3
    assert settings.spawn_cooldown_s == 7.0
    assert settings.host == "127.0.0.1"
    assert settings.upstream_timeout_s == 600.0


def test_strip_hop_headers_drops_hop_and_connection_listed_headers() -> None:
    result = strip_hop_headers(
        [
            ("Connection", "keep-alive, X-Custom"),
            ("Keep-Alive", "timeout=5"),
            ("X-Custom", "secret"),
            ("Transfer-Encoding", "chunked"),
            ("Host", "proxy"),
            ("Content-Type", "application/json"),
            ("X-Keep", "1"),
        ]
    )
    names = {name.lower() for name, _ in result}
    assert names == {"content-type", "x-keep"}


async def test_probe_while_down_replays_cached_models_and_never_spawns(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: list[FakeChild] = []
    activator = build(monkeypatch, spawned=spawned)
    async with asgi_client(activator) as client:
        pre = await client.get("/v1/models")
        assert pre.status_code == 200
        assert pre.json() == {"object": "list", "data": []}

        activator._models_path().parent.mkdir(parents=True, exist_ok=True)
        activator._models_path().write_text(httpx.Response(200, json=MODELS_BODY).text)

        replayed = await client.get("/v1/models")
        assert replayed.json() == MODELS_BODY

        health = await client.get("/health")
        assert health.json() == {"status": "ok", "model": "idle"}

    assert spawned == []
    assert activator.resource.loaded is False


async def test_a_non_allowlisted_path_404s_and_never_spawns(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: list[FakeChild] = []
    activator = build(monkeypatch, spawned=spawned)
    async with asgi_client(activator) as client:
        missing = await client.get("/v1/embeddings")
        wrong_method = await client.get("/v1/chat/completions")
    assert missing.status_code == 404
    assert wrong_method.status_code == 405
    assert spawned == []
    assert activator.resource.loaded is False


def test_wake_paths_defaults_to_the_chat_completion_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATHOME_SERVE_ACTIVATOR_COMMAND", "child {LISTEN_FD}")
    load.cache_clear()
    assert load(ActivatorSettings).wake_paths == ("/v1/chat/completions", "/v1/completions", "/v1/messages")


def test_wake_paths_env_override_parses_a_json_array(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATHOME_SERVE_ACTIVATOR_COMMAND", "child {LISTEN_FD}")
    monkeypatch.setenv("ATHOME_SERVE_ACTIVATOR_WAKE_PATHS", '["/v1/audio/transcriptions"]')
    load.cache_clear()
    assert load(ActivatorSettings).wake_paths == ("/v1/audio/transcriptions",)


async def test_a_custom_wake_path_wakes_while_the_old_default_404s(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: list[FakeChild] = []
    activator = build(monkeypatch, spawned=spawned, wake_paths=("/v1/audio/transcriptions",))
    async with asgi_client(activator) as client:
        woke = await client.post("/v1/audio/transcriptions", content=b"{}")
        stale = await client.post("/v1/chat/completions", content=b"{}")
    assert woke.status_code == 200
    assert len(spawned) == 1
    assert stale.status_code == 404


async def test_two_concurrent_wakes_trigger_exactly_one_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: list[FakeChild] = []
    activator = build(monkeypatch, spawned=spawned)

    async def one(client: httpx.AsyncClient) -> None:
        response = await client.post("/v1/chat/completions", content=b"{}")
        assert response.status_code == 200

    async with asgi_client(activator) as client:
        async with anyio.create_task_group() as tg:
            tg.start_soon(one, client)
            tg.start_soon(one, client)

    assert len(spawned) == 1
    assert activator.resource.inflight == 0
    assert activator.resource.loaded is True


async def test_wake_persists_the_models_body_seen_at_health(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: list[FakeChild] = []
    activator = build(monkeypatch, spawned=spawned)
    async with asgi_client(activator) as client:
        await client.post("/v1/chat/completions", content=b"{}")
    assert json.loads(activator._models_path().read_text()) == MODELS_BODY
    assert activator.resource.value is not None
    assert activator.resource.value.models == MODELS_BODY


async def test_sweep_during_an_open_streamed_response_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: list[FakeChild] = []
    observed: dict[str, object] = {}

    async def body() -> AsyncIterator[bytes]:
        yield b"chunk-1 "
        observed["inflight_mid"] = activator.resource.inflight
        await activator.resource.sweep(now=activator.resource.last_done + 1_000_000.0)
        observed["loaded_mid"] = activator.resource.loaded
        yield b"chunk-2 "

    activator = build(monkeypatch, spawned=spawned, body=body)
    async with asgi_client(activator) as client:
        response = await client.post("/v1/chat/completions", content=b"{}")

    assert response.content == b"chunk-1 chunk-2 "
    assert observed == {"inflight_mid": 1, "loaded_mid": True}
    assert activator.resource.inflight == 0
    assert activator.resource.loaded is True
    assert len(spawned) == 1


async def test_a_crashed_child_is_discarded_then_respawned_on_the_next_wake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned: list[FakeChild] = []
    activator = build(monkeypatch, spawned=spawned)
    async with asgi_client(activator) as client:
        await client.post("/v1/chat/completions", content=b"{}")
        assert len(spawned) == 1
        assert activator.resource.loaded is True

        spawned[0].returncode = 1  # crashed underneath us

        await client.post("/v1/chat/completions", content=b"{}")

    assert len(spawned) == 2
    assert spawned[0].signals == []  # already dead: no SIGTERM to a corpse
    assert activator.process is spawned[1]
    assert activator.resource.loaded is True


async def test_reap_crashed_discards_a_child_that_died_underneath_us(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: list[FakeChild] = []
    activator = build(monkeypatch, spawned=spawned)
    async with asgi_client(activator) as client:
        await client.post("/v1/chat/completions", content=b"{}")
    assert activator.resource.loaded is True

    spawned[0].returncode = 137
    await activator.reap_crashed()

    assert activator.resource.loaded is False
    assert activator.process is None


async def test_a_failed_spawn_arms_the_cooldown_and_the_next_wake_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned: list[FakeChild] = []
    activator = build(monkeypatch, spawned=spawned, child_factory=lambda: FakeChild(returncode=1))

    async with asgi_client(activator) as client:
        first = await client.post("/v1/chat/completions", content=b"{}")
        assert first.status_code == 503
        assert len(spawned) == 1  # launched, exited during startup, cooldown armed

        second = await client.post("/v1/chat/completions", content=b"{}")
        assert second.status_code == 503

    assert len(spawned) == 1  # cooldown gate refused before any second launch
    assert activator.cooldown_until > 0.0


async def test_cooldown_gate_raises_before_launching_within_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: list[FakeChild] = []
    activator = build(monkeypatch, spawned=spawned)
    activator.cooldown_until = anyio.current_time() + 30.0
    with pytest.raises(ChildStartCooldown):
        await activator._spawn()
    assert spawned == []


async def test_a_sigterm_ignoring_child_is_sigkilled_only_after_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activator = build(monkeypatch, spawned=[])
    child = FakeChild()  # returncode stays None: ignores SIGTERM
    activator.process = child

    clock = {"t": 1000.0}
    monkeypatch.setattr(anyio, "current_time", lambda: clock["t"])

    async def fake_sleep(seconds: float) -> None:
        clock["t"] += seconds
        await anyio.lowlevel.checkpoint()

    monkeypatch.setattr(anyio, "sleep", fake_sleep)

    kill_at: dict[str, float] = {}
    real_kill = child.kill

    def record_kill() -> None:
        kill_at["t"] = clock["t"]
        real_kill()

    monkeypatch.setattr(child, "kill", record_kill)

    await activator._stop()

    assert child.signals == ["TERM", "KILL"]
    assert kill_at["t"] == 1000.0 + CHILD_STOP_TIMEOUT_S  # killed exactly at the window boundary, not before
    assert activator.process is None


async def test_stop_reaps_a_child_that_exits_gracefully_without_sigkill(monkeypatch: pytest.MonkeyPatch) -> None:
    activator = build(monkeypatch, spawned=[])

    class GracefulChild(FakeChild):
        def terminate(self) -> None:
            self.signals.append("TERM")
            self.returncode = 0  # exits promptly on SIGTERM

    activator.process = GracefulChild()
    await activator._stop()
    assert activator.process is None


def test_the_serve_group_registers_the_activator_command() -> None:
    result = CliRunner().invoke(serve.cli, ["activator", "--help"])
    assert result.exit_code == 0, result.output
    assert "--host" in result.output


def test_serve_activator_overrides_the_host_and_disables_access_log(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATHOME_SERVE_ACTIVATOR_COMMAND", "child {LISTEN_FD}")
    monkeypatch.setenv("ATHOME_SERVE_ACTIVATOR_PORT", "8123")
    load.cache_clear()
    captured: dict[str, object] = {}

    def fake_run(app: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)

    serve_activator(host="0.0.0.0")
    assert captured == {"host": "0.0.0.0", "port": 8123, "log_level": "info", "access_log": False}

    captured.clear()
    serve_activator(host=None)
    assert captured["host"] == "127.0.0.1"


async def test_launch_child_hands_inherited_stdio_never_pipes(monkeypatch: pytest.MonkeyPatch) -> None:
    activator = Activator(ActivatorSettings(command="child {LISTEN_FD}", child_port=18998))
    recorded: dict[str, object] = {}

    monkeypatch.setattr(activator, "_listener", lambda: type("S", (), {"fileno": lambda self: 7})())

    async def fake_open_process(argv: list[str], **kwargs: object) -> FakeChild:
        recorded["argv"], recorded["kwargs"] = argv, kwargs
        return FakeChild()

    monkeypatch.setattr(anyio, "open_process", fake_open_process)

    child = await activator._launch_child()

    assert isinstance(child, FakeChild)
    assert recorded["argv"] == ["child", "7"]
    kwargs = recorded["kwargs"]
    assert kwargs["pass_fds"] == (7,)
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is None
    assert kwargs["stderr"] is None
    # an undrained PIPE deadlocks a chatty child; none of the three may be PIPE.
    assert subprocess.PIPE not in kwargs.values()


async def test_stop_does_not_null_a_concurrently_respawned_child(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: list[FakeChild] = []
    activator = build(monkeypatch, spawned=spawned)

    stale = FakeChild(returncode=1)  # crashed underneath us
    activator.process = stale
    reached, respawned = anyio.Event(), anyio.Event()

    async def parking_wait() -> int | None:
        reached.set()  # _stop has captured `stale` and reached wait()
        await respawned.wait()  # park until a fresh child has been installed
        return stale.returncode

    monkeypatch.setattr(stale, "wait", parking_wait)

    async with anyio.create_task_group() as tg:
        tg.start_soon(activator._stop)  # stops the stale child; parks in wait()
        await reached.wait()
        async with activator.resource.use():  # concurrent respawn installs a fresh child, loads the resource
            pass
        fresh = activator.process
        respawned.set()  # let the stale _stop resume

    assert stale.signals == []  # already dead: no SIGTERM to a corpse
    assert fresh is spawned[-1]
    assert activator.process is fresh  # fix: the stale _stop did not null the fresh child
    assert activator.resource.loaded is True


async def test_wake_with_a_non_ascii_header_errors_without_leaking_the_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned: list[FakeChild] = []

    class GracefulChild(FakeChild):
        def terminate(self) -> None:
            self.signals.append("TERM")
            self.returncode = 0

    activator = build(monkeypatch, spawned=spawned, child_factory=GracefulChild)

    def inject_bad_header(app: object) -> object:
        async def wrapper(scope: dict, receive: object, send: object) -> None:
            if scope["type"] == "http":
                scope = {**scope, "headers": [*scope["headers"], (b"x-model", b"\xff\xfe")]}
            await app(scope, receive, send)

        return wrapper

    bad_transport = httpx.ASGITransport(app=inject_bad_header(activator.app), raise_app_exceptions=False)
    bad = httpx.AsyncClient(transport=bad_transport, base_url="http://proxy")
    async with bad, asgi_client(activator) as good:
        errored = await bad.post("/v1/chat/completions", content=b"{}")
        assert errored.status_code == 500
        assert activator.resource.inflight == 0
        assert activator.wake_slots.value == activator.settings.wake_concurrency  # wake slot released, not leaked

        # a follow-up (clean-header) wake proceeds (semaphore intact) and streams normally.
        ok = await good.post("/v1/chat/completions", content=b"{}")
        assert ok.status_code == 200
        assert ok.content == b"ok"

    assert activator.resource.inflight == 0
    # sweep can still unload the child the failed wake left loaded.
    await activator.resource.sweep(now=activator.resource.last_done + 1_000_000.0)
    assert activator.resource.loaded is False


async def test_lifespan_startup_fails_loudly_when_the_child_port_is_squatted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activator = build(monkeypatch, spawned=[])
    squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    squatter.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    squatter.bind((CHILD_HOST, activator.settings.child_port))
    squatter.listen(1)
    try:
        with pytest.raises(OSError):
            async with activator._lifespan(activator.app):
                pass
    finally:
        squatter.close()
        if activator.listen_sock is not None:
            activator.listen_sock.close()


async def test_persist_models_writes_atomically_and_leaves_no_temp(monkeypatch: pytest.MonkeyPatch) -> None:
    activator = build(monkeypatch, spawned=[])
    await activator._persist_models(MODELS_BODY)
    path = activator._models_path()
    assert json.loads(path.read_text()) == MODELS_BODY
    # the atomic write stages a temp sibling then os.replace's it — no temp may remain.
    assert [p.name for p in path.parent.iterdir() if p.name != path.name] == []
