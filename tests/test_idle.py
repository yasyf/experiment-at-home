from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import httpx
import pytest

from athome import serve
from athome.config import load
from athome.detach import DetachedRun
from athome.idle import IdleResource
from athome.serve import ManagedServer

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture(autouse=True)
def configure_mlx_vlm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATHOME_SERVE_MLX_VLM_VERSION", "0.3.4")
    load.cache_clear()


def counting_resource(*, ttl_s: float, result: object = "handle") -> tuple[IdleResource[object], dict[str, int]]:
    stats = {"load": 0, "unload": 0}

    async def load() -> object:
        stats["load"] += 1
        await anyio.sleep(0)
        return result

    async def unload() -> None:
        stats["unload"] += 1

    return IdleResource(load, unload, ttl_s=ttl_s), stats


async def checkpoint_sleep(_seconds: float) -> None:
    await anyio.lowlevel.checkpoint()


def spy_on_sweeps(monkeypatch: pytest.MonkeyPatch) -> Callable[[], int]:
    count = 0
    real_sweep = IdleResource.sweep

    async def spy(self: IdleResource[object], *, now: float | None = None) -> None:
        nonlocal count
        count += 1
        await real_sweep(self, now=now)

    monkeypatch.setattr(IdleResource, "sweep", spy)
    return lambda: count


async def drive_reaper(resource: IdleResource[object], sweeps: Callable[[], int], *, until: int) -> None:
    async with anyio.create_task_group() as tg:
        tg.start_soon(resource.run)
        for _ in range(1000):
            if sweeps() >= until:
                break
            await anyio.lowlevel.checkpoint()
        tg.cancel_scope.cancel()


def health_sequence(*results: bool) -> Callable[[ManagedServer], object]:
    calls = iter(results)

    async def fake(self: ManagedServer) -> bool:
        return next(calls)

    return fake


def mock_models(monkeypatch: pytest.MonkeyPatch, *ids: str) -> None:
    body = {"object": "list", "data": [{"id": model_id, "object": "model"} for model_id in ids]}
    monkeypatch.setattr(
        serve,
        "health_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))),
    )


async def test_concurrent_first_uses_trigger_exactly_one_load() -> None:
    resource, stats = counting_resource(ttl_s=10.0)

    async def user() -> None:
        async with resource.use() as value:
            assert value == "handle"

    async with anyio.create_task_group() as tg:
        tg.start_soon(user)
        tg.start_soon(user)

    assert stats["load"] == 1
    assert resource.inflight == 0


async def test_second_use_without_unload_does_not_reload() -> None:
    resource, stats = counting_resource(ttl_s=10.0)
    async with resource.use():
        pass
    async with resource.use():
        pass
    assert stats["load"] == 1


async def test_sweep_refuses_before_ttl() -> None:
    resource, stats = counting_resource(ttl_s=100.0)
    async with resource.use():
        pass
    await resource.sweep(now=resource.last_done + 99.0)
    assert stats["unload"] == 0
    assert resource.loaded is True


async def test_sweep_refuses_while_use_is_inflight() -> None:
    resource, stats = counting_resource(ttl_s=100.0)
    async with resource.use():
        await resource.sweep(now=resource.last_done + 1_000_000.0)
        assert stats["unload"] == 0
        assert resource.inflight == 1
        assert resource.loaded is True


async def test_sweep_on_unloaded_resource_is_a_noop() -> None:
    resource, stats = counting_resource(ttl_s=100.0)
    await resource.sweep(now=1_000_000.0)
    assert stats == {"load": 0, "unload": 0}
    assert resource.loaded is False


async def test_sweep_after_ttl_unloads_with_value_already_dropped() -> None:
    seen: dict[str, object] = {}

    async def load() -> object:
        return object()

    async def unload() -> None:
        seen["calls"] = int(seen.get("calls", 0)) + 1
        seen["loaded"] = resource.loaded
        seen["value"] = resource.value

    resource: IdleResource[object] = IdleResource(load, unload, ttl_s=100.0)
    async with resource.use():
        pass
    assert resource.loaded is True

    await resource.sweep(now=resource.last_done + 101.0)

    assert seen == {"calls": 1, "loaded": False, "value": None}
    assert resource.loaded is False


async def test_use_after_sweep_reloads() -> None:
    resource, stats = counting_resource(ttl_s=100.0)
    async with resource.use():
        pass
    await resource.sweep(now=resource.last_done + 101.0)
    assert resource.loaded is False
    async with resource.use() as value:
        assert value == "handle"
    assert stats["load"] == 2


async def test_raising_load_propagates_and_next_use_retries() -> None:
    calls = 0

    async def load() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")
        return "handle"

    async def unload() -> None:
        raise AssertionError("a failed load must not unload")

    resource: IdleResource[str] = IdleResource(load, unload, ttl_s=10.0)

    with pytest.raises(RuntimeError, match="boom"):
        async with resource.use():
            pass
    assert resource.loaded is False
    assert resource.inflight == 0

    async with resource.use() as value:
        assert value == "handle"
    assert calls == 2


async def test_discard_unloads_before_ttl() -> None:
    resource, stats = counting_resource(ttl_s=1000.0)
    async with resource.use():
        pass
    await resource.sweep(now=resource.last_done + 1.0)
    assert stats["unload"] == 0

    await resource.discard()
    assert stats["unload"] == 1
    assert resource.loaded is False


async def test_discard_bypasses_the_inflight_count() -> None:
    resource, stats = counting_resource(ttl_s=1000.0)
    async with resource.use():
        await resource.discard()
        assert stats["unload"] == 1
        assert resource.loaded is False
    assert resource.inflight == 0


async def test_run_loop_sweeps_after_each_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    resource, _ = counting_resource(ttl_s=100.0)
    sweeps = spy_on_sweeps(monkeypatch)
    monkeypatch.setattr(anyio, "sleep", checkpoint_sleep)

    await drive_reaper(resource, sweeps, until=2)

    assert sweeps() >= 2


async def test_run_loop_survives_a_sweep_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    unloads = 0

    async def load() -> object:
        return "handle"

    async def unload() -> None:
        nonlocal unloads
        unloads += 1
        if unloads == 1:
            raise RuntimeError("unload boom")

    resource: IdleResource[object] = IdleResource(load, unload, ttl_s=0.0)
    async with resource.use():
        pass
    assert resource.loaded is True

    sweeps = spy_on_sweeps(monkeypatch)
    monkeypatch.setattr(anyio, "sleep", checkpoint_sleep)

    await drive_reaper(resource, sweeps, until=2)

    assert unloads == 1
    assert sweeps() >= 2
    assert resource.loaded is False


async def test_run_loop_exits_cleanly_on_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    resource, _ = counting_resource(ttl_s=100.0)
    sweeps = spy_on_sweeps(monkeypatch)
    monkeypatch.setattr(anyio, "sleep", checkpoint_sleep)

    await drive_reaper(resource, sweeps, until=1)

    assert sweeps() >= 1


async def test_idle_use_ensures_the_server_and_yields_its_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_launch(command: object, *, name: str) -> DetachedRun:
        return DetachedRun(name=name, pid=4321, log_path=Path("/tmp/x.log"))

    monkeypatch.setattr(ManagedServer, "health", health_sequence(False, True))
    mock_models(monkeypatch, "mlx-community/dots.ocr-4bit")
    monkeypatch.setattr(serve.detach, "launch", fake_launch)

    resource = ManagedServer("mlx-vlm").idle(ttl_s=300.0)
    async with resource.use() as handle:
        assert handle.base_url == "http://127.0.0.1:8401/v1"
        assert handle.pid == 4321
        assert resource.loaded is True
    assert resource.inflight == 0


async def test_idle_sweep_after_ttl_stops_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[int] = []

    async def fake_launch(command: object, *, name: str) -> DetachedRun:
        return DetachedRun(name=name, pid=4242, log_path=Path("/tmp/x.log"))

    monkeypatch.setattr(ManagedServer, "health", health_sequence(False, True))
    mock_models(monkeypatch, "mlx-community/dots.ocr-4bit")
    monkeypatch.setattr(serve.detach, "launch", fake_launch)
    monkeypatch.setattr(serve.launchd, "installed", lambda **_: [])
    monkeypatch.setattr(serve.detach, "running", lambda name: 4242)
    monkeypatch.setattr(serve, "kill_group", killed.append)

    resource = ManagedServer("mlx-vlm").idle(ttl_s=300.0)
    async with resource.use():
        pass
    assert resource.loaded is True

    await resource.sweep(now=resource.last_done + 301.0)

    assert killed == [4242]
    assert resource.loaded is False
