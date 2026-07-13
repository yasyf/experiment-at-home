from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import httpx
import pytest

from athome import serve
from athome.config import load
from athome.detach import DetachedRun
from athome.launchd import KeepAlive, LaunchdError
from athome.serve import HealthTimeout, ManagedServer, ServeError, ServerHandle, command_for, down, probe_all, up

if TYPE_CHECKING:
    from collections.abc import Callable

MLX_VLM_VERSION = "0.3.4"


@pytest.fixture(autouse=True)
def configure_mlx_vlm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATHOME_SERVE_MLX_VLM_VERSION", MLX_VLM_VERSION)
    load.cache_clear()


def configure_rapid_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATHOME_SERVE_RAPID_MLX_VERSION", "0.10.9")
    monkeypatch.setenv("ATHOME_SERVE_RAPID_MLX_MODEL", "mlx-community/Qwen3-4bit")
    load.cache_clear()


def configure_llama_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATHOME_SERVE_LLAMA_SERVER_COMMAND", "llama-server -m model.gguf --port 8402")
    load.cache_clear()


def mock_health(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    monkeypatch.setattr(
        serve,
        "health_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(status))),
    )


def mock_models(monkeypatch: pytest.MonkeyPatch, *ids: str) -> None:
    body = {"object": "list", "data": [{"id": model_id, "object": "model"} for model_id in ids]}
    monkeypatch.setattr(
        serve,
        "health_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))),
    )


def health_sequence(*results: bool) -> Callable[[ManagedServer], object]:
    calls = iter(results)

    async def fake(self: ManagedServer) -> bool:
        return next(calls)

    return fake


def test_command_rapid_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_rapid_mlx(monkeypatch)
    assert command_for("rapid-mlx") == (
        "uvx",
        "--from",
        "rapid-mlx==0.10.9",
        "rapid-mlx",
        "serve",
        "mlx-community/Qwen3-4bit",
        "--port",
        "8400",
    )


def test_command_mlx_vlm() -> None:
    command = command_for("mlx-vlm")
    assert command == (
        "uvx",
        "--from",
        f"mlx-vlm=={MLX_VLM_VERSION}",
        "mlx_vlm.server",
        "--model",
        "mlx-community/dots.ocr-4bit",
        "--port",
        "8401",
    )
    assert f"mlx-vlm=={MLX_VLM_VERSION}" in command


def test_command_llama_server(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_llama_server(monkeypatch)
    assert command_for("llama-server") == ("llama-server", "-m", "model.gguf", "--port", "8402")


def test_handle_base_url() -> None:
    assert ManagedServer("mlx-vlm").handle().base_url == "http://127.0.0.1:8401/v1"


def test_server_handle_is_frozen() -> None:
    handle = ServerHandle(recipe="mlx-vlm", port=8401, pid=None, base_url="http://127.0.0.1:8401/v1")
    with pytest.raises(AttributeError):
        handle.port = 9999  # type: ignore[misc]


async def test_health_true_on_2xx(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_health(monkeypatch, 200)
    assert await ManagedServer("mlx-vlm").health() is True


async def test_health_false_on_non_2xx(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_health(monkeypatch, 503)
    assert await ManagedServer("mlx-vlm").health() is False


async def test_health_false_on_connect_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(serve, "health_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(boom)))
    assert await ManagedServer("mlx-vlm").health() is False


async def test_ensure_idempotent_when_already_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[object] = []

    async def fail_launch(command: object, *, name: str) -> object:
        launched.append(command)
        raise AssertionError("should not spawn a healthy server")

    monkeypatch.setattr(ManagedServer, "health", health_sequence(True))
    mock_models(monkeypatch, "mlx-community/dots.ocr-4bit")
    monkeypatch.setattr(serve.detach, "launch", fail_launch)
    handle = await ManagedServer("mlx-vlm").ensure()
    assert handle.base_url == "http://127.0.0.1:8401/v1"
    assert launched == []


async def test_ensure_rejects_healthy_server_with_wrong_model(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[object] = []

    async def fail_launch(command: object, *, name: str) -> object:
        launched.append(command)
        raise AssertionError("must not adopt or spawn over a mismatched server")

    monkeypatch.setattr(ManagedServer, "health", health_sequence(True))
    mock_models(monkeypatch, "mlx-community/some-other-model")
    monkeypatch.setattr(serve.detach, "launch", fail_launch)
    with pytest.raises(ServeError, match="dots.ocr-4bit"):
        await ManagedServer("mlx-vlm").ensure()
    assert launched == []


async def test_ensure_rejects_launched_server_with_wrong_model(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[object] = []

    async def fake_launch(command: object, *, name: str) -> DetachedRun:
        launched.append(command)
        return DetachedRun(name=name, pid=555, log_path=Path("/tmp/x.log"))

    monkeypatch.setattr(ManagedServer, "health", health_sequence(False, True))
    mock_models(monkeypatch, "mlx-community/some-other-model")
    monkeypatch.setattr(serve.detach, "launch", fake_launch)
    monkeypatch.setattr(serve.launchd, "installed", lambda **_: [])
    monkeypatch.setattr(serve.detach, "running", lambda name: 555)
    monkeypatch.setattr(serve, "kill_group", lambda pid: None)
    with pytest.raises(ServeError, match="dots.ocr-4bit"):
        await ManagedServer("mlx-vlm").ensure()
    assert launched != []


async def test_ensure_spawns_detached_with_command_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_rapid_mlx(monkeypatch)
    captured: dict[str, object] = {}

    async def fake_launch(command: object, *, name: str) -> DetachedRun:
        captured["command"] = tuple(command)  # type: ignore[arg-type]
        captured["name"] = name
        return DetachedRun(name=name, pid=4321, log_path=Path("/tmp/x.log"))

    monkeypatch.setattr(ManagedServer, "health", health_sequence(False, True))
    mock_models(monkeypatch, "mlx-community/Qwen3-4bit")
    monkeypatch.setattr(serve.detach, "launch", fake_launch)
    handle = await ManagedServer("rapid-mlx").ensure()
    assert captured["command"] == command_for("rapid-mlx")
    assert captured["name"] == "serve-rapid-mlx"
    assert handle.pid == 4321
    assert handle.port == 8400


async def test_ensure_persistent_installs_keepalive_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_install(spec: object) -> Path:
        captured["spec"] = spec
        return Path("/tmp/agent.plist")

    monkeypatch.setattr(ManagedServer, "health", health_sequence(False, True))
    mock_models(monkeypatch, "mlx-community/dots.ocr-4bit")
    monkeypatch.setattr(serve.launchd, "install", fake_install)
    handle = await ManagedServer("mlx-vlm").ensure(persistent=True)
    spec = captured["spec"]
    assert spec.label == "com.athome.serve-mlx-vlm"
    assert spec.command == command_for("mlx-vlm")
    assert isinstance(spec.schedule, KeepAlive)
    assert spec.log_name == "serve-mlx-vlm"
    assert handle.pid is None


async def test_ensure_times_out_when_never_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_launch(command: object, *, name: str) -> DetachedRun:
        return DetachedRun(name=name, pid=1, log_path=Path("/tmp/x.log"))

    monkeypatch.setattr(ManagedServer, "health", health_sequence(False, False))
    monkeypatch.setattr(serve.detach, "launch", fake_launch)
    monkeypatch.setattr(serve.launchd, "installed", lambda **_: [])
    monkeypatch.setattr(serve.detach, "running", lambda name: None)
    monkeypatch.setattr(serve, "READY_TIMEOUT_S", 0.0)
    with pytest.raises(HealthTimeout, match="mlx-vlm"):
        await ManagedServer("mlx-vlm").ensure()


async def test_ensure_tears_down_detached_run_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[int] = []

    async def fake_launch(command: object, *, name: str) -> DetachedRun:
        return DetachedRun(name=name, pid=777, log_path=Path("/tmp/x.log"))

    monkeypatch.setattr(ManagedServer, "health", health_sequence(False, False))
    monkeypatch.setattr(serve.detach, "launch", fake_launch)
    monkeypatch.setattr(serve.launchd, "installed", lambda **_: [])
    monkeypatch.setattr(serve.detach, "running", lambda name: 777)
    monkeypatch.setattr(serve, "kill_group", killed.append)
    monkeypatch.setattr(serve, "READY_TIMEOUT_S", 0.0)
    with pytest.raises(HealthTimeout):
        await ManagedServer("mlx-vlm").ensure()
    assert killed == [777]


async def test_ensure_cancellation_tears_down_launched_server(monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[int] = []
    probes = {"n": 0}

    async def fake_launch(command: object, *, name: str) -> DetachedRun:
        return DetachedRun(name=name, pid=999, log_path=Path("/tmp/x.log"))

    async def probe(self: ManagedServer) -> bool:
        probes["n"] += 1
        if probes["n"] == 1:
            return False
        await anyio.sleep_forever()
        return False

    monkeypatch.setattr(ManagedServer, "health", probe)
    monkeypatch.setattr(serve.detach, "launch", fake_launch)
    monkeypatch.setattr(serve.launchd, "installed", lambda **_: [])
    monkeypatch.setattr(serve.detach, "running", lambda name: 999)
    monkeypatch.setattr(serve, "kill_group", killed.append)
    with anyio.move_on_after(0.05):
        await ManagedServer("mlx-vlm").ensure()
    assert killed == [999]


async def test_ensure_persistent_uninstalls_agent_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    removed: list[str] = []

    async def fake_install(spec: object) -> Path:
        return Path("/tmp/agent.plist")

    async def fake_uninstall(label: str) -> None:
        removed.append(label)

    monkeypatch.setattr(ManagedServer, "health", health_sequence(False, False))
    monkeypatch.setattr(serve.launchd, "install", fake_install)
    monkeypatch.setattr(serve.launchd, "installed", lambda **_: ["com.athome.serve-mlx-vlm"])
    monkeypatch.setattr(serve.launchd, "uninstall", fake_uninstall)
    monkeypatch.setattr(serve.detach, "running", lambda name: None)
    monkeypatch.setattr(serve, "READY_TIMEOUT_S", 0.0)
    with pytest.raises(HealthTimeout):
        await ManagedServer("mlx-vlm").ensure(persistent=True)
    assert removed == ["com.athome.serve-mlx-vlm"]


async def test_stop_covers_both_detached_and_launchd(monkeypatch: pytest.MonkeyPatch) -> None:
    removed: list[str] = []
    killed: list[int] = []

    async def fake_uninstall(label: str) -> None:
        removed.append(label)

    monkeypatch.setattr(serve.launchd, "installed", lambda **_: ["com.athome.serve-mlx-vlm"])
    monkeypatch.setattr(serve.launchd, "uninstall", fake_uninstall)
    monkeypatch.setattr(serve.detach, "running", lambda name: 4242)
    monkeypatch.setattr(serve, "kill_group", killed.append)
    await ManagedServer("mlx-vlm").stop()
    assert removed == ["com.athome.serve-mlx-vlm"]
    assert killed == [4242]


async def test_stop_kills_detached_when_launchd_uninstall_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[int] = []

    async def boom_uninstall(label: str) -> None:
        raise LaunchdError(f"{label} still loaded after launchctl bootout")

    monkeypatch.setattr(serve.launchd, "installed", lambda **_: ["com.athome.serve-mlx-vlm"])
    monkeypatch.setattr(serve.launchd, "uninstall", boom_uninstall)
    monkeypatch.setattr(serve.detach, "running", lambda name: 4242)
    monkeypatch.setattr(serve, "kill_group", killed.append)
    await ManagedServer("mlx-vlm").stop()
    assert killed == [4242]


async def test_wait_healthy_rejects_probe_past_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ManagedServer, "health", health_sequence(True))
    monkeypatch.setattr(serve, "READY_TIMEOUT_S", 0.0)
    with pytest.raises(HealthTimeout, match="mlx-vlm"):
        await ManagedServer("mlx-vlm").wait_healthy()


async def test_stop_uninstalls_launchd_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    removed: list[str] = []

    async def fake_uninstall(label: str) -> None:
        removed.append(label)

    monkeypatch.setattr(serve.launchd, "installed", lambda **_: ["com.athome.serve-mlx-vlm"])
    monkeypatch.setattr(serve.launchd, "uninstall", fake_uninstall)
    monkeypatch.setattr(serve.detach, "running", lambda name: None)
    await down("mlx-vlm")
    assert removed == ["com.athome.serve-mlx-vlm"]


async def test_stop_kills_detached_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[int] = []
    monkeypatch.setattr(serve.launchd, "installed", lambda **_: [])
    monkeypatch.setattr(serve.detach, "running", lambda name: 4242)
    monkeypatch.setattr(serve, "kill_group", killed.append)
    await down("mlx-vlm")
    assert killed == [4242]


async def test_probe_all_skips_unconfigured_recipes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ManagedServer, "health", health_sequence(True))
    assert await probe_all() == [("mlx-vlm", True)]


async def test_probe_all_includes_configured_recipes(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_rapid_mlx(monkeypatch)

    async def always_healthy(self: ManagedServer) -> bool:
        return True

    monkeypatch.setattr(ManagedServer, "health", always_healthy)
    result = dict(await probe_all())
    assert result == {"rapid-mlx": True, "mlx-vlm": True}


async def test_client_targets_recipe_endpoint() -> None:
    pytest.importorskip("openai")
    client = ManagedServer("mlx-vlm").client()
    assert "127.0.0.1:8401/v1" in str(client.base_url)
    assert client.api_key == "local"
    await client.close()


@pytest.mark.live
async def test_rapid_mlx_live_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_rapid_mlx(monkeypatch)
    await up("rapid-mlx")
    assert await ManagedServer("rapid-mlx").health()
    await down("rapid-mlx")
