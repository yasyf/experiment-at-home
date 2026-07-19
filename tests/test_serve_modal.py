from __future__ import annotations

import io
import json
import pickletools
import sys
from dataclasses import dataclass, field
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING

import httpx
import pytest

from athome import serve, serve_modal
from athome.config import load
from athome.serve import (
    HealthTimeout,
    ManagedServer,
    ModalServeBackend,
    ModalVllmSettings,
    ServeError,
    command_for,
    configured_model,
    settings_for,
)

if TYPE_CHECKING:
    from collections.abc import Callable

WORKSPACE = "anetaco"
VLLM_VERSION = "0.11.0"
API_KEY = "sk-ccsteer-test-2f9c7a41b0"
ENDPOINT = f"https://{WORKSPACE}--athome-modal-vllm-{serve_modal.ENDPOINT_FUNCTION}.modal.run/v1"
WEB_URL = ENDPOINT.removesuffix("/v1")


def configure_modal_vllm(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    env = {
        "ATHOME_SERVE_MODAL_VLLM_WORKSPACE": WORKSPACE,
        "ATHOME_SERVE_MODAL_VLLM_VLLM_VERSION": VLLM_VERSION,
        "ATHOME_SERVE_MODAL_VLLM_API_KEY": API_KEY,
    } | overrides
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    load.cache_clear()


def mock_models(monkeypatch: pytest.MonkeyPatch, *ids: str) -> None:
    body = {"object": "list", "data": [{"id": model_id, "object": "model"} for model_id in ids]}
    monkeypatch.setattr(
        serve,
        "health_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))),
    )


def forbid_http(monkeypatch: pytest.MonkeyPatch, message: str) -> None:
    # Loopback probes (the always-configured stt recipe) see a down server; only the
    # hosted web endpoint is forbidden outright.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host in ("127.0.0.1", "localhost"):
            raise httpx.ConnectError("local server down", request=request)
        raise AssertionError(message)

    monkeypatch.setattr(serve, "health_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def health_sequence(*results: bool) -> Callable[[ManagedServer], object]:
    calls = iter(results)

    async def fake(self: ManagedServer) -> bool:
        return next(calls)

    return fake


class ModalNotFound(Exception):
    """Stand-in for ``modal.exception.NotFoundError`` in the mocked boundary."""


@dataclass(slots=True)
class ModalRecorder:
    """Records every builder-chain and lifecycle call the fake ``modal`` module receives."""

    web_url: str = WEB_URL
    image: list[tuple[str, tuple[object, ...], dict[str, object]]] = field(default_factory=list)
    app_name: str | None = None
    app_tags: dict[str, str] = field(default_factory=dict)
    function_options: dict[str, object] = field(default_factory=dict)
    volumes: list[tuple[str, bool]] = field(default_factory=list)
    secrets: list[dict[str, str]] = field(default_factory=list)
    web_server: dict[str, object] = field(default_factory=dict)
    concurrent: dict[str, object] = field(default_factory=dict)
    deployed: int = 0
    deployment: dict[str, object] | None = None
    deploy_error: BaseException | None = None

    @property
    def serve_argv(self) -> list[str]:
        env = next(args[0] for name, args, _ in self.image if name == "env")
        return json.loads(env[serve_modal.ARGV_ENV])

    def mark_deployed(self, *, tags: dict[str, str]) -> None:
        self.deployment = {"tags": dict(tags), "web_url": self.web_url}


def install_fake_modal(monkeypatch: pytest.MonkeyPatch, *, web_url: str = WEB_URL) -> ModalRecorder:
    rec = ModalRecorder(web_url=web_url)

    class Chain:
        def __getattr__(self, name: str) -> Callable[..., Chain]:
            def step(*args: object, **kwargs: object) -> Chain:
                rec.image.append((name, args, kwargs))
                return self

            return step

    class Image:
        @staticmethod
        def from_registry(ref: str, *, add_python: str) -> Chain:
            rec.image.append(("from_registry", (ref,), {"add_python": add_python}))
            return Chain()

    class Volume:
        @staticmethod
        def from_name(name: str, *, create_if_missing: bool = False) -> object:
            rec.volumes.append((name, create_if_missing))
            return SimpleNamespace(name=name)

    class Secret:
        @staticmethod
        def from_dict(values: dict[str, str]) -> object:
            rec.secrets.append(values)
            return values

    def web_server(*, port: int, startup_timeout: int) -> Callable[[Callable[..., object]], Callable[..., object]]:
        def decorate(fn: Callable[..., object]) -> Callable[..., object]:
            rec.web_server = {"port": port, "startup_timeout": startup_timeout, "fn": fn}
            return fn

        return decorate

    def concurrent(*, max_inputs: int) -> Callable[[Callable[..., object]], Callable[..., object]]:
        def decorate(fn: Callable[..., object]) -> Callable[..., object]:
            rec.concurrent = {"max_inputs": max_inputs, "fn": fn}
            return fn

        return decorate

    @dataclass(slots=True)
    class FakeFunction:
        endpoint_url: str

        def hydrate(self) -> FakeFunction:
            return self

        def get_web_url(self) -> str:
            return self.endpoint_url

        def get_current_stats(self) -> object:
            return SimpleNamespace(backlog=0, num_total_runners=0, num_running_inputs=0, input_headroom=0)

    class Function:
        @staticmethod
        def from_name(app_name: str, name: str, **kwargs: object) -> FakeFunction:
            if rec.deployment is None:
                raise ModalNotFound(f"{app_name}/{name} is not deployed")
            return FakeFunction(rec.deployment["web_url"])

    class App:
        def __init__(self, name: str, *, tags: dict[str, str] | None = None) -> None:
            rec.app_name = name
            rec.app_tags = dict(tags or {})

        def function(self, **options: object) -> Callable[[Callable[..., object]], Callable[..., object]]:
            def decorate(fn: Callable[..., object]) -> Callable[..., object]:
                rec.function_options = options
                return fn

            return decorate

        def deploy(self) -> None:
            if rec.deploy_error is not None:
                raise rec.deploy_error
            rec.deployed += 1
            rec.deployment = {"tags": dict(rec.app_tags), "web_url": rec.web_url}

        @staticmethod
        def lookup(name: str, **kwargs: object) -> object:
            if rec.deployment is None:
                raise ModalNotFound(f"{name} is not deployed")
            return SimpleNamespace(get_tags=lambda: rec.deployment["tags"])

    exception = ModuleType("modal.exception")
    exception.NotFoundError = ModalNotFound

    module = ModuleType("modal")
    module.__spec__ = ModuleSpec("modal", None)
    module.Image = Image
    module.Volume = Volume
    module.Secret = Secret
    module.web_server = web_server
    module.concurrent = concurrent
    module.App = App
    module.Function = Function
    module.exception = exception
    monkeypatch.setitem(sys.modules, "modal", module)
    monkeypatch.setitem(sys.modules, "modal.exception", exception)
    return rec


def test_settings_for_loads_the_modal_vllm_section(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch)
    settings = settings_for("modal-vllm")
    assert isinstance(settings, ModalVllmSettings)
    assert (settings.workspace, settings.vllm_version, settings.api_key) == (WORKSPACE, VLLM_VERSION, API_KEY)
    assert settings.app_name == "athome-modal-vllm"
    assert settings.hf_revision == "b968826d9c46dd6066d109eabc6255188de91218"


def test_command_for_refuses_the_hosted_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch)
    with pytest.raises(ServeError, match="hosted Modal recipe"):
        command_for("modal-vllm")


def test_configured_model_is_the_served_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch)
    assert configured_model("modal-vllm") == "qwen3-8b"


def test_base_url_is_the_workspace_scoped_modal_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch)
    server = ManagedServer("modal-vllm")
    assert server.base_url == ENDPOINT
    assert server.served_port is None
    assert server.handle().port is None
    assert server.handle().base_url == ENDPOINT


def test_the_handle_carries_the_configured_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch)
    assert ManagedServer("modal-vllm").handle().api_key == API_KEY


async def test_client_targets_the_hosted_endpoint_with_the_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")
    configure_modal_vllm(monkeypatch)
    client = ManagedServer("modal-vllm").client()
    assert str(client.base_url).rstrip("/") == ENDPOINT
    assert client.api_key == API_KEY
    await client.close()


async def test_health_sends_the_bearer_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch)
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        assert str(request.url) == f"{ENDPOINT}/models"
        return httpx.Response(200)

    monkeypatch.setattr(serve, "health_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert await ManagedServer("modal-vllm").health() is True
    assert seen == [f"Bearer {API_KEY}"]


async def test_ensure_deploys_the_app_then_health_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch)
    rec = install_fake_modal(monkeypatch)
    monkeypatch.setattr(ManagedServer, "health", health_sequence(False, True))
    mock_models(monkeypatch, "qwen3-8b", "watcher")

    handle = await ManagedServer("modal-vllm").ensure()

    assert rec.deployed == 1
    assert rec.app_name == "athome-modal-vllm"
    assert handle.base_url == ENDPOINT
    assert handle.api_key == API_KEY
    assert handle.port is None
    assert handle.pid is None


async def test_ensure_adopts_a_warm_endpoint_without_deploying(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch)
    rec = install_fake_modal(monkeypatch)
    monkeypatch.setattr(ManagedServer, "health", health_sequence(True))
    mock_models(monkeypatch, "qwen3-8b", "watcher")

    handle = await ManagedServer("modal-vllm").ensure()

    assert rec.deployed == 0
    assert handle.base_url == ENDPOINT
    assert handle.pid is None


async def test_ensure_skips_the_redeploy_when_the_deployed_pins_match(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch)
    rec = install_fake_modal(monkeypatch)
    rec.mark_deployed(tags={serve_modal.FINGERPRINT_TAG: serve_modal.fingerprint(load(ModalVllmSettings))})
    monkeypatch.setattr(ManagedServer, "health", health_sequence(False, True))
    mock_models(monkeypatch, "qwen3-8b", "watcher")

    await ManagedServer("modal-vllm").ensure()

    assert rec.deployed == 0


async def test_ensure_redeploys_when_the_deployed_pins_drifted(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch, ATHOME_SERVE_MODAL_VLLM_VLLM_VERSION="0.10.0")
    stale = serve_modal.fingerprint(load(ModalVllmSettings))
    configure_modal_vllm(monkeypatch)
    rec = install_fake_modal(monkeypatch)
    rec.mark_deployed(tags={serve_modal.FINGERPRINT_TAG: stale})
    monkeypatch.setattr(ManagedServer, "health", health_sequence(False, True))
    mock_models(monkeypatch, "qwen3-8b", "watcher")

    await ManagedServer("modal-vllm").ensure()

    assert rec.deployed == 1


async def test_ensure_rejects_a_hosted_endpoint_serving_another_model(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch)
    install_fake_modal(monkeypatch)
    monkeypatch.setattr(ManagedServer, "health", health_sequence(True))
    mock_models(monkeypatch, "some-other-model")

    with pytest.raises(ServeError, match="qwen3-8b"):
        await ManagedServer("modal-vllm").ensure()


async def test_ensure_rejects_a_hosted_endpoint_missing_the_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch)
    install_fake_modal(monkeypatch)
    monkeypatch.setattr(ManagedServer, "health", health_sequence(True))
    mock_models(monkeypatch, "qwen3-8b")

    with pytest.raises(ServeError, match="watcher"):
        await ManagedServer("modal-vllm").ensure()


async def test_ensure_raises_when_the_deployed_url_differs_from_the_fabricated_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_modal_vllm(monkeypatch)
    install_fake_modal(monkeypatch, web_url=f"https://{WORKSPACE}--athome-modal-vllm-elsewhere.modal.run")
    monkeypatch.setattr(ManagedServer, "health", health_sequence(False))

    with pytest.raises(ServeError, match="deployed at"):
        await ManagedServer("modal-vllm").ensure()


async def test_ensure_propagates_a_deploy_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch)
    rec = install_fake_modal(monkeypatch)
    rec.deploy_error = RuntimeError("modal auth failed")
    monkeypatch.setattr(ManagedServer, "health", health_sequence(False))

    with pytest.raises(RuntimeError, match="modal auth failed"):
        await ManagedServer("modal-vllm").ensure()


async def test_ensure_times_out_on_a_cold_hosted_endpoint_without_local_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_modal_vllm(monkeypatch, ATHOME_SERVE_MODAL_VLLM_STARTUP_TIMEOUT="0")
    rec = install_fake_modal(monkeypatch)
    rec.mark_deployed(tags={serve_modal.FINGERPRINT_TAG: serve_modal.fingerprint(load(ModalVllmSettings))})

    def boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("a hosted timeout must not touch local teardown primitives")

    monkeypatch.setattr(serve.launchd, "installed", boom)
    monkeypatch.setattr(serve.detach, "running", boom)
    monkeypatch.setattr(serve, "kill_group", boom)
    monkeypatch.setattr(ManagedServer, "health", health_sequence(False, False))

    with pytest.raises(HealthTimeout) as excinfo:
        await ManagedServer("modal-vllm").ensure()
    assert "after 0s" in str(excinfo.value)
    assert rec.deployed == 0


def test_ready_timeout_is_the_configured_startup_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch, ATHOME_SERVE_MODAL_VLLM_STARTUP_TIMEOUT="450")
    settings = load(ModalVllmSettings)
    assert ModalServeBackend().ready_timeout_s(ManagedServer("modal-vllm")) == float(settings.startup_timeout) == 450.0


async def test_a_port_override_is_refused_for_the_hosted_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch)
    with pytest.raises(ServeError, match="no port override"):
        ManagedServer("modal-vllm", port=8410).served_port


async def test_a_model_override_is_refused_before_any_deploy(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch)
    rec = install_fake_modal(monkeypatch)
    with pytest.raises(ServeError, match="no model override"):
        await ManagedServer("modal-vllm", model="x").ensure()
    assert rec.deployed == 0


async def test_persistent_is_refused_for_the_hosted_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch)
    rec = install_fake_modal(monkeypatch)
    monkeypatch.setattr(ManagedServer, "health", health_sequence(False))
    with pytest.raises(ServeError, match="no launchd agent"):
        await ManagedServer("modal-vllm").ensure(persistent=True)
    assert rec.deployed == 0


async def test_stop_is_a_noop_for_a_hosted_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch)

    def boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("a hosted recipe owns no local process to tear down")

    monkeypatch.setattr(serve.launchd, "installed", boom)
    monkeypatch.setattr(serve.detach, "running", boom)
    monkeypatch.setattr(serve, "kill_group", boom)

    await ManagedServer("modal-vllm").stop()


async def test_probe_all_observes_the_hosted_recipe_without_an_http_request(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch)
    rec = install_fake_modal(monkeypatch)
    rec.mark_deployed(tags={serve_modal.FINGERPRINT_TAG: serve_modal.fingerprint(load(ModalVllmSettings))})
    forbid_http(monkeypatch, "probe must not touch the web endpoint")

    assert ("modal-vllm", True) in await serve.probe_all()


async def test_probe_reports_false_for_an_undeployed_hosted_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch)
    install_fake_modal(monkeypatch)
    forbid_http(monkeypatch, "probe must not touch the web endpoint")

    assert await ManagedServer("modal-vllm").probe() is False


def test_vllm_command_pins_the_base_revision_and_unfused_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch)
    argv = serve_modal.vllm_command(load(ModalVllmSettings))
    assert argv[:3] == ["vllm", "serve", "Qwen/Qwen3-8B"]
    revision = "b968826d9c46dd6066d109eabc6255188de91218"
    assert argv[argv.index("--revision") + 1] == revision
    assert argv[argv.index("--tokenizer-revision") + 1] == revision
    assert argv[argv.index("--served-model-name") + 1] == "qwen3-8b"
    assert "--enable-lora" in argv
    assert argv[argv.index("--lora-modules") + 1] == "watcher=/adapter"
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert argv[argv.index("--port") + 1] == "8000"
    assert "--api-key" not in argv


def test_vllm_command_pins_the_cold_start_and_scoring_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch)
    argv = serve_modal.vllm_command(load(ModalVllmSettings))
    assert "--enforce-eager" in argv
    assert argv[argv.index("--max-model-len") + 1] == "4096"
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.92"
    assert argv[argv.index("--max-logprobs") + 1] == "40"
    assert argv[argv.index("--max-lora-rank") + 1] == "32"


def test_build_app_pins_image_gpu_volumes_secret_and_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch)
    rec = install_fake_modal(monkeypatch)

    serve_modal.build_app(load(ModalVllmSettings))

    assert rec.app_name == "athome-modal-vllm"
    assert rec.app_tags == {serve_modal.FINGERPRINT_TAG: serve_modal.fingerprint(load(ModalVllmSettings))}
    assert rec.image[0] == ("from_registry", ("nvidia/cuda:12.8.1-devel-ubuntu22.04",), {"add_python": "3.12"})
    pip = next(args for name, args, _ in rec.image if name == "pip_install")
    assert f"vllm=={VLLM_VERSION}" in pip
    assert rec.serve_argv[:3] == ["vllm", "serve", "Qwen/Qwen3-8B"]
    assert rec.function_options["gpu"] == "A10G"
    assert rec.function_options["scaledown_window"] == 300
    assert rec.function_options["min_containers"] == 0
    assert rec.function_options["timeout"] == 3600
    assert rec.function_options["serialized"] is True
    assert rec.volumes == [("cc-steer-hf-cache", True), ("cc-steer-watcher-adapter", True)]
    assert rec.secrets == [{"VLLM_API_KEY": API_KEY}]
    assert rec.web_server["port"] == 8000
    assert rec.web_server["startup_timeout"] == 900
    assert callable(rec.web_server["fn"])
    assert rec.web_server["fn"].__name__ == serve_modal.ENDPOINT_FUNCTION
    assert rec.concurrent["max_inputs"] == 32


def test_the_web_server_entrypoint_is_serialized_by_value(monkeypatch: pytest.MonkeyPatch) -> None:
    cloudpickle = pytest.importorskip("cloudpickle")
    configure_modal_vllm(monkeypatch)
    rec = install_fake_modal(monkeypatch)

    serve_modal.build_app(load(ModalVllmSettings))
    payload = cloudpickle.dumps(rec.web_server["fn"])

    # A by-reference pickle is ~43 bytes and STACK_GLOBALs athome.serve_modal (unimportable cold).
    assert len(payload) > 200
    modules: list[str | None] = []
    recent: list[str] = []
    for op, arg, _ in pickletools.genops(io.BytesIO(payload)):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE") and isinstance(arg, str):
            recent.append(arg)
        elif op.name == "STACK_GLOBAL":
            modules.append(recent[-2] if len(recent) >= 2 else None)
    assert "athome.serve_modal" not in modules


async def test_deploy_builds_then_ships_the_app(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch)
    rec = install_fake_modal(monkeypatch)

    await serve_modal.deploy(load(ModalVllmSettings))

    assert rec.deployed == 1
    assert rec.app_name == "athome-modal-vllm"


@pytest.mark.live
async def test_modal_vllm_live_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_modal_vllm(monkeypatch)
    await serve.up("modal-vllm")
    assert await ManagedServer("modal-vllm").health()
