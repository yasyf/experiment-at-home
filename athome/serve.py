from __future__ import annotations

import hashlib
import os
import shlex
import signal
import sys
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, runtime_checkable

import anyio
import click
import httpx
from pydantic import ValidationError

from athome import detach, launchd, llmcache
from athome.cli import coro, emit, json_option
from athome.config import SectionSettings, load
from athome.errors import AthomeError
from athome.idle import IdleResource
from athome.launchd import AgentSpec, KeepAlive, LaunchdError
from athome.stt.server import SttServeSettings

if TYPE_CHECKING:
    from openai import AsyncOpenAI

type Recipe = Literal["rapid-mlx", "mlx-vlm", "llama-server", "modal-vllm", "stt"]

RECIPES: tuple[Recipe, ...] = ("rapid-mlx", "mlx-vlm", "llama-server", "modal-vllm", "stt")
DEFAULT_RECIPE: Recipe = "rapid-mlx"
HEALTH_TIMEOUT_S = 1.0
MODAL_PROBE_TIMEOUT_S = 10.0
READY_TIMEOUT_S = 120.0
READY_POLL_S = 1.0
IDENTITY_CHARS = 8
RECIPE_CHOICE = click.Choice(list(RECIPES))


class ServeError(AthomeError):
    """Raised when a managed server cannot be started, stopped, or reached."""


class HealthTimeout(ServeError):
    """Raised when a spawned server does not report healthy within the ready timeout."""


class RapidMlxSettings(SectionSettings):
    """The ``[serve.rapid-mlx]`` section: the daily-pinned rapid-mlx version, model, and port.

    The default text lane. On ordinary tool calling it and :class:`LlamaServerSettings` are
    equivalent: serving the same Qwen3.6-35B-A3B to both, 42 calls each, every call
    dispatched and every argument came back valid and schema-conformant — flat, enum,
    nested, arrays of objects, multi-tool selection, and numeric-verbatim alike. Pick
    between them on the two edges below, not on general quality.

    rapid-mlx serves MLX-native weights only, and honors ``tool_choice="required"``.

    Warning:
        rapid-mlx extracts tool calls with a delimiter-based parser (it advertises
        ``qwen3_coder_xml``), and a string argument containing that delimiter is truncated
        at it: send ``</function>`` inside a payload and the argument arrives silently
        short, as well-formed JSON no downstream check can catch. Serve
        :class:`LlamaServerSettings` when arguments can carry tool-call syntax — quoted
        model output, agent transcripts, scraped markup.
    """

    section: ClassVar[tuple[str, ...]] = ("serve", "rapid-mlx")
    version: str
    model: str
    port: int = 8400


class MlxVlmSettings(SectionSettings):
    """The ``[serve.mlx-vlm]`` section: the daily-pinned mlx-vlm version, vision model, and port."""

    section: ClassVar[tuple[str, ...]] = ("serve", "mlx-vlm")
    version: str
    model: str = "mlx-community/dots.ocr-4bit"
    port: int = 8401


class LlamaServerSettings(SectionSettings):
    """The ``[serve.llama-server]`` section: a full llama-server command string and its port.

    The opt-in GGUF lane, equivalent to the default :class:`RapidMlxSettings` on ordinary
    tool calling. Its one advantage is string fidelity: it returns arguments containing
    tool-call delimiter syntax verbatim, where rapid-mlx's parser truncates them. That is
    the reason to select it.

    Takes a full command string rather than a model and port, because a llama-server
    invocation carries flags no recipe can generalize.

    Warning:
        llama-server ignores ``tool_choice="required"`` — it applies no grammar constraint,
        so instead of forcing a call it can answer in prose and run to ``max_tokens``,
        returning no tool call at all. Code that depends on being handed one must stay on
        rapid-mlx or treat a missing call as a real outcome.
    """

    section: ClassVar[tuple[str, ...]] = ("serve", "llama-server")
    command: str
    port: int = 8402


class ModalVllmSettings(SectionSettings):
    """The ``[serve.modal-vllm]`` section: a scale-to-zero hosted vLLM endpoint on Modal.

    The hosted text lane. Every field the throwaway spike hard-coded lands here as a
    configurable pin — the vLLM version, base model and its commit, the LoRA adapter, and
    the GPU class — so the deployed endpoint is byte-reproducible and its served results
    stay comparable with the local MLX baseline.

    Unlike the local recipes it carries no ``port``: a Modal web endpoint answers on a
    workspace-scoped HTTPS URL (``https://{workspace}--{app_name}-serve.modal.run``), not
    a loopback port, and Modal's own ``scaledown_window`` — not launchd — reaps it once idle.

    ``workspace``, ``vllm_version``, and ``api_key`` are required: the first two so the
    endpoint URL and image are reproducible, the last so the public Modal URL is not open.
    A recipe missing any of them reads as unconfigured and is skipped by ``probe_all``.
    """

    section: ClassVar[tuple[str, ...]] = ("serve", "modal-vllm")
    workspace: str
    vllm_version: str
    api_key: str
    app_name: str = "athome-modal-vllm"
    gpu: str = "A10G"
    base_model: str = "Qwen/Qwen3-8B"
    hf_revision: str = "b968826d9c46dd6066d109eabc6255188de91218"
    served_model_name: str = "qwen3-8b"
    adapter_name: str = "watcher"
    adapter_volume: str = "cc-steer-watcher-adapter"
    hf_cache_volume: str = "cc-steer-hf-cache"
    max_lora_rank: int = 32
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.92
    max_logprobs: int = 40
    max_concurrent_inputs: int = 32
    scaledown_window: int = 300
    startup_timeout: int = 900
    request_timeout: int = 3600


type RecipeSettings = RapidMlxSettings | MlxVlmSettings | LlamaServerSettings | ModalVllmSettings | SttServeSettings


@dataclass(frozen=True, slots=True)
class ServerHandle:
    """A running recipe's port (``None`` for a hosted endpoint), pid (``None`` for a launchd
    service or a hosted endpoint), OpenAI base URL, and the bearer credential that reaches it.

    ``api_key`` is ``"local"`` for a loopback recipe and the recipe's configured key for a hosted
    one, so a consumer holding only the handle can authenticate against the endpoint.

    Example:
        >>> handle = await up("mlx-vlm")
        >>> handle.base_url
        'http://127.0.0.1:8401/v1'
    """

    recipe: Recipe
    port: int | None
    pid: int | None
    base_url: str
    api_key: str


def settings_for(recipe: Recipe) -> RecipeSettings:
    match recipe:
        case "rapid-mlx":
            return load(RapidMlxSettings)
        case "mlx-vlm":
            return load(MlxVlmSettings)
        case "llama-server":
            return load(LlamaServerSettings)
        case "modal-vllm":
            return load(ModalVllmSettings)
        case "stt":
            return load(SttServeSettings)


def command_for(recipe: Recipe, *, model: str | None = None, port: int | None = None) -> tuple[str, ...]:
    """The recipe's command vector; ``model`` and ``port`` override its configured pair."""
    match recipe:
        case "rapid-mlx":
            settings = load(RapidMlxSettings)
            return (
                "uvx",
                "--from",
                f"rapid-mlx=={settings.version}",
                "rapid-mlx",
                "serve",
                model or settings.model,
                "--port",
                str(port or settings.port),
            )
        case "mlx-vlm":
            settings = load(MlxVlmSettings)
            return (
                "uvx",
                "--from",
                f"mlx-vlm=={settings.version}",
                "mlx_vlm.server",
                "--model",
                model or settings.model,
                "--port",
                str(port or settings.port),
            )
        case "llama-server":
            if model is not None or port is not None:
                raise ServeError("llama-server is a full command string; it takes no model or port override")
            return tuple(shlex.split(load(LlamaServerSettings).command))
        case "modal-vllm":
            raise ServeError("modal-vllm is a hosted Modal recipe; it has no local command vector")
        case "stt":
            if model is not None or port is not None:
                raise ServeError("stt is config-driven via [serve.stt]; it takes no model or port override")
            return (sys.executable, "-m", "athome", "serve", "stt")


def configured_model(recipe: Recipe) -> str | None:
    match settings_for(recipe):
        case RapidMlxSettings(model=model) | MlxVlmSettings(model=model):
            return model
        case SttServeSettings(variant=variant):
            return variant
        case LlamaServerSettings():
            return None
        case ModalVllmSettings(served_model_name=model):
            return model


def configured(recipe: Recipe) -> bool:
    try:
        settings_for(recipe)
    except ValidationError:
        return False
    return True


def agent_label(recipe: Recipe) -> str:
    return f"com.athome.serve-{recipe}"


def detach_name(recipe: Recipe) -> str:
    return f"serve-{recipe}"


def health_client() -> httpx.AsyncClient:
    return httpx.AsyncClient()


def kill_group(pid: int) -> None:
    os.killpg(os.getpgid(pid), signal.SIGTERM)


@runtime_checkable
class ServeBackend(Protocol):
    """The substrate a recipe runs on, behind the one :class:`ManagedServer` abstraction.

    Health, model verification, readiness polling, and the OpenAI client are substrate-agnostic
    HTTP against ``base_url`` — they live on :class:`ManagedServer`. The rest differs between a
    local subprocess and a hosted endpoint, and it is all a backend answers: where the endpoint
    lives, which api key reaches it, the pid of an already-running instance, how to start and stop
    it, how long a cold start and a single probe may take, and how to observe health without waking
    a scale-to-zero endpoint.
    """

    def base_url(self, server: ManagedServer) -> str:
        """The OpenAI-compatible ``/v1`` endpoint this recipe answers on."""
        ...

    def api_key(self, server: ManagedServer) -> str:
        """The bearer credential that reaches the endpoint."""
        ...

    def adopt(self, server: ManagedServer) -> int | None:
        """The pid of an instance already answering healthy, for the adopted handle."""
        ...

    def ready_timeout_s(self, server: ManagedServer) -> float:
        """The cold-start budget: how long :meth:`ManagedServer.wait_healthy` waits before giving up."""
        ...

    def probe_timeout_s(self, server: ManagedServer) -> float:
        """The per-probe timeout for a single ``GET /v1/models`` health check."""
        ...

    async def start(self, server: ManagedServer, *, persistent: bool) -> int | None:
        """Bring the server up and return its pid, or ``None`` when it owns no local process."""
        ...

    async def stop(self, server: ManagedServer) -> None:
        """Tear the server down."""
        ...

    async def probe(self, server: ManagedServer) -> bool:
        """Report health without waking a scale-to-zero endpoint (a passive, side-effect-free check)."""
        ...


@dataclass(frozen=True, slots=True)
class LocalServeBackend:
    """A recipe served as a local uvx subprocess: a detached run, or a launchd KeepAlive agent."""

    def base_url(self, server: ManagedServer) -> str:
        return f"http://127.0.0.1:{server.served_port}/v1"

    def api_key(self, server: ManagedServer) -> str:
        return "local"

    def adopt(self, server: ManagedServer) -> int | None:
        return detach.running(server.run_name)

    def ready_timeout_s(self, server: ManagedServer) -> float:
        return READY_TIMEOUT_S

    def probe_timeout_s(self, server: ManagedServer) -> float:
        return HEALTH_TIMEOUT_S

    async def start(self, server: ManagedServer, *, persistent: bool) -> int | None:
        command = command_for(server.recipe, model=server.model, port=server.port)
        if persistent:
            await launchd.install(
                AgentSpec(label=server.label, command=command, schedule=KeepAlive(), log_name=server.run_name)
            )
            return None
        return (await detach.launch(command, name=server.run_name)).pid

    async def stop(self, server: ManagedServer) -> None:
        if server.label in launchd.installed():
            try:
                await launchd.uninstall(server.label)
            except LaunchdError:
                pass
        if (pid := detach.running(server.run_name)) is not None:
            try:
                kill_group(pid)
            except ProcessLookupError:
                pass

    async def probe(self, server: ManagedServer) -> bool:
        return await server.health()


@dataclass(frozen=True, slots=True)
class ModalServeBackend:
    """A recipe served as a scale-to-zero Modal vLLM web endpoint.

    ``start`` deploys the athome-owned Modal app only when it is missing or its pins have drifted:
    a warm, matching deployment is woken by :meth:`ManagedServer.wait_healthy`'s first request off a
    cold, scaled-to-zero container, not redeployed. Deploy and wake are distinct — redeploying on
    every cold start would mint a fresh revision, demand deploy credentials for a read-side wake, and
    add control-plane latency to the readiness path. There is no local process to adopt or kill:
    ``stop`` is a no-op, because Modal's ``scaledown_window`` — not athome — reaps an idle endpoint.
    """

    def base_url(self, server: ManagedServer) -> str:
        from athome import serve_modal

        settings = load(ModalVllmSettings)
        return f"https://{settings.workspace}--{settings.app_name}-{serve_modal.ENDPOINT_FUNCTION}.modal.run/v1"

    def api_key(self, server: ManagedServer) -> str:
        return load(ModalVllmSettings).api_key

    def adopt(self, server: ManagedServer) -> int | None:
        return None

    def ready_timeout_s(self, server: ManagedServer) -> float:
        return float(load(ModalVllmSettings).startup_timeout)

    def probe_timeout_s(self, server: ManagedServer) -> float:
        return MODAL_PROBE_TIMEOUT_S

    async def start(self, server: ManagedServer, *, persistent: bool) -> int | None:
        from athome import serve_modal

        if persistent:
            raise ServeError("a hosted recipe has no launchd agent; Modal's scaledown owns its lifecycle")
        settings = load(ModalVllmSettings)
        if await serve_modal.deployed_fingerprint(settings) == serve_modal.fingerprint(settings):
            return None
        await serve_modal.deploy(settings)
        resolved = serve_modal.web_url(await serve_modal.locate(settings))
        expected = self.base_url(server).removesuffix("/v1")
        if resolved is not None and resolved.rstrip("/") != expected:
            raise ServeError(f"modal-vllm deployed at {resolved!r}, not the expected {self.base_url(server)!r}")
        return None

    async def stop(self, server: ManagedServer) -> None:
        return None

    async def probe(self, server: ManagedServer) -> bool:
        from athome import serve_modal

        return await serve_modal.locate(load(ModalVllmSettings)) is not None


def backend_for(recipe: Recipe) -> ServeBackend:
    match recipe:
        case "rapid-mlx" | "mlx-vlm" | "llama-server" | "stt":
            return LocalServeBackend()
        case "modal-vllm":
            return ModalServeBackend()


@dataclass(frozen=True, slots=True)
class ManagedServer:
    """A recipe-configured OpenAI-compatible server with lifecycle + health, over any backend.

    The recipe picks the backend: a local uvx subprocess (``rapid-mlx``, ``mlx-vlm``,
    ``llama-server``) or a scale-to-zero hosted Modal vLLM endpoint (``modal-vllm``). Health,
    model verification, readiness, and the client are identical across both — only where the
    endpoint lives and how it starts and stops differ, which is all a :class:`ServeBackend` answers.

    ``model`` and ``port`` override the recipe's configured pair for one server — serving an
    arbitrary artifact (a freshly fused MLX directory, say) without touching the process-global
    settings. An overridden server is a distinct managed process: it carries its own detach run
    name and launchd label, so it neither adopts nor tears down the config-driven one.

    Example:
        >>> server = ManagedServer("rapid-mlx")
        >>> handle = await server.ensure()
        >>> fused = ManagedServer("rapid-mlx", model="/runs/watcher/fused", port=8410)
    """

    recipe: Recipe
    model: str | None = field(default=None, kw_only=True)
    port: int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.recipe != "modal-vllm":
            return
        if self.port is not None:
            raise ServeError("modal-vllm answers on a workspace-scoped HTTPS URL; it takes no port override")
        if self.model is not None:
            raise ServeError("modal-vllm serves the model pinned by its deploy; it takes no model override")

    @property
    def served_model(self) -> str | None:
        """The model this server serves: the override, else the recipe's configured model."""
        return self.model if self.model is not None else configured_model(self.recipe)

    @property
    def required_models(self) -> set[str]:
        """Every model id the endpoint must report, not just the base.

        An override serves exactly the one model it names. A hosted vLLM recipe serves its base
        model *and* the LoRA adapter it exists to score under a second id — an endpoint serving only
        the base would pass a base-only identity check while missing the adapter. A ``llama-server``
        command advertises nothing checkable, so it requires none.
        """
        if self.model is not None:
            return {self.model}
        match settings_for(self.recipe):
            case ModalVllmSettings(served_model_name=base, adapter_name=adapter):
                return {base, adapter}
            case _:
                return {model} if (model := configured_model(self.recipe)) is not None else set()

    @property
    def served_port(self) -> int | None:
        """The port this server listens on: the override, else the configured port; ``None`` when hosted.

        A hosted recipe answers on a workspace-scoped HTTPS URL, not a loopback port, so it has none.
        """
        if self.port is not None:
            return self.port
        match settings_for(self.recipe):
            case ModalVllmSettings():
                return None
            case RapidMlxSettings(port=port) | MlxVlmSettings(port=port) | LlamaServerSettings(port=port):
                return port

    @property
    def identity(self) -> str:
        """What distinguishes an overridden server: the port it listens on and the model it serves.

        The model belongs in the identity, not just the port: two runs that land on one port
        would otherwise share a run name and a launchd label, and each would tear down the
        other's process while verifying it served the wrong model.
        """
        return f"{self.served_port}-{hashlib.sha256(str(self.served_model).encode()).hexdigest()[:IDENTITY_CHARS]}"

    @property
    def run_name(self) -> str:
        """The detach run name owning this server's process."""
        return detach_name(self.recipe) if self.model is None else f"{detach_name(self.recipe)}-{self.identity}"

    @property
    def label(self) -> str:
        """The launchd label owning this server's agent."""
        return agent_label(self.recipe) if self.model is None else f"{agent_label(self.recipe)}-{self.identity}"

    @property
    def backend(self) -> ServeBackend:
        """The substrate this recipe runs on: a local subprocess, or a hosted Modal endpoint."""
        return backend_for(self.recipe)

    @property
    def base_url(self) -> str:
        """The OpenAI-compatible ``/v1`` endpoint this server answers on."""
        return self.backend.base_url(self)

    @property
    def api_key(self) -> str:
        """The bearer credential that reaches this server's endpoint."""
        return self.backend.api_key(self)

    @property
    def auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def handle(self, *, pid: int | None = None) -> ServerHandle:
        return ServerHandle(
            recipe=self.recipe, port=self.served_port, pid=pid, base_url=self.base_url, api_key=self.api_key
        )

    async def health(self) -> bool:
        """Return ``True`` when ``GET /v1/models`` answers 2xx within the backend's probe timeout."""
        async with health_client() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/models", headers=self.auth, timeout=self.backend.probe_timeout_s(self)
                )
            except httpx.HTTPError:
                return False
        return response.is_success

    async def probe(self) -> bool:
        """Report health without waking a scale-to-zero backend — a passive, side-effect-free check.

        A local recipe probes its loopback endpoint (free); a hosted recipe observes its deployment
        through Modal's control plane, never an HTTP request that would cold-boot and bill a container.
        """
        return await self.backend.probe(self)

    async def verify_served_model(self) -> None:
        """Raise :class:`ServeError` unless ``GET /v1/models`` lists every model this server requires.

        This is an identity check, not authentication: a process holding the endpoint could spoof
        the model ids, and trusting the resolved endpoint is inherent to the design.
        """
        if not (want := self.required_models):
            return
        async with health_client() as client:
            response = await client.get(
                f"{self.base_url}/models", headers=self.auth, timeout=self.backend.probe_timeout_s(self)
            )
        served = {model["id"] for model in response.json()["data"]}
        if missing := (want - served):
            raise ServeError(
                f"{self.recipe} at {self.base_url} is missing {sorted(missing)}; it serves {sorted(served)}"
            )

    async def wait_healthy(self) -> None:
        timeout = self.backend.ready_timeout_s(self)
        deadline = anyio.current_time() + timeout
        while True:
            healthy = await self.health()
            if anyio.current_time() >= deadline:
                raise HealthTimeout(f"{self.recipe} not healthy at {self.base_url} after {timeout:.0f}s")
            if healthy:
                await self.verify_served_model()
                return
            await anyio.sleep(READY_POLL_S)

    async def ensure(self, *, persistent: bool = False) -> ServerHandle:
        """Return a healthy server for this recipe, starting it if necessary.

        A recipe already answering healthy returns its handle without restarting. Otherwise the
        backend brings it up — a local recipe launches its command vector, detached via
        :mod:`athome.detach` (``persistent=False``) or as a launchd ``KeepAlive`` agent
        (``persistent=True``); a hosted recipe deploys its Modal app — and the call blocks until
        health polling succeeds.

        Args:
            persistent: Install a launchd KeepAlive agent instead of a detached run (local recipes only).

        Returns:
            The running server's handle.

        Raises:
            ServeError: The adopted or launched server serves a different model.
            HealthTimeout: The server did not report healthy within the ready timeout.
        """
        if await self.health():
            await self.verify_served_model()
            return self.handle(pid=self.backend.adopt(self))
        pid: int | None = None
        try:
            pid = await self.backend.start(self, persistent=persistent)
            await self.wait_healthy()
        except BaseException:
            with anyio.CancelScope(shield=True):
                await self.stop()
            raise
        return self.handle(pid=pid)

    async def stop(self) -> None:
        """Stop this recipe by tearing down its backend.

        A local recipe boots out its launchd agent *and* kills its detached process group — the
        two run independently, so a launchd uninstall failure never skips the process kill. A
        hosted recipe has no local process: Modal's scaledown reaps it, so this is a no-op.
        """
        await self.backend.stop(self)

    def idle(self, *, ttl_s: float) -> IdleResource[ServerHandle]:
        """An :class:`~athome.idle.IdleResource` that spawns this server on first use and stops it once idle.

        Each :meth:`~athome.idle.IdleResource.use` ensures the server is healthy and yields its handle;
        the resource's reaper stops the server once it has sat unused past ``ttl_s``.

        Args:
            ttl_s: Idle seconds before the reaper stops the server.

        Returns:
            A resource wired to :meth:`ensure` and :meth:`stop`.
        """
        return IdleResource(self.ensure, self.stop, ttl_s=ttl_s)

    def client(self, *, cached: bool = False) -> AsyncOpenAI:
        """Return an ``AsyncOpenAI`` client bound to this recipe's endpoint.

        Args:
            cached: Route requests through :func:`athome.llmcache.transport` for record/replay.

        Returns:
            A client with ``base_url`` set to the recipe's ``/v1`` endpoint and its api key.
        """
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=httpx.AsyncClient(transport=llmcache.transport() if cached else None),
        )


async def up(recipe: Recipe, *, persistent: bool = False) -> ServerHandle:
    """Ensure ``recipe`` is running and healthy, returning its handle."""
    return await ManagedServer(recipe).ensure(persistent=persistent)


async def down(recipe: Recipe) -> None:
    """Stop ``recipe`` — launchd uninstall or detached-process kill."""
    await ManagedServer(recipe).stop()


async def probe_all() -> list[tuple[Recipe, bool]]:
    """Report the health of every configured recipe, without waking a scale-to-zero endpoint."""
    return [(recipe, await ManagedServer(recipe).probe()) for recipe in RECIPES if configured(recipe)]


@click.group("serve")
def cli() -> None:
    """Start, stop, and inspect recipe-configured local model servers."""


@cli.command("up")
@click.argument("recipe", type=RECIPE_CHOICE, default=DEFAULT_RECIPE)
@click.option("--persistent", is_flag=True, help="Install a launchd KeepAlive agent instead of a detached run.")
@json_option
@coro
async def up_command(recipe: str, persistent: bool, as_json: bool) -> None:
    """Start RECIPE (default rapid-mlx) and wait for it to report healthy."""
    emit(asdict(await up(recipe, persistent=persistent)), as_json=as_json)


@cli.command("down")
@click.argument("recipe", type=RECIPE_CHOICE, default=DEFAULT_RECIPE)
@json_option
@coro
async def down_command(recipe: str, as_json: bool) -> None:
    """Stop RECIPE (default rapid-mlx)."""
    await down(recipe)
    emit({"stopped": recipe}, as_json=as_json)


@cli.command("status")
@click.argument("recipe", type=RECIPE_CHOICE, default=DEFAULT_RECIPE)
@json_option
@coro
async def status_command(recipe: str, as_json: bool) -> None:
    """Print RECIPE's current health (default rapid-mlx), without waking a scale-to-zero endpoint."""
    emit({"recipe": recipe, "healthy": await ManagedServer(recipe).probe()}, as_json=as_json)


@cli.command("activator")
@click.option("--host", default=None, help="Bind address; overrides the configured activator host.")
def activator_command(host: str | None) -> None:
    """Run the probe-safe idle-unload activator proxy (needs the ``activator`` extra)."""
    from athome.activator import serve_activator

    serve_activator(host=host)


@cli.command("stt")
@click.option(
    "--fd",
    type=int,
    default=None,
    help="Serve on this inherited listener fd (the activator's {LISTEN_FD}) instead of the configured host/port.",
)
def stt_command(fd: int | None) -> None:
    """Run the OpenAI-compatible STT transcription server (needs the ``stt`` extra)."""
    from athome.stt.server import serve_stt

    serve_stt(fd=fd)


@click.command("status")
@json_option
@coro
async def status_cli(as_json: bool) -> None:
    """Report installed athome launchd agents and configured-server health."""
    emit(
        {
            "agents": [asdict(await launchd.status(label)) for label in launchd.installed()],
            "servers": [{"recipe": recipe, "healthy": healthy} for recipe, healthy in await probe_all()],
        },
        as_json=as_json,
    )
