from __future__ import annotations

import hashlib
import os
import shlex
import signal
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal

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

if TYPE_CHECKING:
    from openai import AsyncOpenAI

type Recipe = Literal["rapid-mlx", "mlx-vlm", "llama-server"]

RECIPES: tuple[Recipe, ...] = ("rapid-mlx", "mlx-vlm", "llama-server")
DEFAULT_RECIPE: Recipe = "rapid-mlx"
HEALTH_TIMEOUT_S = 1.0
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


type RecipeSettings = RapidMlxSettings | MlxVlmSettings | LlamaServerSettings


@dataclass(frozen=True, slots=True)
class ServerHandle:
    """A running recipe's port, pid (``None`` for a launchd service), and OpenAI base URL.

    Example:
        >>> handle = await up("mlx-vlm")
        >>> handle.base_url
        'http://127.0.0.1:8401/v1'
    """

    recipe: Recipe
    port: int
    pid: int | None
    base_url: str


def settings_for(recipe: Recipe) -> RecipeSettings:
    match recipe:
        case "rapid-mlx":
            return load(RapidMlxSettings)
        case "mlx-vlm":
            return load(MlxVlmSettings)
        case "llama-server":
            return load(LlamaServerSettings)


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


def configured_model(recipe: Recipe) -> str | None:
    match settings_for(recipe):
        case RapidMlxSettings(model=model) | MlxVlmSettings(model=model):
            return model
        case LlamaServerSettings():
            return None


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
    return httpx.AsyncClient(timeout=HEALTH_TIMEOUT_S)


def kill_group(pid: int) -> None:
    os.killpg(os.getpgid(pid), signal.SIGTERM)


@dataclass(frozen=True, slots=True)
class ManagedServer:
    """A recipe-configured OpenAI-compatible local server with lifecycle + health.

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

    @property
    def served_model(self) -> str | None:
        """The model this server serves: the override, else the recipe's configured model."""
        return self.model if self.model is not None else configured_model(self.recipe)

    @property
    def served_port(self) -> int:
        """The port this server listens on: the override, else the recipe's configured port."""
        return self.port if self.port is not None else settings_for(self.recipe).port

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

    def handle(self, *, pid: int | None = None) -> ServerHandle:
        port = self.served_port
        return ServerHandle(recipe=self.recipe, port=port, pid=pid, base_url=f"http://127.0.0.1:{port}/v1")

    async def health(self) -> bool:
        """Return ``True`` when ``GET /v1/models`` answers 2xx within the health timeout."""
        url = f"http://127.0.0.1:{self.served_port}/v1/models"
        async with health_client() as client:
            try:
                response = await client.get(url)
            except httpx.HTTPError:
                return False
        return response.is_success

    async def verify_served_model(self) -> None:
        """Raise :class:`ServeError` unless ``GET /v1/models`` lists the model this server serves.

        This is an identity check, not authentication: a hostile process holding the port could
        spoof the model id, and trusting a localhost endpoint is inherent to the design.
        """
        if (want := self.served_model) is None:
            return
        port = self.served_port
        async with health_client() as client:
            response = await client.get(f"http://127.0.0.1:{port}/v1/models")
        if want not in (served := [model["id"] for model in response.json()["data"]]):
            raise ServeError(f"{self.recipe} on port {port} serves {served}, not the configured model {want!r}")

    async def wait_healthy(self) -> None:
        deadline = anyio.current_time() + READY_TIMEOUT_S
        while True:
            healthy = await self.health()
            if anyio.current_time() >= deadline:
                raise HealthTimeout(
                    f"{self.recipe} not healthy on port {self.served_port} after {READY_TIMEOUT_S:.0f}s"
                )
            if healthy:
                await self.verify_served_model()
                return
            await anyio.sleep(READY_POLL_S)

    async def ensure(self, *, persistent: bool = False) -> ServerHandle:
        """Return a healthy server for this recipe, spawning it if necessary.

        A recipe already answering healthy on its port returns its handle without
        respawning. Otherwise the recipe's command vector is launched — detached via
        :mod:`athome.detach` (``persistent=False``) or as a launchd ``KeepAlive`` agent
        (``persistent=True``) — and the call blocks until health polling succeeds.

        Args:
            persistent: Install a launchd KeepAlive agent instead of a detached run.

        Returns:
            The running server's handle.

        Raises:
            ServeError: The adopted or launched server serves a different model.
            HealthTimeout: The server did not report healthy within the ready timeout.
        """
        if await self.health():
            await self.verify_served_model()
            return self.handle(pid=detach.running(self.run_name))
        command = command_for(self.recipe, model=self.model, port=self.port)
        pid: int | None = None
        try:
            if persistent:
                await launchd.install(
                    AgentSpec(
                        label=self.label,
                        command=command,
                        schedule=KeepAlive(),
                        log_name=self.run_name,
                    )
                )
            else:
                pid = (await detach.launch(command, name=self.run_name)).pid
            await self.wait_healthy()
        except BaseException:
            with anyio.CancelScope(shield=True):
                await self.stop()
            raise
        return self.handle(pid=pid)

    async def stop(self) -> None:
        """Stop this recipe — boot out its launchd agent *and* kill its detached process group.

        The two teardowns run independently: a launchd uninstall failure never skips the
        detached-process kill.
        """
        if self.label in launchd.installed():
            try:
                await launchd.uninstall(self.label)
            except LaunchdError:
                pass
        if (pid := detach.running(self.run_name)) is not None:
            try:
                kill_group(pid)
            except ProcessLookupError:
                pass

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
            A client with ``base_url`` set to the recipe's ``/v1`` endpoint and a local api key.
        """
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            base_url=self.handle().base_url,
            api_key="local",
            http_client=httpx.AsyncClient(transport=llmcache.transport() if cached else None),
        )


async def up(recipe: Recipe, *, persistent: bool = False) -> ServerHandle:
    """Ensure ``recipe`` is running and healthy, returning its handle."""
    return await ManagedServer(recipe).ensure(persistent=persistent)


async def down(recipe: Recipe) -> None:
    """Stop ``recipe`` — launchd uninstall or detached-process kill."""
    await ManagedServer(recipe).stop()


async def probe_all() -> list[tuple[Recipe, bool]]:
    """Report the health of every configured recipe."""
    return [(recipe, await ManagedServer(recipe).health()) for recipe in RECIPES if configured(recipe)]


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
    """Print RECIPE's current health (default rapid-mlx)."""
    emit({"recipe": recipe, "healthy": await ManagedServer(recipe).health()}, as_json=as_json)


@cli.command("activator")
@click.option("--host", default=None, help="Bind address; overrides the configured activator host.")
def activator_command(host: str | None) -> None:
    """Run the probe-safe idle-unload activator proxy (needs the ``activator`` extra)."""
    from athome.activator import serve_activator

    serve_activator(host=host)


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
