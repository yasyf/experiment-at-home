from __future__ import annotations

import os
import shlex
import signal
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import anyio
import click
import httpx
from pydantic import ValidationError

from athome import detach, launchd, llmcache
from athome.cli import coro, emit, json_option
from athome.config import SectionSettings, load
from athome.errors import AthomeError
from athome.launchd import AgentSpec, KeepAlive, LaunchdError

if TYPE_CHECKING:
    from openai import AsyncOpenAI

type Recipe = Literal["rapid-mlx", "mlx-vlm", "llama-server"]

RECIPES: tuple[Recipe, ...] = ("rapid-mlx", "mlx-vlm", "llama-server")
HEALTH_TIMEOUT_S = 1.0
READY_TIMEOUT_S = 120.0
READY_POLL_S = 1.0
RECIPE_CHOICE = click.Choice(list(RECIPES))


class ServeError(AthomeError):
    """Raised when a managed server cannot be started, stopped, or reached."""


class HealthTimeout(ServeError):
    """Raised when a spawned server does not report healthy within the ready timeout."""


class RapidMlxSettings(SectionSettings):
    """The ``[serve.rapid-mlx]`` section: the daily-pinned rapid-mlx version, model, and port."""

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
    """The ``[serve.llama-server]`` section: a full llama-server command string and its port."""

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


def command_for(recipe: Recipe) -> tuple[str, ...]:
    match recipe:
        case "rapid-mlx":
            settings = load(RapidMlxSettings)
            return (
                "uvx",
                "--from",
                f"rapid-mlx=={settings.version}",
                "rapid-mlx",
                "serve",
                settings.model,
                "--port",
                str(settings.port),
            )
        case "mlx-vlm":
            settings = load(MlxVlmSettings)
            return (
                "uvx",
                "--from",
                f"mlx-vlm=={settings.version}",
                "mlx_vlm.server",
                "--model",
                settings.model,
                "--port",
                str(settings.port),
            )
        case "llama-server":
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

    Example:
        >>> server = ManagedServer("rapid-mlx")
        >>> handle = await server.ensure()
    """

    recipe: Recipe

    def handle(self, *, pid: int | None = None) -> ServerHandle:
        port = settings_for(self.recipe).port
        return ServerHandle(recipe=self.recipe, port=port, pid=pid, base_url=f"http://127.0.0.1:{port}/v1")

    async def health(self) -> bool:
        """Return ``True`` when ``GET /v1/models`` answers 2xx within the health timeout."""
        url = f"http://127.0.0.1:{settings_for(self.recipe).port}/v1/models"
        async with health_client() as client:
            try:
                response = await client.get(url)
            except httpx.HTTPError:
                return False
        return response.is_success

    async def verify_served_model(self) -> None:
        """Raise :class:`ServeError` unless ``GET /v1/models`` lists this recipe's configured model.

        This is an identity check, not authentication: a hostile process holding the port could
        spoof the model id, and trusting a localhost endpoint is inherent to the design.
        """
        if (want := configured_model(self.recipe)) is None:
            return
        port = settings_for(self.recipe).port
        async with health_client() as client:
            response = await client.get(f"http://127.0.0.1:{port}/v1/models")
        if want not in (served := [model["id"] for model in response.json()["data"]]):
            raise ServeError(f"{self.recipe} on port {port} serves {served}, not the configured model {want!r}")

    async def wait_healthy(self) -> None:
        deadline = anyio.current_time() + READY_TIMEOUT_S
        while True:
            healthy = await self.health()
            if anyio.current_time() >= deadline:
                port = settings_for(self.recipe).port
                raise HealthTimeout(f"{self.recipe} not healthy on port {port} after {READY_TIMEOUT_S:.0f}s")
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
            return self.handle(pid=detach.running(detach_name(self.recipe)))
        command = command_for(self.recipe)
        pid: int | None = None
        try:
            if persistent:
                await launchd.install(
                    AgentSpec(
                        label=agent_label(self.recipe),
                        command=command,
                        schedule=KeepAlive(),
                        log_name=detach_name(self.recipe),
                    )
                )
            else:
                pid = (await detach.launch(command, name=detach_name(self.recipe))).pid
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
        if agent_label(self.recipe) in launchd.installed():
            try:
                await launchd.uninstall(agent_label(self.recipe))
            except LaunchdError:
                pass
        if (pid := detach.running(detach_name(self.recipe))) is not None:
            try:
                kill_group(pid)
            except ProcessLookupError:
                pass

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
@click.argument("recipe", type=RECIPE_CHOICE)
@click.option("--persistent", is_flag=True, help="Install a launchd KeepAlive agent instead of a detached run.")
@json_option
@coro
async def up_command(recipe: str, persistent: bool, as_json: bool) -> None:
    """Start RECIPE and wait for it to report healthy."""
    emit(asdict(await up(recipe, persistent=persistent)), as_json=as_json)


@cli.command("down")
@click.argument("recipe", type=RECIPE_CHOICE)
@json_option
@coro
async def down_command(recipe: str, as_json: bool) -> None:
    """Stop RECIPE."""
    await down(recipe)
    emit({"stopped": recipe}, as_json=as_json)


@cli.command("status")
@click.argument("recipe", type=RECIPE_CHOICE)
@json_option
@coro
async def status_command(recipe: str, as_json: bool) -> None:
    """Print RECIPE's current health."""
    emit({"recipe": recipe, "healthy": await ManagedServer(recipe).health()}, as_json=as_json)


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
