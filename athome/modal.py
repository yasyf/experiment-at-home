from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from anyio import Lock

from athome.config import SectionSettings, load
from athome.errors import AthomeError

if TYPE_CHECKING:
    from collections.abc import Callable

    from athome.wire import Wire


class ModalError(AthomeError):
    """Root of every athome Modal-backend error."""


class ParityMismatch(ModalError):
    """The remote Modal image or engine params drifted from the local pins."""


class ModalSettings(SectionSettings):
    """The ``[modal]`` config section: how remote Modal apps are named.

    Modal authenticates with the SDK's ambient credentials (``modal token``
    / ``MODAL_TOKEN_*``); athome routes no token through settings. ``app_prefix``
    names a workspace-scoped app, so aiming at the wrong workspace fails loud on
    ``modal.Cls.from_name`` rather than silently serving a stranger's app.
    """

    section = ("modal",)
    app_prefix: str = "athome"


@dataclass(frozen=True, slots=True)
class ServiceSpec:
    """Declares the Modal side of an engine from the SAME module that runs it locally.

    ``version_packages`` and ``params`` are the parity fingerprint inputs: the remote
    container reports them back and the client refuses to serve on any drift, so remote
    and local results stay byte-comparable for bake-offs and replay caches.

    Example:
        >>> ServiceSpec("ocr-paddle", ("onnxruntime", "rapidocr"), {"upscale": 2})
    """

    name: str
    version_packages: tuple[str, ...]
    params: dict[str, Wire]


@dataclass(slots=True, eq=False)
class ModalWorkerTransport:
    """Wire-compatible relay to a modal.Cls — a drop-in for a local PipeWorker.

    On first call: invokes the remote fingerprint() (forces hydration so a missing
    token / undeployed app fails BEFORE ready), asserts remote package versions +
    params match local pins -> ParityMismatch on skew. NO local fallback: Modal
    unreachable = the worker dies loudly, same as a crashed local engine.

    Example:
        >>> transport = ModalWorkerTransport(spec, "PaddleRemote")
        >>> await transport.call("tokens", image)
    """

    spec: ServiceSpec
    class_name: str
    lock: Lock = field(default_factory=Lock)
    remote: object | None = None

    async def call(self, method: str, payload: Wire) -> Wire:
        await self.ensure_ready()
        return await getattr(self.remote, method).remote.aio(payload)

    async def aclose(self) -> None:
        self.remote = None

    async def ensure_ready(self) -> None:
        if self.remote is not None:
            return
        async with self.lock:
            if self.remote is not None:
                return
            remote = self.locate()
            if mismatches := parity_mismatches(self.spec, await remote.fingerprint.remote.aio()):
                raise ParityMismatch(f"{self.app_name}: " + "; ".join(mismatches))
            self.remote = remote

    def locate(self) -> object:
        import modal

        return modal.Cls.from_name(self.app_name, self.class_name)()

    @property
    def app_name(self) -> str:
        return f"{load(ModalSettings).app_prefix}-{self.spec.name}"


def fingerprint_for(spec: ServiceSpec) -> dict[str, Wire]:
    """Compute a service's parity fingerprint: installed package versions folded with its params.

    Packages and params live in disjoint namespaces (``pkg:`` / ``param:``) so a param
    named like a package cannot overwrite its pinned version.
    """
    return {f"pkg:{pkg}": importlib.metadata.version(pkg) for pkg in spec.version_packages} | {
        f"param:{key}": value for key, value in spec.params.items()
    }


def parity_mismatches(spec: ServiceSpec, remote: dict[str, Wire]) -> list[str]:
    local = fingerprint_for(spec)
    return [
        f"{key}: local {local.get(key)!r} != remote {remote.get(key)!r}"
        for key in sorted(local.keys() | remote.keys())
        if (key in local) != (key in remote) or local.get(key) != remote.get(key)
    ]


def image_recipe(spec: ServiceSpec, *, python: str, local_source: str, download_models: Callable) -> object:
    """Build the Modal image that hosts a service, baking its model weights in at build time.

    ``download_models`` runs during the build so runtime containers resolve every weight
    from the baked cache, never the network. The local source is copied in first so that
    build step can import it.
    """
    import modal

    return (
        modal.Image.debian_slim(python_version=python)
        .uv_sync()
        .add_local_python_source(local_source, copy=True)
        .run_function(download_models)
    )
