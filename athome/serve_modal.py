from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from anyio import to_thread

if TYPE_CHECKING:
    import modal

    from athome.serve import ModalVllmSettings

CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu22.04"
PYTHON = "3.12"
VLLM_PORT = 8000
HF_CACHE_MOUNT = "/root/.cache/huggingface"
ADAPTER_MOUNT = "/adapter"
ARGV_ENV = "VLLM_SERVE_ARGV"
API_KEY_ENV = "VLLM_API_KEY"
ENDPOINT_FUNCTION = "serve"
FINGERPRINT_TAG = "athome-pins"


def vllm_command(settings: ModalVllmSettings) -> list[str]:
    """The ``vllm serve`` argv the container launches: the pinned base at its exact commit,
    the unfused LoRA adapter served alongside it, and bf16 weights.

    The api key is not on the argv — vLLM reads it from the ``VLLM_API_KEY`` environment variable,
    injected as a Modal secret, so the credential never lands in an image layer or a process listing.
    """
    return [
        "vllm",
        "serve",
        settings.base_model,
        "--revision",
        settings.hf_revision,
        "--tokenizer-revision",
        settings.hf_revision,
        "--served-model-name",
        settings.served_model_name,
        "--enable-lora",
        "--max-lora-rank",
        str(settings.max_lora_rank),
        "--lora-modules",
        f"{settings.adapter_name}={ADAPTER_MOUNT}",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(settings.max_model_len),
        "--gpu-memory-utilization",
        str(settings.gpu_memory_utilization),
        "--enforce-eager",
        "--max-logprobs",
        str(settings.max_logprobs),
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]


def fingerprint(settings: ModalVllmSettings) -> str:
    """A stable digest of the pins that define a deployment: the vLLM version, the GPU class,
    and the full ``vllm serve`` argv (base revision, adapter, and engine flags).

    Stamped Modal-side as an app tag at deploy time and compared on wake, so a scaled-to-zero
    endpoint is re-deployed only when its pins have actually drifted — never on every cold start.
    """
    payload = json.dumps(
        {"vllm_version": settings.vllm_version, "gpu": settings.gpu, "argv": vllm_command(settings)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def serve_image(settings: ModalVllmSettings) -> modal.Image:
    """Build the vLLM serving image: the CUDA devel base, the pinned vLLM, and the resolved argv.

    The argv is baked into the image as an environment variable rather than closed over by the
    entrypoint, so a change to any served pin busts the cheap env layer — the deployed endpoint can
    never drift from the recipe's configuration — while leaving the expensive vLLM install cached.
    """
    import modal

    return (
        modal.Image.from_registry(CUDA_IMAGE, add_python=PYTHON)
        .entrypoint([])
        .pip_install(f"vllm=={settings.vllm_version}", "huggingface_hub[hf_transfer]")
        .env(
            {
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "VLLM_USE_V1": "1",
                ARGV_ENV: json.dumps(vllm_command(settings)),
            }
        )
    )


def build_app(settings: ModalVllmSettings) -> modal.App:
    """Assemble the Modal app: one scale-to-zero GPU web server fronting ``vllm serve``.

    The web-server entrypoint is a nested function, not a module-level one: Modal serializes the
    registered function with cloudpickle, and only a nested (or otherwise non-importable) function
    is embedded by value. A module-level entrypoint would be pickled by reference — a 43-byte pickle
    naming ``athome.serve_modal.serve`` — that every cold container fails to load, because the image
    carries vLLM and ``huggingface_hub`` but never ``athome``. The entrypoint closes over nothing
    from this module: the argv travels in the ``VLLM_SERVE_ARGV`` image env var, so its body reads
    that variable and launches the server, preserving the no-drift design.

    ``min_containers=0`` with ``scaledown_window`` is the scale-to-zero: the endpoint costs nothing
    idle and cold-starts on the next request, bounded by ``startup_timeout``. ``max_concurrent_inputs``
    is vLLM's continuous-batching fan-in, and the HF cache and adapter volumes carry the base weights
    and the unfused LoRA across cold starts. The pin fingerprint rides along as an app tag so a wake
    can tell a matching deployment from a drifted one without redeploying.
    """
    import modal

    def serve() -> None:
        import json
        import os
        import subprocess

        subprocess.Popen(json.loads(os.environ[ARGV_ENV]))

    serve.__name__ = ENDPOINT_FUNCTION

    app = modal.App(settings.app_name, tags={FINGERPRINT_TAG: fingerprint(settings)})
    endpoint = modal.web_server(port=VLLM_PORT, startup_timeout=settings.startup_timeout)(serve)
    endpoint = modal.concurrent(max_inputs=settings.max_concurrent_inputs)(endpoint)
    app.function(
        image=serve_image(settings),
        gpu=settings.gpu,
        volumes={
            HF_CACHE_MOUNT: modal.Volume.from_name(settings.hf_cache_volume, create_if_missing=True),
            ADAPTER_MOUNT: modal.Volume.from_name(settings.adapter_volume, create_if_missing=True),
        },
        secrets=[modal.Secret.from_dict({API_KEY_ENV: settings.api_key})],
        timeout=settings.request_timeout,
        scaledown_window=settings.scaledown_window,
        min_containers=0,
        serialized=True,
    )(endpoint)
    return app


async def deploy_app(app: modal.App) -> None:
    await to_thread.run_sync(app.deploy)


async def deploy(settings: ModalVllmSettings) -> None:
    """Deploy the recipe's Modal app so its endpoint is reachable, then return.

    Deploying mints a fresh revision, so a caller redeploys only when the endpoint is missing or its
    pins have drifted — a warm, matching endpoint is woken by its first request, not redeployed.
    """
    await deploy_app(build_app(settings))


async def locate(settings: ModalVllmSettings) -> modal.Function | None:
    """The deployed web-server function, hydrated, or ``None`` when the app is not deployed.

    A control-plane lookup only: hydrating the handle talks to Modal's API, never the web endpoint,
    so it observes a scaled-to-zero deployment without waking (and billing) a container.
    """
    import modal
    from modal.exception import NotFoundError

    def resolve() -> modal.Function:
        function = modal.Function.from_name(settings.app_name, ENDPOINT_FUNCTION)
        function.hydrate()
        return function

    try:
        return await to_thread.run_sync(resolve)
    except NotFoundError:
        return None


async def deployed_fingerprint(settings: ModalVllmSettings) -> str | None:
    """The pin fingerprint stamped on the currently deployed app, or ``None`` when it is not deployed.

    Read from the app's Modal-side tags — the same control plane the deploy stamps — so a wake sees
    the pins the live endpoint actually runs, not a machine-local guess that a peer's redeploy could
    silently invalidate.
    """
    import modal
    from modal.exception import NotFoundError

    def read() -> str | None:
        return modal.App.lookup(settings.app_name).get_tags().get(FINGERPRINT_TAG)

    try:
        return await to_thread.run_sync(read)
    except NotFoundError:
        return None


def web_url(function: modal.Function) -> str | None:
    """The authoritative web endpoint Modal assigned the deployed function."""
    return function.get_web_url()
