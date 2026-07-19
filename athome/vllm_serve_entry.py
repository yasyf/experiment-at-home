"""Container-synced vLLM launcher for the hosted ``modal-vllm`` serve recipe.

Modal registers the web-server function by reference and syncs *this* module's source into the vLLM
container, importing it there to resolve the function. The container carries vLLM and Modal but not
``athome`` (nor its dependencies), so this module must stay importable with only the standard library:
no ``athome`` import and no third-party import at container import time. Everything the launcher needs
travels through the environment — the ``vllm serve`` argv baked into the serving image at deploy time,
and the API key injected as a Modal secret — so its body reads a single environment variable and execs
the server. Loading the file as a top-level module (never ``athome.vllm_serve_entry``) is what makes
Modal mount the single file and import it by its bare stem, so ``athome/__init__.py`` never runs in
the container.
"""

from __future__ import annotations

import json
import os
import subprocess

ARGV_ENV = "VLLM_SERVE_ARGV"


def serve() -> None:
    """Launch ``vllm serve`` from the deploy-baked argv in ``VLLM_SERVE_ARGV``.

    vLLM reads its API key from the ``VLLM_API_KEY`` environment variable (a Modal secret), so the
    credential never lands on the argv or in a process listing.
    """
    subprocess.Popen(json.loads(os.environ[ARGV_ENV]))
