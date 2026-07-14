from __future__ import annotations

import subprocess
import sys
import sysconfig

import pytest

HEAVY = ("tinker", "mlx", "mlx_lm", "torch", "trl", "peft", "transformers", "datasets", "modal")
FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def test_importing_train_pulls_in_no_heavy_dependency() -> None:
    import athome.train  # noqa: F401
    import athome.train.sidecar  # noqa: F401

    assert not set(HEAVY) & sys.modules.keys()


@pytest.mark.skipif(not FREE_THREADED, reason="`-X gil=0` is fatal on a GIL-enabled build, not ignored")
def test_train_imports_clean_with_the_gil_disabled() -> None:
    probe = (
        "import sys, athome.train, athome.train.sidecar;"
        "assert not sys._is_gil_enabled();"
        f"assert not {set(HEAVY)!r} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-X", "gil=0", "-c", probe], check=True)
