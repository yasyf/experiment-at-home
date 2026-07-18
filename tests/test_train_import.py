from __future__ import annotations

import subprocess
import sys
import sysconfig

import pytest

HEAVY = ("tinker", "mlx", "mlx_lm", "torch", "trl", "peft", "transformers", "datasets", "modal")
FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
FACADE = {
    "TinkerBackend": "athome.train.tinker",
    "observe": "athome.train.observe",
    "ObserveOutcome": "athome.train.observe",
    "SpendGuard": "athome.llm.spend",
    "SpendExceeded": "athome.llm.spend",
    "InvalidBudget": "athome.llm.spend",
}


def test_importing_train_pulls_in_no_heavy_dependency() -> None:
    import athome.train  # noqa: F401
    import athome.train.sidecar  # noqa: F401

    assert not set(HEAVY) & sys.modules.keys()


@pytest.mark.parametrize(("name", "source"), FACADE.items(), ids=list(FACADE))
def test_train_facade_re_exports_the_full_surface(name: str, source: str) -> None:
    import importlib

    import athome.train

    assert getattr(athome.train, name) is getattr(importlib.import_module(source), name)


@pytest.mark.skipif(not FREE_THREADED, reason="`-X gil=0` is fatal on a GIL-enabled build, not ignored")
def test_train_imports_clean_with_the_gil_disabled() -> None:
    probe = (
        "import sys, athome.train, athome.train.sidecar;"
        "assert not sys._is_gil_enabled();"
        f"assert not {set(HEAVY)!r} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-X", "gil=0", "-c", probe], check=True)
