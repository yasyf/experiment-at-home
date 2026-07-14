from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from athome.config import load
from athome.train import sidecar
from athome.train.sidecar import SIDECAR, convert_peft_to_mlx, fuse, mlx_lm_command, run_convert, sidecar_command
from athome.train.spec import BASE_MODELS, STD_MODULES, LoraSpec

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

BASE = BASE_MODELS["qwen3-8b"]
VERSION = "0.31.3"


@dataclass(frozen=True, slots=True)
class FakeArray:
    label: str

    @property
    def T(self) -> FakeArray:  # noqa: N802 — mlx's transpose accessor
        return FakeArray(f"{self.label}.T")


@pytest.fixture
def commands(monkeypatch: pytest.MonkeyPatch) -> list[Sequence[str]]:
    seen: list[Sequence[str]] = []

    async def run_process(command: Sequence[str]) -> None:
        seen.append(command)

    monkeypatch.setattr(sidecar, "run_process", run_process)
    return seen


@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, FakeArray]]:
    written: dict[str, dict[str, FakeArray]] = {}
    core = ModuleType("mlx.core")
    core.load = lambda path: {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": FakeArray("q_a"),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": FakeArray("q_b"),
        "base_model.model.model.layers.1.mlp.up_proj.lora_A.weight": FakeArray("up_a"),
        "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight": FakeArray("qkv_a"),
        "base_model.model.unembed_tokens.weight": FakeArray("unembed"),
    }
    core.save_safetensors = lambda path, weights: written.__setitem__(path, weights)
    package = ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return written


@pytest.fixture(autouse=True)
def pinned_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATHOME_TRAIN_MLX_LM_VERSION", VERSION)
    load.cache_clear()


def test_the_sidecar_runs_in_the_pinned_mlx_lm_environment() -> None:
    assert mlx_lm_command("fuse") == ("uvx", "--from", f"mlx-lm=={VERSION}", "python", "-m", "mlx_lm", "fuse")
    assert sidecar_command("convert") == ("uvx", "--from", f"mlx-lm=={VERSION}", "python", str(SIDECAR), "convert")
    assert SIDECAR.name == "sidecar.py"


async def test_convert_peft_to_mlx_passes_the_adapter_shape_to_the_sidecar(
    commands: list[Sequence[str]], tmp_path: Path
) -> None:
    out = await convert_peft_to_mlx(tmp_path / "peft", tmp_path / "adapter", base=BASE, lora=LoraSpec())

    assert out == tmp_path / "adapter"
    assert commands == [
        (
            "uvx",
            "--from",
            f"mlx-lm=={VERSION}",
            "python",
            str(SIDECAR),
            "convert",
            "--peft",
            str(tmp_path / "peft"),
            "--out",
            str(tmp_path / "adapter"),
            "--num-layers",
            "36",
            "--rank",
            "16",
            "--alpha",
            "32",
            "--modules",
            ",".join(STD_MODULES),
        )
    ]


async def test_fuse_saves_a_standalone_model_from_the_base_and_the_adapter(
    commands: list[Sequence[str]], tmp_path: Path
) -> None:
    out = await fuse(tmp_path / "adapter", tmp_path / "fused", base=BASE)

    assert out == tmp_path / "fused"
    assert commands == [
        (
            "uvx",
            "--from",
            f"mlx-lm=={VERSION}",
            "python",
            "-m",
            "mlx_lm",
            "fuse",
            "--model",
            "mlx-community/Qwen3-8B-4bit",
            "--adapter-path",
            str(tmp_path / "adapter"),
            "--save-path",
            str(tmp_path / "fused"),
        )
    ]
    assert "--dequantize" not in commands[0]


def test_convert_transposes_the_standard_modules_and_drops_the_rest(
    saved: dict[str, dict[str, FakeArray]], tmp_path: Path
) -> None:
    report = run_convert(tmp_path / "peft", tmp_path / "adapter", num_layers=36, rank=16, alpha=32, modules=STD_MODULES)

    assert saved[str(tmp_path / "adapter" / "adapters.safetensors")] == {
        "model.layers.0.self_attn.q_proj.lora_a": FakeArray("q_a.T"),
        "model.layers.0.self_attn.q_proj.lora_b": FakeArray("q_b.T"),
        "model.layers.1.mlp.up_proj.lora_a": FakeArray("up_a.T"),
    }
    assert report == {"n_lora_weights": 3, "dropped": ["linear_attn.in_proj_qkv", "unembed_tokens.weight"]}


def test_convert_writes_the_adapter_config_mlx_lm_reads(saved: dict[str, dict[str, FakeArray]], tmp_path: Path) -> None:
    run_convert(tmp_path / "peft", tmp_path / "adapter", num_layers=36, rank=16, alpha=32, modules=STD_MODULES)

    assert json.loads((tmp_path / "adapter" / "adapter_config.json").read_text()) == {
        "fine_tune_type": "lora",
        "num_layers": 36,
        "lora_parameters": {"rank": 16, "scale": 2.0, "dropout": 0.0, "keys": list(STD_MODULES)},
    }
