from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from athome.config import load
from athome.train import sidecar
from athome.train.sidecar import (
    GENERATION_FENCE,
    PEFT_CONFIG,
    SIDECAR,
    convert_peft_to_mlx,
    fuse,
    generate,
    mlx_lm_command,
    run_convert,
    sidecar_command,
    strip_generation,
)
from athome.train.spec import BASE_MODELS, STD_MODULES, UnservableBase

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

BASE = BASE_MODELS["qwen3-8b"]
VERSION = "0.31.3"
FUSABLE: dict[str, str] = {
    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": "q_a",
    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": "q_b",
    "base_model.model.model.layers.1.mlp.up_proj.lora_A.weight": "up_a",
    "base_model.model.model.layers.1.mlp.up_proj.lora_B.weight": "up_b",
}


@dataclass(frozen=True, slots=True)
class FakeArray:
    label: str

    @property
    def T(self) -> FakeArray:  # noqa: N802 — mlx's transpose accessor
        return FakeArray(f"{self.label}.T")


@dataclass(frozen=True, slots=True)
class Archive:
    """A PEFT adapter on disk: its ``adapter_config.json``, and the tensors it trained."""

    peft_dir: Path
    out_dir: Path
    saved: dict[str, dict[str, FakeArray]]


@pytest.fixture
def commands(monkeypatch: pytest.MonkeyPatch) -> list[Sequence[str]]:
    seen: list[Sequence[str]] = []

    async def run_process(command: Sequence[str]) -> None:
        seen.append(command)

    monkeypatch.setattr(sidecar, "run_process", run_process)
    return seen


@pytest.fixture
def archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Archive:
    """Writes a PEFT archive whose config and weights are the *trained* shape, not a requested one."""
    state = Archive(tmp_path / "peft", tmp_path / "adapter", {})
    state.peft_dir.mkdir()
    weights: dict[str, FakeArray] = {key: FakeArray(label) for key, label in FUSABLE.items()}
    core = ModuleType("mlx.core")
    core.load = lambda path: weights
    core.save_safetensors = lambda path, tensors: state.saved.__setitem__(path, tensors)
    package = ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return state


def write_config(archive: Archive, **overrides: object) -> None:
    (archive.peft_dir / PEFT_CONFIG).write_text(
        json.dumps({"r": 16, "lora_alpha": 32, "lora_dropout": 0.0} | overrides)
    )


def add_weight(archive: Archive, key: str) -> None:
    sys.modules["mlx.core"].load = lambda path: {k: FakeArray(v) for k, v in FUSABLE.items()} | {key: FakeArray("x")}


@pytest.fixture(autouse=True)
def pinned_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATHOME_TRAIN_MLX_LM_VERSION", VERSION)
    load.cache_clear()


def test_the_sidecar_runs_in_the_pinned_mlx_lm_environment() -> None:
    assert mlx_lm_command("fuse") == ("uvx", "--from", f"mlx-lm=={VERSION}", "python", "-m", "mlx_lm", "fuse")
    assert sidecar_command("convert") == ("uvx", "--from", f"mlx-lm=={VERSION}", "python", str(SIDECAR), "convert")
    assert SIDECAR.name == "sidecar.py"


async def test_convert_never_passes_a_requested_shape_to_the_sidecar(
    commands: list[Sequence[str]], tmp_path: Path
) -> None:
    """#6 — the fused shape is read off the archive, so no rank/alpha/modules can be imposed on it."""
    out = await convert_peft_to_mlx(tmp_path / "peft", tmp_path / "adapter", base=BASE)

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
        )
    ]
    assert not {"--rank", "--alpha", "--modules"} & set(commands[0])


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


async def test_convert_refuses_a_base_with_no_mlx_lm_counterpart_before_any_work(
    commands: list[Sequence[str]], tmp_path: Path
) -> None:
    with pytest.raises(UnservableBase, match="mlx-lm LoRA counterpart"):
        await convert_peft_to_mlx(tmp_path / "peft", tmp_path / "adapter", base=BASE_MODELS["qwen3.5-4b"])

    assert commands == []


async def test_fuse_refuses_a_base_with_no_mlx_lm_counterpart_before_any_work(
    commands: list[Sequence[str]], tmp_path: Path
) -> None:
    with pytest.raises(UnservableBase, match="mlx-lm LoRA counterpart"):
        await fuse(tmp_path / "adapter", tmp_path / "fused", base=BASE_MODELS["qwen3.5-4b"])

    assert commands == []


def test_convert_transposes_every_trained_lora_matrix(archive: Archive) -> None:
    write_config(archive)

    report = run_convert(archive.peft_dir, archive.out_dir, num_layers=36)

    assert archive.saved[str(archive.out_dir / "adapters.safetensors")] == {
        "model.layers.0.self_attn.q_proj.lora_a": FakeArray("q_a.T"),
        "model.layers.0.self_attn.q_proj.lora_b": FakeArray("q_b.T"),
        "model.layers.1.mlp.up_proj.lora_a": FakeArray("up_a.T"),
        "model.layers.1.mlp.up_proj.lora_b": FakeArray("up_b.T"),
    }
    assert report == {
        "n_lora_weights": 4,
        "rank": 16,
        "alpha": 32,
        "keys": ["mlp.up_proj", "self_attn.q_proj"],
    }


@pytest.mark.parametrize(
    ("rank", "alpha", "dropout", "scale"),
    [
        pytest.param(16, 32, 0.0, 2.0, id="the-default-scale"),
        pytest.param(8, 8, 0.05, 1.0, id="a-non-default-alpha-reinterprets-the-adapter"),
        pytest.param(32, 128, 0.1, 4.0, id="a-high-alpha-archive"),
    ],
)
def test_the_fused_config_is_the_archives_own_not_a_requested_one(
    archive: Archive, rank: int, alpha: int, dropout: float, scale: float
) -> None:
    """#6 — the converter used to write the *requested* alpha/rank, rescaling weights it never trained."""
    write_config(archive, r=rank, lora_alpha=alpha, lora_dropout=dropout)

    report = run_convert(archive.peft_dir, archive.out_dir, num_layers=36)

    assert json.loads((archive.out_dir / "adapter_config.json").read_text()) == {
        "fine_tune_type": "lora",
        "num_layers": 36,
        "lora_parameters": {
            "rank": rank,
            "scale": scale,
            "dropout": dropout,
            "keys": ["mlp.up_proj", "self_attn.q_proj"],
        },
    }
    assert (report["rank"], report["alpha"]) == (rank, alpha)


def test_the_written_keys_are_the_modules_the_archive_actually_carries(archive: Archive) -> None:
    """#5/#6 — a toggled-off module is absent from the archive, so it is absent from the fused keys."""
    write_config(archive)

    run_convert(archive.peft_dir, archive.out_dir, num_layers=36)

    keys = json.loads((archive.out_dir / "adapter_config.json").read_text())["lora_parameters"]["keys"]
    assert keys == ["mlp.up_proj", "self_attn.q_proj"]
    assert set(keys) < set(STD_MODULES)


@pytest.mark.parametrize(
    "unfusable",
    [
        pytest.param("base_model.model.unembed_tokens.weight", id="an-unembedding-tensor"),
        pytest.param("base_model.model.lm_head.weight", id="a-saved-lm-head"),
    ],
)
def test_a_tensor_that_is_not_a_lora_matrix_raises_instead_of_being_dropped(archive: Archive, unfusable: str) -> None:
    """#6 — silently dropping a trained tensor spends money on weights that never reach the artifact."""
    write_config(archive)
    add_weight(archive, unfusable)

    with pytest.raises(ValueError, match="cannot fuse 1 trained tensor"):
        run_convert(archive.peft_dir, archive.out_dir, num_layers=36)

    assert archive.saved == {}


def furniture(text: str) -> str:
    """A whole ``mlx_lm generate`` stdout: the completion fenced by ``==========`` and its stats."""
    return (
        f"{GENERATION_FENCE}\n{text}\n{GENERATION_FENCE}\n"
        "Prompt: 12 tokens, 480.771 tokens-per-sec\n"
        "Generation: 5 tokens, 39.501 tokens-per-sec\n"
        "Peak memory: 4.321 GB\n"
    )


@pytest.fixture
def generations(monkeypatch: pytest.MonkeyPatch) -> tuple[list[Sequence[str]], list[str]]:
    """Records the argv ``generate`` ran and hands back the canned stdout the sidecar would print."""
    seen: list[Sequence[str]] = []
    canned: list[str] = []

    async def run_output(command: Sequence[str]) -> str:
        seen.append(command)
        return canned[0]

    monkeypatch.setattr(sidecar, "run_output", run_output)
    return seen, canned


async def test_generate_runs_mlx_lm_in_the_pinned_env_and_strips_the_fences(
    generations: tuple[list[Sequence[str]], list[str]],
) -> None:
    seen, canned = generations
    canned.append(furniture("the quick brown fox"))

    out = await generate("mlx-community/Qwen3-8B-4bit", "continue:", max_tokens=32, temperature=0.7, seed=1729)

    assert out == "the quick brown fox"
    assert seen == [
        (
            "uvx",
            "--from",
            f"mlx-lm=={VERSION}",
            "python",
            "-m",
            "mlx_lm",
            "generate",
            "--model",
            "mlx-community/Qwen3-8B-4bit",
            "--prompt",
            "continue:",
            "--max-tokens",
            "32",
            "--temp",
            "0.7",
            "--ignore-chat-template",
            "--seed",
            "1729",
        )
    ]


async def test_generate_omits_the_seed_flag_when_unseeded(
    generations: tuple[list[Sequence[str]], list[str]],
) -> None:
    seen, canned = generations
    canned.append(furniture("x"))

    await generate("m", "p", max_tokens=4, temperature=0.0)

    assert "--seed" not in seen[0]
    assert seen[0][-3:] == ("--temp", "0.0", "--ignore-chat-template")


def test_strip_generation_recovers_multiline_text_between_the_fences() -> None:
    assert strip_generation(furniture("line one\nline two")) == "line one\nline two"


def test_strip_generation_returns_empty_when_nothing_was_generated() -> None:
    stdout = f"{GENERATION_FENCE}\n\n{GENERATION_FENCE}\nNo text generated for this prompt\n"

    assert strip_generation(stdout) == ""
