from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from athome import config
from athome.config import load
from athome.train.spec import (
    BASE_MODELS,
    STD_MODULES,
    BaseModelSpec,
    Checkpoint,
    Hyperparams,
    LocalJsonlRef,
    LocalTrainSettings,
    LoraSpec,
    ModalTrainSettings,
    TinkerSettings,
    TrainSettings,
    TrainSpec,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def tinker_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("TINKER_API_KEY", "sk-tinker-test")
    load.cache_clear()
    yield
    load.cache_clear()


def spec(**overrides: object) -> TrainSpec:
    return dataclasses.replace(
        TrainSpec(
            name="watcher",
            base=BASE_MODELS["qwen3-8b"],
            dataset=LocalJsonlRef(path=Path("corpus.jsonl")),
            hyperparams=Hyperparams(steps=100),
        ),
        **overrides,
    )


def test_base_models_catalog() -> None:
    assert BASE_MODELS["qwen3-8b"] == BaseModelSpec(
        mlx="mlx-community/Qwen3-8B-4bit",
        hf="Qwen/Qwen3-8B",
        tinker="Qwen/Qwen3-8B",
        num_layers=36,
        serves_locally=True,
    )
    assert BASE_MODELS["qwen3.5-4b"].serves_locally is False


def test_spec_defaults() -> None:
    assert spec().method == "sft"
    assert spec().backend is None
    assert spec().max_usd is None
    assert spec().lora == LoraSpec(rank=16, alpha=32, target_modules=STD_MODULES)
    assert spec().hyperparams == Hyperparams(steps=100, batch_size=4, learning_rate=1e-4, max_seq_len=4096, seed=1729)


def test_spec_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec().method = "dpo"  # ty: ignore[invalid-assignment]


def test_checkpoint_carries_the_fused_artifact_and_its_provenance() -> None:
    checkpoint = Checkpoint(
        base=BASE_MODELS["qwen3-8b"],
        backend="tinker",
        method="sft",
        step=100,
        mlx_path=Path("/runs/watcher/fused"),
        adapter_dir=Path("/runs/watcher/adapter"),
        train_cost_usd=1.25,
    )
    assert checkpoint.mlx_path == Path("/runs/watcher/fused")
    assert checkpoint.adapter_dir == Path("/runs/watcher/adapter")


def test_train_settings_defaults_expand_user_paths() -> None:
    settings = load(TrainSettings)
    assert settings.backend is None
    assert settings.mlx_lm_version == "0.31.3"
    assert settings.registry_root == Path.home() / ".athome/train/registry"
    assert settings.work_root == Path.home() / ".athome/train/runs"


def test_train_settings_read_the_train_section_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATHOME_TRAIN_BACKEND", "modal")
    monkeypatch.setenv("ATHOME_TRAIN_MLX_LM_VERSION", "0.28.3")
    load.cache_clear()
    assert load(TrainSettings).backend == "modal"
    assert load(TrainSettings).mlx_lm_version == "0.28.3"


def test_backend_name_is_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATHOME_TRAIN_BACKEND", "colab")
    load.cache_clear()
    with pytest.raises(ValidationError):
        load(TrainSettings)


def test_tinker_settings_source_the_secret_from_the_canonical_env_var() -> None:
    settings = load(TinkerSettings)
    assert settings.api_key.get_secret_value() == "sk-tinker-test"
    assert "sk-tinker-test" not in repr(settings)
    assert settings.spend_cap_usd == 60.0
    assert settings.price_per_mtok["Qwen/Qwen3-8B"] == 0.40


def test_tinker_settings_require_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TINKER_API_KEY")
    load.cache_clear()
    with pytest.raises(ValidationError):
        load(TinkerSettings)


def test_nested_settings_bind_their_own_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATHOME_TRAIN_LOCAL_VAL_FRACTION", "0.25")
    monkeypatch.setenv("ATHOME_TRAIN_MODAL_GPU", "A100")
    monkeypatch.setenv("ATHOME_TRAIN_TINKER_SPEND_CAP_USD", "12.5")
    load.cache_clear()
    assert load(LocalTrainSettings).val_fraction == 0.25
    assert load(ModalTrainSettings).gpu == "A100"
    assert load(TinkerSettings).spend_cap_usd == 12.5


def test_nested_settings_read_their_toml_subsection() -> None:
    config.CONFIG_FILE.write_text(
        '[train]\nmlx_lm_version = "0.29.0"\n[train.local]\ngrad_checkpoint = false\n[train.modal]\ngpu = "H200"\n'
    )
    load.cache_clear()
    assert load(TrainSettings).mlx_lm_version == "0.29.0"
    assert load(LocalTrainSettings).grad_checkpoint is False
    assert load(ModalTrainSettings).gpu == "H200"
