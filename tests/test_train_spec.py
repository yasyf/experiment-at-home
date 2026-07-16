from __future__ import annotations

import dataclasses
import re
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
    TinkerPrice,
    TinkerSettings,
    TrainSettings,
    TrainSpec,
    UnsupportedLoraShape,
    lora_keys,
    spend_cap,
    std_lora_keys,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

COMMIT = re.compile(r"^[0-9a-f]{40}$")
ATTN = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
MLP = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


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
        hf_revision="b968826d9c46dd6066d109eabc6255188de91218",
        mlx_revision="545dc4251c05440727734bcd94334791f6ab0192",
        tinker="Qwen/Qwen3-8B",
        num_layers=36,
        serves_locally=True,
    )
    assert BASE_MODELS["qwen3.5-4b"].serves_locally is False


def test_every_base_pins_both_repos_to_a_commit() -> None:
    """#7 — a floating repo id lets two runs train different weights under one fingerprint."""
    revisions = [(base.hf_revision, base.mlx_revision) for base in BASE_MODELS.values()]

    assert all(COMMIT.match(revision) for pair in revisions for revision in pair)
    assert len({revision for pair in revisions for revision in pair}) == 2 * len(BASE_MODELS)


@pytest.mark.parametrize(
    ("max_usd", "expected"),
    [
        pytest.param(None, 60.0, id="unset-takes-the-configured-cap"),
        pytest.param(0.0, 0.0, id="zero-is-spend-nothing-not-unset"),
        pytest.param(2.5, 2.5, id="a-set-cap-binds"),
    ],
)
def test_spend_cap_treats_only_none_as_unset(max_usd: float | None, expected: float) -> None:
    """#4 — ``spec.max_usd or default`` let a 0.0 cap fall through to the configured $60."""
    assert spend_cap(spec(max_usd=max_usd), 60.0) == expected


@pytest.mark.parametrize(
    ("lora", "expected"),
    [
        pytest.param(LoraSpec(), STD_MODULES, id="both-toggles-on-keeps-every-standard-module"),
        pytest.param(LoraSpec(train_mlp=False), ATTN, id="mlp-off-drops-the-mlp-modules"),
        pytest.param(LoraSpec(train_attn=False), MLP, id="attn-off-drops-the-attention-modules"),
        pytest.param(
            LoraSpec(target_modules=("self_attn.q_proj", "mlp.up_proj"), train_mlp=False),
            ("self_attn.q_proj",),
            id="the-toggles-filter-a-narrowed-target-list",
        ),
    ],
)
def test_lora_keys_filters_the_target_modules_by_the_toggles(lora: LoraSpec, expected: tuple[str, ...]) -> None:
    """#5 — the toggles select which modules an adapter wraps, on every backend."""
    assert lora_keys(lora) == expected


@pytest.mark.parametrize(
    ("lora", "match"),
    [
        pytest.param(LoraSpec(train_unembed=True), "unembedding", id="unembed-cannot-be-fused"),
        pytest.param(LoraSpec(train_mlp=False, train_attn=False), "nothing to train", id="both-toggles-off"),
        pytest.param(
            LoraSpec(target_modules=("mlp.up_proj",), train_mlp=False), "nothing to train", id="toggles-empty-the-list"
        ),
    ],
)
def test_an_inexpressible_lora_shape_raises(lora: LoraSpec, match: str) -> None:
    """#6 — a shape no backend can fuse must raise, not silently train weights that get dropped."""
    with pytest.raises(UnsupportedLoraShape, match=match):
        lora_keys(lora)


def test_std_lora_keys_ignores_the_requested_target_list() -> None:
    """What a toggles-only backend (tinker) actually trains, whatever the spec asked to target."""
    assert std_lora_keys(LoraSpec(target_modules=("self_attn.q_proj",))) == STD_MODULES
    assert std_lora_keys(LoraSpec(target_modules=("self_attn.q_proj",), train_mlp=False)) == ATTN


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
    assert settings.baseline_root == Path.home() / ".athome/train/baselines.db"
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
    assert settings.price_per_mtok["Qwen/Qwen3-8B"] == TinkerPrice(prefill=0.13, sample=0.40, train=0.40)
    assert settings.price_per_mtok["Qwen/Qwen3.5-9B"] == TinkerPrice(prefill=0.44, sample=1.33, train=1.33)


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
