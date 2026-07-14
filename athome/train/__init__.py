"""LoRA fine-tuning over tinker, local mlx-lm, or modal, converging on one servable MLX artifact."""

from __future__ import annotations

from athome.train.backend import NoBackendAvailable, TrainBackend, backends, select
from athome.train.data import (
    DpoExample,
    Message,
    SftExample,
    TinkerPreference,
    TrainExample,
    normalize,
    render_mlx_jsonl,
    render_tinker_dpo,
    render_tinker_sft,
    render_trl,
)
from athome.train.spec import (
    BASE_MODELS,
    STD_MODULES,
    BackendName,
    BaseModelSpec,
    Checkpoint,
    DatasetSource,
    HfDatasetRef,
    Hyperparams,
    LocalJsonlRef,
    LocalTrainSettings,
    LoraSpec,
    Method,
    ModalTrainSettings,
    TinkerSettings,
    TrainResult,
    TrainSettings,
    TrainSpec,
)
