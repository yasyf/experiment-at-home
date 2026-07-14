from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from athome.train import data
from athome.train.data import (
    DpoExample,
    Message,
    SftExample,
    from_local_jsonl,
    normalize,
    render_mlx_jsonl,
    render_tinker_dpo,
    render_tinker_sft,
    render_trl,
)
from athome.train.spec import HfDatasetRef, LocalJsonlRef, MlxModelId

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

MLX_ID = MlxModelId("mlx-community/Qwen3-8B-4bit")
PROMPT: tuple[Message, ...] = ({"role": "user", "content": "hi"},)
CHOSEN: tuple[Message, ...] = ({"role": "assistant", "content": "yo"},)
REJECTED: tuple[Message, ...] = ({"role": "assistant", "content": "nope!"},)


@dataclass(frozen=True, slots=True)
class FakeTensor:
    data: list[float] | list[int]
    dtype: str
    shape: list[int]


@dataclass(frozen=True, slots=True)
class FakeModelInput:
    ids: list[int]

    @staticmethod
    def from_ints(ids: list[int]) -> FakeModelInput:
        return FakeModelInput(ids)


@dataclass(frozen=True, slots=True)
class FakeDatum:
    model_input: FakeModelInput
    loss_fn_inputs: dict[str, FakeTensor]


@dataclass(frozen=True, slots=True)
class FakeDataset:
    rows: list[dict[str, object]]

    @staticmethod
    def from_list(rows: list[dict[str, object]]) -> FakeDataset:
        return FakeDataset(rows)


@dataclass(slots=True)
class FakeTokenizer:
    """A char-level chat tokenizer: the templated prompt is always a prefix of the full text."""

    calls: list[bool] = field(default_factory=list)

    def apply_chat_template(
        self, messages: list[Message], *, tokenize: bool, add_generation_prompt: bool, enable_thinking: bool
    ) -> str:
        assert tokenize is False
        assert enable_thinking is False
        self.calls.append(add_generation_prompt)
        rendered = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        return f"{rendered}<assistant>" if add_generation_prompt else rendered

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(char) for char in text]


@pytest.fixture(autouse=True)
def fake_tinker(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("tinker")
    module.Datum = FakeDatum
    module.ModelInput = FakeModelInput
    module.TensorData = FakeTensor
    monkeypatch.setitem(sys.modules, "tinker", module)


@pytest.fixture(autouse=True)
def fake_datasets(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = ModuleType("datasets")
    module.Dataset = FakeDataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    return module


@pytest.fixture(autouse=True)
def fake_tokenizer(monkeypatch: pytest.MonkeyPatch) -> FakeTokenizer:
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(data, "tokenizer", lambda mlx_id: tokenizer)
    return tokenizer


def ids(text: str) -> list[int]:
    return [ord(char) for char in text]


def test_render_tinker_sft_weights_only_the_completion(fake_tokenizer: FakeTokenizer) -> None:
    datum = render_tinker_sft(SftExample(prompt=PROMPT, completion=CHOSEN, id="a"), MLX_ID)

    full = ids("<user>hi<assistant>yo")
    assert datum.model_input == FakeModelInput(full[:-1])
    assert datum.loss_fn_inputs["target_tokens"] == FakeTensor(full[1:], "int64", [len(full) - 1])
    weights = datum.loss_fn_inputs["weights"]
    assert weights.dtype == "float32"
    assert weights.shape == [len(full) - 1]
    assert weights.data == [0.0] * (len(ids("<user>hi<assistant>")) - 1) + [1.0, 1.0]
    assert fake_tokenizer.calls == [False, True]


def test_render_tinker_sft_scales_the_mask_by_the_example_weight() -> None:
    datum = render_tinker_sft(SftExample(prompt=PROMPT, completion=CHOSEN, id="a", weight=0.5), MLX_ID)
    assert sum(datum.loss_fn_inputs["weights"].data) == 1.0


def test_render_tinker_dpo_masks_both_continuations_against_the_same_prompt() -> None:
    preference = render_tinker_dpo(DpoExample(prompt=PROMPT, chosen=CHOSEN, rejected=REJECTED, id="a"), MLX_ID)

    assert preference.chosen.model_input == FakeModelInput(ids("<user>hi<assistant>yo")[:-1])
    assert preference.rejected.model_input == FakeModelInput(ids("<user>hi<assistant>nope!")[:-1])
    assert sum(preference.chosen.loss_fn_inputs["weights"].data) == len("yo")
    assert sum(preference.rejected.loss_fn_inputs["weights"].data) == len("nope!")
    assert set(preference.chosen.loss_fn_inputs) == {"weights", "target_tokens"}


def sft_examples(count: int) -> list[SftExample]:
    return [
        SftExample(
            prompt=({"role": "user", "content": f"q{index}"},),
            completion=({"role": "assistant", "content": f"a{index}"},),
            id=str(index),
        )
        for index in range(count)
    ]


def read_jsonl(path: Path) -> list[dict[str, list[Message]]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_render_mlx_jsonl_carves_a_seeded_validation_split(tmp_path: Path) -> None:
    out = render_mlx_jsonl(sft_examples(10), tmp_path / "data", val_fraction=0.2)

    train, valid = read_jsonl(out / "train.jsonl"), read_jsonl(out / "valid.jsonl")
    assert len(train) == 8
    assert len(valid) == 2
    assert train[0]["messages"][0]["role"] == "user"
    assert train[0]["messages"][1]["role"] == "assistant"
    contents = {row["messages"][0]["content"] for row in train + valid}
    assert contents == {f"q{index}" for index in range(10)}


def test_render_mlx_jsonl_is_deterministic_for_a_seed(tmp_path: Path) -> None:
    first = render_mlx_jsonl(sft_examples(10), tmp_path / "a", seed=7)
    second = render_mlx_jsonl(sft_examples(10), tmp_path / "b", seed=7)
    third = render_mlx_jsonl(sft_examples(10), tmp_path / "c", seed=8)

    assert (first / "valid.jsonl").read_text() == (second / "valid.jsonl").read_text()
    assert (first / "valid.jsonl").read_text() != (third / "valid.jsonl").read_text()


def test_render_trl_sft_emits_prompt_and_completion_columns() -> None:
    dataset = render_trl(sft_examples(2), method="sft")
    assert dataset.rows == [
        {"prompt": [{"role": "user", "content": "q0"}], "completion": [{"role": "assistant", "content": "a0"}]},
        {"prompt": [{"role": "user", "content": "q1"}], "completion": [{"role": "assistant", "content": "a1"}]},
    ]


def test_render_trl_dpo_emits_chosen_and_rejected_columns() -> None:
    dataset = render_trl([DpoExample(prompt=PROMPT, chosen=CHOSEN, rejected=REJECTED, id="a")], method="dpo")
    assert dataset.rows == [{"prompt": list(PROMPT), "chosen": list(CHOSEN), "rejected": list(REJECTED)}]


def write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> Path:
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows))
    return path


async def test_normalize_local_sft_splits_the_last_turn_off_as_the_completion(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "corpus.jsonl",
        [
            {
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ]
            },
        ],
    )

    assert await normalize(LocalJsonlRef(path=path), method="sft") == [
        SftExample(
            prompt=({"role": "system", "content": "s"}, {"role": "user", "content": "q"}),
            completion=({"role": "assistant", "content": "a"},),
            id="corpus:0",
        )
    ]


async def test_normalize_local_dpo_reads_the_preference_columns(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "pairs.jsonl",
        [{"prompt": list(PROMPT), "chosen": list(CHOSEN), "rejected": list(REJECTED)}],
    )

    assert await normalize(LocalJsonlRef(path=path), method="dpo") == [
        DpoExample(prompt=PROMPT, chosen=CHOSEN, rejected=REJECTED, id="pairs:0")
    ]


def test_from_local_jsonl_reads_every_file_in_a_directory(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "train.jsonl", [{"messages": [*PROMPT, *CHOSEN]}])
    write_jsonl(tmp_path / "valid.jsonl", [{"messages": [*PROMPT, *CHOSEN]}])

    assert [example.id for example in from_local_jsonl(LocalJsonlRef(path=tmp_path), method="sft")] == [
        "train:0",
        "valid:0",
    ]


async def test_normalize_hf_loads_the_requested_config_and_split(
    monkeypatch: pytest.MonkeyPatch, fake_datasets: ModuleType
) -> None:
    seen: list[tuple[str, str, str]] = []

    def load_dataset(repo: str, config: str, *, split: str) -> list[dict[str, object]]:
        seen.append((repo, config, split))
        return [{"prompt": list(PROMPT), "completion": list(CHOSEN), "id": "row-1"}]

    fake_datasets.load_dataset = load_dataset

    examples = await normalize(HfDatasetRef(repo="yasyf/cc-steer-traces", config="sft"), method="sft")

    assert seen == [("yasyf/cc-steer-traces", "sft", "train")]
    assert examples == [SftExample(prompt=PROMPT, completion=CHOSEN, id="row-1")]


async def test_normalize_hf_dpo_rows_become_preference_examples(fake_datasets: ModuleType) -> None:
    fake_datasets.load_dataset = lambda repo, config, *, split: [
        {"prompt": list(PROMPT), "chosen": list(CHOSEN), "rejected": list(REJECTED), "id": "row-2"}
    ]

    assert await normalize(HfDatasetRef(repo="yasyf/cc-steer-traces", config="dpo", split="test"), method="dpo") == [
        DpoExample(prompt=PROMPT, chosen=CHOSEN, rejected=REJECTED, id="row-2")
    ]
