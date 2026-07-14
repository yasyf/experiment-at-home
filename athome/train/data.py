from __future__ import annotations

import functools
import json
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict, overload

from anyio import to_thread

from athome.train.spec import SEED, HfDatasetRef, LocalJsonlRef

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    import tinker
    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase

    from athome.train.spec import DatasetSource, Method, MlxModelId


class Message(TypedDict):
    role: str
    content: str


class SftRow(TypedDict):
    prompt: list[Message]
    completion: list[Message]
    id: str


class DpoRow(TypedDict):
    prompt: list[Message]
    chosen: list[Message]
    rejected: list[Message]
    id: str


@dataclass(frozen=True, slots=True)
class SftExample:
    """A prompt and the completion the model should learn to produce."""

    prompt: tuple[Message, ...]
    completion: tuple[Message, ...]
    id: str
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class DpoExample:
    """A prompt and the two completions a preference loss ranks against each other."""

    prompt: tuple[Message, ...]
    chosen: tuple[Message, ...]
    rejected: tuple[Message, ...]
    id: str
    weight: float = 1.0


type TrainExample = SftExample | DpoExample


@dataclass(frozen=True, slots=True)
class TinkerPreference:
    """The chosen/rejected ``Datum`` pair a Tinker preference loss consumes."""

    chosen: tinker.Datum
    rejected: tinker.Datum


@functools.lru_cache
def tokenizer(mlx_id: MlxModelId) -> PreTrainedTokenizerBase:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(mlx_id)


def chat_ids(messages: Sequence[Message], mlx_id: MlxModelId, *, add_generation_prompt: bool) -> list[int]:
    tok = tokenizer(mlx_id)
    return tok.encode(
        tok.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
        ),
        add_special_tokens=False,
    )


def boundary(full: list[int], prompt: list[int]) -> int:
    return next(
        (i for i, (left, right) in enumerate(zip(full, prompt, strict=False)) if left != right),
        min(len(full), len(prompt)),
    )


def masked_datum(
    prompt: Sequence[Message], completion: Sequence[Message], mlx_id: MlxModelId, *, weight: float
) -> tinker.Datum:
    import tinker

    full = chat_ids((*prompt, *completion), mlx_id, add_generation_prompt=False)
    cut = boundary(full, chat_ids(prompt, mlx_id, add_generation_prompt=True))
    mask = [0.0] * cut + [weight] * (len(full) - cut)
    targets, weights = full[1:], mask[1:]
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(full[:-1]),
        loss_fn_inputs={
            "weights": tinker.TensorData(data=weights, dtype="float32", shape=[len(weights)]),
            "target_tokens": tinker.TensorData(data=targets, dtype="int64", shape=[len(targets)]),
        },
    )


def sft_example(row: SftRow) -> SftExample:
    return SftExample(prompt=tuple(row["prompt"]), completion=tuple(row["completion"]), id=row["id"])


def dpo_example(row: DpoRow) -> DpoExample:
    return DpoExample(
        prompt=tuple(row["prompt"]),
        chosen=tuple(row["chosen"]),
        rejected=tuple(row["rejected"]),
        id=row["id"],
    )


def jsonl_rows(path: Path) -> Iterable[tuple[str, dict[str, list[Message]]]]:
    for file in sorted(path.glob("*.jsonl")) if path.is_dir() else [path]:
        for index, line in enumerate(file.read_text().splitlines()):
            if line:
                yield f"{file.stem}:{index}", json.loads(line)


@overload
async def normalize(source: DatasetSource, *, method: Literal["sft"]) -> list[SftExample]: ...
@overload
async def normalize(source: DatasetSource, *, method: Literal["dpo"]) -> list[DpoExample]: ...
async def normalize(source: DatasetSource, *, method: Method) -> list[TrainExample]:
    """Read ``source`` and normalize every row into a :data:`TrainExample`.

    Args:
        source: An HF export config, or a local jsonl file or directory.
        method: The method being trained; picks the row shape each source is read as.

    Returns:
        :class:`SftExample` s for ``sft``, :class:`DpoExample` s for ``dpo``.
    """
    match source:
        case HfDatasetRef():
            return await from_hf(source, method=method)
        case LocalJsonlRef():
            return from_local_jsonl(source, method=method)


async def from_hf(ref: HfDatasetRef, *, method: Method) -> list[TrainExample]:
    """Load one config of a cc-steer HF export; its ``sft``/``dpo`` rows are already chat messages."""
    from datasets import load_dataset

    rows: Iterable[SftRow] | Iterable[DpoRow] = await to_thread.run_sync(
        functools.partial(load_dataset, ref.repo, ref.config, split=ref.split)
    )
    match method:
        case "sft":
            return [sft_example(row) for row in rows]
        case "dpo":
            return [dpo_example(row) for row in rows]


def from_local_jsonl(ref: LocalJsonlRef, *, method: Method) -> list[TrainExample]:
    """Read local jsonl rows: mlx-lm chat rows for ``sft``, ``prompt``/``chosen``/``rejected`` for ``dpo``.

    An ``sft`` row's last message is the completion and every message before it is the prompt.
    """
    match method:
        case "sft":
            return [
                SftExample(prompt=tuple(row["messages"][:-1]), completion=(row["messages"][-1],), id=row_id)
                for row_id, row in jsonl_rows(ref.path)
            ]
        case "dpo":
            return [
                DpoExample(
                    prompt=tuple(row["prompt"]),
                    chosen=tuple(row["chosen"]),
                    rejected=tuple(row["rejected"]),
                    id=row_id,
                )
                for row_id, row in jsonl_rows(ref.path)
            ]


def render_tinker_sft(example: SftExample, mlx_id: MlxModelId) -> tinker.Datum:
    """One prompt-masked SFT ``Datum``: the example's weight on the completion tokens, zero on the prompt."""
    return masked_datum(example.prompt, example.completion, mlx_id, weight=example.weight)


def render_tinker_dpo(example: DpoExample, mlx_id: MlxModelId) -> TinkerPreference:
    """The chosen and rejected prompt-masked ``Datum`` s a preference loss scores against each other."""
    return TinkerPreference(
        chosen=masked_datum(example.prompt, example.chosen, mlx_id, weight=example.weight),
        rejected=masked_datum(example.prompt, example.rejected, mlx_id, weight=example.weight),
    )


def render_mlx_jsonl(
    examples: Sequence[SftExample], out_dir: Path, *, val_fraction: float = 0.1, seed: int = SEED
) -> Path:
    """Write ``out_dir/{train,valid}.jsonl`` in mlx-lm chat shape and return ``out_dir``.

    The validation split is a seeded carve of ``val_fraction`` of the examples. SFT only —
    mlx-lm has no DPO trainer.

    Args:
        examples: The SFT examples to write.
        out_dir: The directory the two jsonl files land in.
        val_fraction: The fraction of examples carved into ``valid.jsonl``.
        seed: Seeds the carve, so the same corpus always splits the same way.

    Returns:
        ``out_dir``, the ``--data`` directory mlx-lm's LoRA trainer reads.
    """
    rows = [{"messages": [*example.prompt, *example.completion]} for example in examples]
    shuffled = random.Random(seed).sample(rows, len(rows))
    cut = round(len(shuffled) * val_fraction)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("valid", shuffled[:cut]), ("train", shuffled[cut:])):
        (out_dir / f"{name}.jsonl").write_text("".join(f"{json.dumps(row)}\n" for row in split))
    return out_dir


@overload
def render_trl(examples: Sequence[SftExample], *, method: Literal["sft"]) -> Dataset: ...
@overload
def render_trl(examples: Sequence[DpoExample], *, method: Literal["dpo"]) -> Dataset: ...
def render_trl(examples: Sequence[TrainExample], *, method: Method) -> Dataset:
    """Build the TRL dataset for ``method``: prompt/completion columns for SFT, prompt/chosen/rejected for DPO."""
    from datasets import Dataset

    match method:
        case "sft":
            rows = [{"prompt": list(e.prompt), "completion": list(e.completion)} for e in examples]
        case "dpo":
            rows = [
                {"prompt": list(e.prompt), "chosen": list(e.chosen), "rejected": list(e.rejected)} for e in examples
            ]
    return Dataset.from_list(rows)
