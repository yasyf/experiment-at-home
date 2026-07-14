from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from athome.config import load
from athome.progress import RunSink
from athome.train import local, sidecar
from athome.train.local import LocalBackend, UnsupportedLoraShape
from athome.train.spec import BASE_MODELS, STD_MODULES, Hyperparams, LocalJsonlRef, LoraSpec, TrainSpec

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

BASE = BASE_MODELS["qwen3-8b"]
VERSION = "0.31.3"
VAL_FRACTION = 0.25
EXAMPLES = 8


@pytest.fixture(autouse=True)
def settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATHOME_TRAIN_MLX_LM_VERSION", VERSION)
    monkeypatch.setenv("ATHOME_TRAIN_LOCAL_VAL_FRACTION", str(VAL_FRACTION))
    load.cache_clear()


@pytest.fixture
def commands(monkeypatch: pytest.MonkeyPatch) -> list[Sequence[str]]:
    seen: list[Sequence[str]] = []

    async def run_process(command: Sequence[str]) -> None:
        seen.append(command)

    monkeypatch.setattr(sidecar, "run_process", run_process)
    return seen


@pytest.fixture
def corpus(tmp_path: Path) -> LocalJsonlRef:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {"messages": [{"role": "user", "content": f"ask {i}"}, {"role": "assistant", "content": f"say {i}"}]}
            )
            + "\n"
            for i in range(EXAMPLES)
        )
    )
    return LocalJsonlRef(path=path)


@pytest.fixture
def sink(tmp_path: Path) -> RunSink:
    return RunSink.open(tmp_path / "journal.jsonl")


def spec(corpus: LocalJsonlRef, **overrides: object) -> TrainSpec:
    return TrainSpec(
        name="watcher",
        base=BASE,
        dataset=corpus,
        hyperparams=Hyperparams(steps=120, batch_size=2, learning_rate=0.0002, max_seq_len=1024, seed=7),
        **overrides,
    )


def run_dir(tmp_path: Path) -> Path:
    """The private work directory ``athome.train.run`` mints for one run and hands the backend."""
    return tmp_path / "runs" / "watcher" / "20260714-120000-0f1e2d3c"


def test_local_trains_sft_and_never_dpo() -> None:
    assert LocalBackend.supports("sft") is True
    assert LocalBackend.supports("dpo") is False


@pytest.mark.parametrize(
    ("system", "machine", "uvx", "expected"),
    [
        ("darwin", "arm64", "/opt/homebrew/bin/uvx", True),
        ("darwin", "arm64", None, False),
        ("darwin", "x86_64", "/usr/local/bin/uvx", False),
        ("linux", "aarch64", "/usr/bin/uvx", False),
    ],
    ids=["apple-silicon-with-uvx", "no-uvx", "intel-mac", "linux"],
)
def test_available_only_on_an_arm64_mac_that_can_launch_the_sidecar(
    monkeypatch: pytest.MonkeyPatch, system: str, machine: str, uvx: str | None, expected: bool
) -> None:
    monkeypatch.setattr(local.sys, "platform", system)
    monkeypatch.setattr(local.platform, "machine", lambda: machine)
    monkeypatch.setattr(local.shutil, "which", lambda name: uvx if name == "uvx" else None)

    assert LocalBackend.available() is expected


async def test_train_writes_the_seeded_mlx_chat_split(
    commands: list[Sequence[str]], corpus: LocalJsonlRef, sink: RunSink, tmp_path: Path
) -> None:
    await LocalBackend.from_settings().train(spec(corpus), sink=sink, work_dir=run_dir(tmp_path))

    data = run_dir(tmp_path) / "data"
    splits = {
        name: [json.loads(line) for line in (data / f"{name}.jsonl").read_text().splitlines()]
        for name in ("train", "valid")
    }
    assert [len(splits["train"]), len(splits["valid"])] == [6, 2]
    assert sorted(row["messages"][0]["content"] for rows in splits.values() for row in rows) == [
        f"ask {i}" for i in range(EXAMPLES)
    ]
    assert splits["train"][0] == {
        "messages": [{"role": "user", "content": "ask 3"}, {"role": "assistant", "content": "say 3"}]
    }


async def test_train_runs_the_pinned_mlx_lm_lora_sidecar(
    commands: list[Sequence[str]], corpus: LocalJsonlRef, sink: RunSink, tmp_path: Path
) -> None:
    await LocalBackend.from_settings().train(spec(corpus), sink=sink, work_dir=run_dir(tmp_path))

    assert commands[0] == (
        "uvx",
        "--from",
        f"mlx-lm=={VERSION}",
        "python",
        "-m",
        "mlx_lm",
        "lora",
        "--train",
        "--model",
        "mlx-community/Qwen3-8B-4bit",
        "--data",
        str(run_dir(tmp_path) / "data"),
        "--adapter-path",
        str(run_dir(tmp_path) / "adapter"),
        "--config",
        str(run_dir(tmp_path) / "lora.yaml"),
        "--fine-tune-type",
        "lora",
        "--optimizer",
        "adamw",
        "--mask-prompt",
        "--num-layers",
        "36",
        "--iters",
        "120",
        "--batch-size",
        "2",
        "--learning-rate",
        "0.0002",
        "--max-seq-length",
        "1024",
        "--seed",
        "7",
        "--grad-checkpoint",
    )


async def test_the_lora_shape_rides_the_config_file_mlx_lm_has_no_flags_for(
    commands: list[Sequence[str]], corpus: LocalJsonlRef, sink: RunSink, tmp_path: Path
) -> None:
    await LocalBackend.from_settings().train(
        spec(corpus, lora=LoraSpec(rank=8, alpha=32, dropout=0.05)), sink=sink, work_dir=run_dir(tmp_path)
    )

    assert json.loads((run_dir(tmp_path) / "lora.yaml").read_text()) == {
        "lora_parameters": {"rank": 8, "scale": 4.0, "dropout": 0.05, "keys": list(STD_MODULES)}
    }


@pytest.mark.parametrize(
    ("lora", "expected"),
    [
        (LoraSpec(train_attn=False), [key for key in STD_MODULES if key.startswith("mlp.")]),
        (LoraSpec(train_mlp=False), [key for key in STD_MODULES if key.startswith("self_attn.")]),
        (LoraSpec(target_modules=("self_attn.q_proj", "mlp.up_proj")), ["self_attn.q_proj", "mlp.up_proj"]),
    ],
    ids=["attn-off-keeps-mlp", "mlp-off-keeps-attn", "both-on-keeps-target-modules"],
)
async def test_the_lora_toggles_filter_the_keys_mlx_lm_wraps(
    commands: list[Sequence[str]],
    corpus: LocalJsonlRef,
    sink: RunSink,
    tmp_path: Path,
    lora: LoraSpec,
    expected: list[str],
) -> None:
    await LocalBackend.from_settings().train(spec(corpus, lora=lora), sink=sink, work_dir=run_dir(tmp_path))

    assert json.loads((run_dir(tmp_path) / "lora.yaml").read_text())["lora_parameters"]["keys"] == expected


@pytest.mark.parametrize(
    ("lora", "match"),
    [
        (LoraSpec(train_unembed=True), "unembedding"),
        (LoraSpec(train_attn=False, train_mlp=False), "nothing to train"),
    ],
    ids=["unembed-is-inexpressible", "no-trainable-module"],
)
async def test_a_lora_shape_mlx_lm_cannot_express_raises_before_anything_runs(
    commands: list[Sequence[str]], corpus: LocalJsonlRef, sink: RunSink, tmp_path: Path, lora: LoraSpec, match: str
) -> None:
    with pytest.raises(UnsupportedLoraShape, match=match):
        await LocalBackend.from_settings().train(spec(corpus, lora=lora), sink=sink, work_dir=run_dir(tmp_path))

    assert commands == []
    assert not sink.path.exists()


async def test_grad_checkpoint_off_drops_the_flag(
    commands: list[Sequence[str]],
    corpus: LocalJsonlRef,
    sink: RunSink,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATHOME_TRAIN_LOCAL_GRAD_CHECKPOINT", "false")
    load.cache_clear()

    await LocalBackend.from_settings().train(spec(corpus), sink=sink, work_dir=run_dir(tmp_path))

    assert "--grad-checkpoint" not in commands[0]


async def test_the_fused_model_is_the_checkpoints_serve_path(
    commands: list[Sequence[str]], corpus: LocalJsonlRef, sink: RunSink, tmp_path: Path
) -> None:
    checkpoint = await LocalBackend.from_settings().train(spec(corpus), sink=sink, work_dir=run_dir(tmp_path))

    assert commands[1] == (
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
        str(run_dir(tmp_path) / "adapter"),
        "--save-path",
        str(run_dir(tmp_path) / "fused"),
    )
    assert checkpoint.mlx_path == run_dir(tmp_path) / "fused"
    assert checkpoint.adapter_dir == run_dir(tmp_path) / "adapter"
    assert (checkpoint.base, checkpoint.backend, checkpoint.method, checkpoint.step) == (BASE, "local", "sft", 120)


async def test_local_training_is_free(
    commands: list[Sequence[str]], corpus: LocalJsonlRef, sink: RunSink, tmp_path: Path
) -> None:
    trained = await LocalBackend.from_settings().train(spec(corpus), sink=sink, work_dir=run_dir(tmp_path))

    assert trained.train_cost_usd == 0.0


async def test_train_journals_every_stage(
    commands: list[Sequence[str]], corpus: LocalJsonlRef, sink: RunSink, tmp_path: Path
) -> None:
    await LocalBackend.from_settings().train(spec(corpus), sink=sink, work_dir=run_dir(tmp_path))

    records = [json.loads(line) for line in sink.path.read_text().splitlines()]
    assert [record["stage"] for record in records] == ["data", "train", "fused"]
    assert records[0] == {"stage": "data", "examples": EXAMPLES, "data_dir": str(run_dir(tmp_path) / "data")}
    assert records[1]["command"] == list(commands[0])
    assert records[2] == {"stage": "fused", "mlx_path": str(run_dir(tmp_path) / "fused")}


@pytest.mark.live
async def test_the_config_file_parses_as_the_yaml_mlx_lm_loads(tmp_path: Path) -> None:
    import anyio

    config = local.lora_config(LoraSpec(rank=8, alpha=32), tmp_path / "lora.yaml")
    probe = f"import json, sys, yaml;print(json.dumps(yaml.load(open({str(config)!r}), yaml.SafeLoader)))"
    result = await anyio.run_process([*sidecar.uvx("python", "-c", probe)])

    assert json.loads(result.stdout) == {
        "lora_parameters": {"rank": 8, "scale": 4.0, "dropout": 0.0, "keys": list(STD_MODULES)}
    }
