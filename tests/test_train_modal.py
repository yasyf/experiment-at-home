from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from athome.config import load
from athome.llm.spend import SpendExceeded
from athome.modal import ParityMismatch, fingerprint_for
from athome.progress import RunSink
from athome.train import modal as train_modal
from athome.train.backend import TrainBackend
from athome.train.modal import (
    ModalTrainBackend,
    RemoteResult,
    budget_seconds,
    projected_usd,
    service_spec,
)
from athome.train.spec import (
    BASE_MODELS,
    STD_MODULES,
    Hyperparams,
    LocalJsonlRef,
    LoraSpec,
    ModalTrainSettings,
    TrainSpec,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from athome.train.spec import Method

BASE = BASE_MODELS["qwen3-8b"]
MATCHING_FINGERPRINT: dict[str, object] = {
    "param:trl": "0.21.0",
    "param:peft": "0.14.0",
    "param:torch": "2.6.0",
    "param:transformers": "4.48.0",
    "param:lora_rank": 16,
    "param:lora_alpha": 32,
    "param:lora_dropout": 0.0,
    "param:lora_target_modules": sorted(STD_MODULES),
}
TRAINED = RemoteResult(repo="athome-train/pilot", revision="c0ffee", step=4, seconds=1800.0)
SFT_ROWS = [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}]
DPO_ROWS = [
    {
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": [{"role": "assistant", "content": "yes"}],
        "rejected": [{"role": "assistant", "content": "no"}],
    }
]


@dataclass(frozen=True, slots=True)
class FakeDataset:
    rows: list[dict[str, object]]

    @staticmethod
    def from_list(rows: list[dict[str, object]]) -> FakeDataset:
        return FakeDataset(rows)


@dataclass(slots=True)
class Remote:
    """What the fake Modal side returns, and what it was asked to do."""

    fingerprint: dict[str, object]
    result: RemoteResult
    image: list[tuple[object, ...]] = field(default_factory=list)
    options: dict[str, dict[str, object]] = field(default_factory=dict)
    calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)
    app: tuple[str, object] | None = None
    secrets: list[dict[str, str]] = field(default_factory=list)
    ran: bool = False

    def called(self, name: str) -> list[tuple[object, ...]]:
        return [args for called, args in self.calls if called == name]


@dataclass(slots=True)
class FakeFunction:
    remote_side: Remote
    name: str

    @property
    def remote(self) -> SimpleNamespace:
        async def aio(*args: object) -> object:
            self.remote_side.calls.append((self.name, args))
            return self.remote_side.fingerprint if self.name == "fingerprint_remote" else self.remote_side.result

        return SimpleNamespace(aio=aio)


def install_fake_modal(monkeypatch: pytest.MonkeyPatch, remote: Remote) -> Remote:
    class Chain:
        def __getattr__(self, name: str) -> Callable[..., Chain]:
            def step(*args: object, **kwargs: object) -> Chain:
                remote.image.append((name, args, kwargs))
                return self

            return step

    class Image:
        @staticmethod
        def debian_slim(*, python_version: str) -> Chain:
            remote.image.append(("debian_slim", (python_version,), {}))
            return Chain()

    class Run:
        async def __aenter__(self) -> None:
            remote.ran = True

        async def __aexit__(self, *exc: object) -> None: ...

    class App:
        def __init__(self, name: str, *, image: object) -> None:
            remote.app = (name, image)

        def function(self, **options: object) -> Callable[[Callable[..., object]], FakeFunction]:
            def decorate(fn: Callable[..., object]) -> FakeFunction:
                remote.options[fn.__name__] = options
                return FakeFunction(remote, fn.__name__)

            return decorate

        def run(self) -> Run:
            return Run()

    class Secret:
        @staticmethod
        def from_dict(values: dict[str, str]) -> object:
            remote.secrets.append(values)
            return values

    module = ModuleType("modal")
    module.__spec__ = ModuleSpec("modal", None)
    module.Image = Image
    module.App = App
    module.Secret = Secret
    monkeypatch.setitem(sys.modules, "modal", module)
    return remote


@dataclass(slots=True)
class Converge:
    """The mocked HF-and-sidecar tail: snapshot, PEFT-to-mlx convert, then fuse."""

    peft_dir: Path
    adapter_dir: Path
    mlx_path: Path
    order: list[str] = field(default_factory=list)


@pytest.fixture(autouse=True)
def fake_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("datasets")
    module.Dataset = FakeDataset
    monkeypatch.setitem(sys.modules, "datasets", module)


@pytest.fixture(autouse=True)
def settings_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_write_token")
    monkeypatch.setenv("ATHOME_TRAIN_WORK_ROOT", str(tmp_path / "runs"))
    load.cache_clear()


@pytest.fixture
def converge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Converge:
    state = Converge(tmp_path / "peft", tmp_path / "adapter", tmp_path / "mlx")

    async def ensure_write_auth() -> None:
        state.order.append("preflight")

    async def snapshot(repo: str, *, revision: str | None = None) -> Path:
        state.order.append(f"snapshot:{repo}@{revision}")
        return state.peft_dir

    async def convert_peft_to_mlx(peft_dir: Path, out_dir: Path, *, base: object, lora: object) -> Path:
        state.order.append(f"convert:{peft_dir}->{out_dir}")
        return state.adapter_dir

    async def fuse(adapter_dir: Path, out_dir: Path, *, base: object) -> Path:
        state.order.append(f"fuse:{adapter_dir}->{out_dir}")
        return state.mlx_path

    monkeypatch.setattr(train_modal, "ensure_write_auth", ensure_write_auth)
    monkeypatch.setattr(train_modal, "snapshot", snapshot)
    monkeypatch.setattr(train_modal, "convert_peft_to_mlx", convert_peft_to_mlx)
    monkeypatch.setattr(train_modal, "fuse", fuse)
    return state


def corpus(tmp_path: Path, rows: Sequence[dict[str, object]]) -> LocalJsonlRef:
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows))
    return LocalJsonlRef(path)


def train_spec(tmp_path: Path, *, method: Method = "sft", max_usd: float | None = None) -> TrainSpec:
    return TrainSpec(
        name="pilot",
        base=BASE,
        dataset=corpus(tmp_path, SFT_ROWS if method == "sft" else DPO_ROWS),
        hyperparams=Hyperparams(steps=4, batch_size=2, max_seq_len=128),
        method=method,
        lora=LoraSpec(),
        max_usd=max_usd,
    )


def sink(tmp_path: Path) -> RunSink:
    return RunSink.open(tmp_path / "run.jsonl")


def records(tmp_path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (tmp_path / "run.jsonl").read_text().splitlines()]


def test_backend_satisfies_the_protocol_and_trains_both_trl_methods() -> None:
    assert ModalTrainBackend.name == "modal"
    assert ModalTrainBackend.supports("sft")
    assert ModalTrainBackend.supports("dpo")
    assert isinstance(ModalTrainBackend(ModalTrainSettings()), TrainBackend)


def test_available_needs_modal_and_a_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_modal(monkeypatch, Remote(MATCHING_FINGERPRINT, TRAINED))
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-123")
    assert ModalTrainBackend.available()

    monkeypatch.setattr(train_modal.importlib.util, "find_spec", lambda name: None)
    assert not ModalTrainBackend.available()


def test_service_spec_fingerprints_the_pins_and_the_lora_shape() -> None:
    assert fingerprint_for(service_spec(ModalTrainSettings(), LoraSpec())) == MATCHING_FINGERPRINT


@pytest.mark.parametrize(
    ("skew", "expected"),
    [
        pytest.param({"param:trl": "0.22.0"}, "trl", id="trl-skew"),
        pytest.param({"param:peft": "0.15.0"}, "peft", id="peft-skew"),
        pytest.param({"param:torch": "2.7.0"}, "torch", id="torch-skew"),
        pytest.param({"param:transformers": "4.49.0"}, "transformers", id="transformers-skew"),
        pytest.param({"param:lora_rank": 8}, "lora_rank", id="lora-rank-skew"),
        pytest.param({"param:lora_alpha": 64}, "lora_alpha", id="lora-alpha-skew"),
        pytest.param({"param:lora_target_modules": ["mlp.gate_proj"]}, "lora_target_modules", id="lora-modules-skew"),
    ],
)
async def test_parity_mismatch_raises_before_the_gpu_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, converge: Converge, skew: dict[str, object], expected: str
) -> None:
    remote = install_fake_modal(monkeypatch, Remote(MATCHING_FINGERPRINT | skew, TRAINED))

    with pytest.raises(ParityMismatch) as excinfo:
        await ModalTrainBackend(ModalTrainSettings()).train(train_spec(tmp_path), sink=sink(tmp_path))

    assert expected in str(excinfo.value)
    assert "athome-train" in str(excinfo.value)
    assert remote.called("fingerprint_remote")
    assert remote.called("train_remote") == []
    assert converge.order == ["preflight"]


async def test_sft_feeds_the_trainer_prompt_and_completion_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, converge: Converge
) -> None:
    remote = install_fake_modal(monkeypatch, Remote(MATCHING_FINGERPRINT, TRAINED))

    await ModalTrainBackend(ModalTrainSettings()).train(train_spec(tmp_path, method="sft"), sink=sink(tmp_path))

    ((config, dataset),) = remote.called("train_remote")
    assert config.method == "sft"
    assert dataset.rows == [
        {"prompt": [{"role": "user", "content": "q"}], "completion": [{"role": "assistant", "content": "a"}]}
    ]


async def test_dpo_feeds_the_trainer_chosen_and_rejected_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, converge: Converge
) -> None:
    remote = install_fake_modal(monkeypatch, Remote(MATCHING_FINGERPRINT, TRAINED))

    await ModalTrainBackend(ModalTrainSettings()).train(train_spec(tmp_path, method="dpo"), sink=sink(tmp_path))

    ((config, dataset),) = remote.called("train_remote")
    assert config.method == "dpo"
    assert dataset.rows == [
        {
            "prompt": [{"role": "user", "content": "q"}],
            "chosen": [{"role": "assistant", "content": "yes"}],
            "rejected": [{"role": "assistant", "content": "no"}],
        }
    ]


async def test_a_projected_breach_aborts_before_modal_is_touched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, converge: Converge
) -> None:
    remote = install_fake_modal(monkeypatch, Remote(MATCHING_FINGERPRINT, TRAINED))
    settings = ModalTrainSettings()
    spec = train_spec(tmp_path, max_usd=0.10)
    assert projected_usd(spec, settings) > 0.10

    with pytest.raises(SpendExceeded, match=r"exceeds cap \$0.1000"):
        await ModalTrainBackend(settings).train(spec, sink=sink(tmp_path))

    assert not remote.ran
    assert remote.calls == []
    assert remote.app is None


async def test_a_run_that_bills_past_the_cap_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, converge: Converge
) -> None:
    install_fake_modal(
        monkeypatch, Remote(MATCHING_FINGERPRINT, RemoteResult("athome-train/pilot", "c0ffee", 4, 7200.0))
    )

    with pytest.raises(SpendExceeded, match=r"billed \$7.9000, over the \$1.0000 cap"):
        await ModalTrainBackend(ModalTrainSettings()).train(train_spec(tmp_path, max_usd=1.0), sink=sink(tmp_path))

    assert "convert" not in " ".join(converge.order)


def test_the_cap_caps_the_gpu_functions_timeout() -> None:
    settings = ModalTrainSettings()
    assert budget_seconds(1.0, settings) == 911
    assert budget_seconds(settings.spend_cap_usd, settings) == 54683
    assert budget_seconds(1_000_000.0, settings) == 86400


async def test_dpo_projects_two_passes_of_the_corpus(tmp_path: Path) -> None:
    settings = ModalTrainSettings()
    sft, dpo = train_spec(tmp_path, method="sft"), train_spec(tmp_path, method="dpo")
    startup = projected_usd(
        TrainSpec(name="empty", base=BASE, dataset=sft.dataset, hyperparams=Hyperparams(steps=0)), settings
    )

    assert projected_usd(dpo, settings) - startup == pytest.approx(2 * (projected_usd(sft, settings) - startup))


async def test_the_gpu_function_carries_the_gpu_class_the_budget_and_the_hf_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, converge: Converge
) -> None:
    brief = RemoteResult(repo="athome-train/pilot", revision="c0ffee", step=4, seconds=600.0)
    remote = install_fake_modal(monkeypatch, Remote(MATCHING_FINGERPRINT, brief))

    await ModalTrainBackend(ModalTrainSettings()).train(train_spec(tmp_path, max_usd=1.0), sink=sink(tmp_path))

    assert remote.options["train_remote"]["gpu"] == "H100"
    assert remote.options["train_remote"]["timeout"] == 911
    assert "gpu" not in remote.options["fingerprint_remote"]
    assert remote.secrets == [{"HF_TOKEN": "hf_write_token"}]
    assert remote.app == ("athome-train", remote.app[1])


async def test_the_image_pins_the_trl_stack_and_bakes_the_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, converge: Converge
) -> None:
    remote = install_fake_modal(monkeypatch, Remote(MATCHING_FINGERPRINT, TRAINED))

    await ModalTrainBackend(ModalTrainSettings()).train(train_spec(tmp_path), sink=sink(tmp_path))

    steps = {name: (args, kwargs) for name, args, kwargs in remote.image}
    assert steps["pip_install"][0][:4] == ("trl==0.21.0", "peft==0.14.0", "torch==2.6.0", "transformers==4.48.0")
    assert steps["add_local_python_source"] == (("athome",), {"copy": True})
    assert steps["run_function"] == ((train_modal.download_base,), {"args": (BASE.hf,)})
    assert list(steps) == ["debian_slim", "uv_sync", "pip_install", "env", "add_local_python_source", "run_function"]


async def test_the_checkpoint_is_the_fused_mlx_model_the_sidecar_produced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, converge: Converge
) -> None:
    install_fake_modal(monkeypatch, Remote(MATCHING_FINGERPRINT, TRAINED))

    checkpoint = await ModalTrainBackend(ModalTrainSettings()).train(train_spec(tmp_path), sink=sink(tmp_path))

    assert checkpoint.mlx_path == converge.mlx_path
    assert checkpoint.adapter_dir == converge.adapter_dir
    assert checkpoint.backend == "modal"
    assert checkpoint.method == "sft"
    assert checkpoint.base == BASE
    assert checkpoint.step == 4
    assert checkpoint.train_cost_usd == pytest.approx(1.975)
    run_dir = tmp_path / "runs" / "pilot" / "modal"
    assert converge.order == [
        "preflight",
        "snapshot:athome-train/pilot@c0ffee",
        f"convert:{converge.peft_dir}->{run_dir / 'adapter'}",
        f"fuse:{converge.adapter_dir}->{run_dir / 'mlx'}",
    ]


async def test_the_run_is_journaled_from_launch_to_the_fused_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, converge: Converge
) -> None:
    install_fake_modal(monkeypatch, Remote(MATCHING_FINGERPRINT, TRAINED))

    await ModalTrainBackend(ModalTrainSettings()).train(train_spec(tmp_path), sink=sink(tmp_path))

    journal = records(tmp_path)
    assert [record["stage"] for record in journal] == ["launch", "trained", "converged"]
    assert journal[0]["gpu"] == "H100"
    assert journal[1]["revision"] == "c0ffee"
    assert journal[1]["usd"] == pytest.approx(1.975)
    assert journal[2]["mlx_path"] == str(converge.mlx_path)
