from __future__ import annotations

import dataclasses
import importlib.machinery
import importlib.util
import inspect
import json
import math
import sys
import tarfile
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING

import httpx
import pytest

from athome.config import load
from athome.llm.spend import SpendExceeded
from athome.progress import RunSink, load_journal
from athome.train import data, sidecar, tinker
from athome.train.spec import BASE_MODELS, STD_MODULES, Hyperparams, LocalJsonlRef, LoraSpec, TrainSpec
from athome.train.tinker import (
    BETA,
    TORCH_HINT,
    TinkerBackend,
    TorchRequired,
    UnservableBase,
    UnsupportedLoraShape,
    download_adapter,
    dpo_loss,
    tinker_lora,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

    from athome.train.data import TinkerPreference
    from athome.train.spec import Method

LOGPROB = -0.5
CANNED = {"loss": 0.61, "margin": 0.2, "accuracy": 1.0}
type LossFn = Callable[..., tuple[object, dict[str, float]]]


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

    @property
    def length(self) -> int:
        return len(self.ids)


@dataclass(frozen=True, slots=True)
class FakeDatum:
    model_input: FakeModelInput
    loss_fn_inputs: dict[str, FakeTensor]


@dataclass(frozen=True, slots=True)
class FakeAdamParams:
    learning_rate: float


@dataclass(frozen=True, slots=True)
class FakeFuture[T]:
    value: T

    async def result_async(self) -> T:
        return self.value


@dataclass(frozen=True, slots=True)
class FakeOutput:
    loss_fn_outputs: list[dict[str, FakeTensor]]
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FakeSaved:
    path: str


@dataclass(frozen=True, slots=True)
class FakeUrl:
    url: str


@dataclass(frozen=True, slots=True)
class CustomCall:
    datums: list[FakeDatum]
    loss_fn: LossFn
    loss_type_input: str


def logprobs_for(datums: Sequence[FakeDatum]) -> list[dict[str, FakeTensor]]:
    return [
        {
            "logprobs": FakeTensor(
                data=[LOGPROB] * len(datum.loss_fn_inputs["target_tokens"].data),
                dtype="float32",
                shape=[len(datum.loss_fn_inputs["target_tokens"].data)],
            )
        }
        for datum in datums
    ]


@dataclass(slots=True)
class FakeTraining:
    """Records every call and answers each forward with a flat ``LOGPROB`` per target token."""

    base_model: str
    rank: int
    seed: int
    toggles: tuple[bool, bool, bool]
    forward_backward: list[tuple[list[FakeDatum], str]] = field(default_factory=list)
    custom: list[CustomCall] = field(default_factory=list)
    forward: list[list[FakeDatum]] = field(default_factory=list)
    optim: list[FakeAdamParams] = field(default_factory=list)
    saved: list[str] = field(default_factory=list)

    async def forward_backward_async(self, datums: list[FakeDatum], loss_fn: str) -> FakeFuture[FakeOutput]:
        self.forward_backward.append((datums, loss_fn))
        return FakeFuture(FakeOutput(logprobs_for(datums)))

    async def forward_async(self, datums: list[FakeDatum], loss_fn: str) -> FakeFuture[FakeOutput]:
        self.forward.append(datums)
        return FakeFuture(FakeOutput(logprobs_for(datums)))

    async def forward_backward_custom_async(
        self, datums: list[FakeDatum], loss_fn: LossFn, *, loss_type_input: str = "logprobs"
    ) -> FakeFuture[FakeOutput]:
        self.custom.append(CustomCall(datums, loss_fn, loss_type_input))
        return FakeFuture(FakeOutput(logprobs_for(datums), metrics=dict(CANNED)))

    async def optim_step_async(self, adam_params: FakeAdamParams) -> FakeFuture[None]:
        self.optim.append(adam_params)
        return FakeFuture(None)

    async def save_weights_for_sampler_async(self, name: str) -> FakeFuture[FakeSaved]:
        self.saved.append(name)
        return FakeFuture(FakeSaved(f"tinker://run/{name}"))


@dataclass(slots=True)
class FakeService:
    api_key: str
    clients: list[FakeTraining] = field(default_factory=list)

    async def create_lora_training_client_async(
        self, base_model: str, rank: int, seed: int, train_mlp: bool, train_attn: bool, train_unembed: bool
    ) -> FakeTraining:
        """Tinker's trainer takes a rank and the three toggles — no alpha, dropout, or module list."""
        self.clients.append(
            FakeTraining(base_model=base_model, rank=rank, seed=seed, toggles=(train_mlp, train_attn, train_unembed))
        )
        return self.clients[-1]


class FakeTokenizer:
    """A char-level chat tokenizer: the templated prompt is always a prefix of the full text."""

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool, enable_thinking: bool
    ) -> str:
        rendered = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        return f"{rendered}<assistant>" if add_generation_prompt else rendered

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return [ord(char) for char in text]


@pytest.fixture(autouse=True)
def sdk(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = ModuleType("tinker")
    module.__spec__ = importlib.machinery.ModuleSpec("tinker", None)
    module.Datum = FakeDatum
    module.ModelInput = FakeModelInput
    module.TensorData = FakeTensor
    module.AdamParams = FakeAdamParams
    monkeypatch.setitem(sys.modules, "tinker", module)
    return module


@pytest.fixture(autouse=True)
def fake_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "tokenizer", lambda mlx_id: FakeTokenizer())


@pytest.fixture(autouse=True)
def tinker_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("TINKER_API_KEY", "sk-tinker-test")
    monkeypatch.setenv("ATHOME_TRAIN_WORK_ROOT", str(tmp_path / "runs"))
    monkeypatch.setattr(tinker, "TINKER_ENV", tmp_path / "tinker.env")
    load.cache_clear()
    yield
    load.cache_clear()


@pytest.fixture
def service(sdk: ModuleType) -> FakeService:
    instance = FakeService("sk-tinker-test")
    sdk.ServiceClient = lambda api_key: instance
    return instance


@pytest.fixture
def torch_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Declare torch importable without importing it: the fake SDK stands in for the backprop it would do."""
    real = importlib.util.find_spec
    installed = importlib.machinery.ModuleSpec("torch", None)
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *args: installed if name == "torch" else real(name, *args)
    )


@pytest.fixture
def converged(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    seen: dict[str, object] = {}

    async def download(service: object, tinker_path: str, out_dir: Path) -> Path:
        seen["tinker_path"] = tinker_path
        seen["peft"] = out_dir
        return out_dir

    async def convert(peft_dir: Path, out_dir: Path, *, base: object) -> Path:
        seen["convert_from"] = peft_dir
        seen["adapter"] = out_dir
        return out_dir

    async def fuse(adapter_dir: Path, out_dir: Path, *, base: object) -> Path:
        seen["fuse_from"] = adapter_dir
        seen["mlx"] = out_dir
        return out_dir

    monkeypatch.setattr(tinker, "download_adapter", download)
    monkeypatch.setattr(tinker.sidecar, "convert_peft_to_mlx", convert)
    monkeypatch.setattr(tinker.sidecar, "fuse", fuse)
    return seen


def corpus(tmp_path: Path, *, method: Method, rows: int = 4) -> LocalJsonlRef:
    match method:
        case "sft":
            lines = [
                {"messages": [{"role": "user", "content": f"q{index}"}, {"role": "assistant", "content": "yes"}]}
                for index in range(rows)
            ]
        case "dpo":
            lines = [
                {
                    "prompt": [{"role": "user", "content": f"q{index}"}],
                    "chosen": [{"role": "assistant", "content": "yes"}],
                    "rejected": [{"role": "assistant", "content": "no"}],
                }
                for index in range(rows)
            ]
    path = tmp_path / f"{method}.jsonl"
    path.write_text("".join(f"{json.dumps(line)}\n" for line in lines))
    return LocalJsonlRef(path=path)


def spec(dataset: LocalJsonlRef, **overrides: object) -> TrainSpec:
    return dataclasses.replace(
        TrainSpec(
            name="watcher",
            base=BASE_MODELS["qwen3-8b"],
            dataset=dataset,
            hyperparams=Hyperparams(steps=3, batch_size=2, learning_rate=2e-4, max_seq_len=64),
        ),
        **overrides,
    )


def sink(tmp_path: Path) -> RunSink:
    return RunSink.open(tmp_path / "run.jsonl")


async def pairs_of(request: TrainSpec) -> list[TinkerPreference]:
    return [
        data.render_tinker_dpo(example, request.base.mlx)
        for example in await data.normalize(request.dataset, method="dpo")
    ]


def completion_of(datum: FakeDatum) -> str:
    return bytes(
        token
        for token, weight in zip(datum.loss_fn_inputs["target_tokens"].data, tinker.weights_of(datum), strict=True)
        if weight
    ).decode()


def hide(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    """Make one module un-importable to ``find_spec``, leaving every other lookup real."""
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args: None if name == missing else real(name, *args))


def test_supports_sft_always_and_the_name_is_stable() -> None:
    assert TinkerBackend.supports("sft")
    assert TinkerBackend.name == "tinker"


def test_supports_dpo_only_where_torch_can_back_the_custom_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    """#9 — `supports("dpo")` claimed True without torch, so select() picked tinker and then died."""
    assert TinkerBackend.supports("dpo")

    hide(monkeypatch, "torch")

    assert not TinkerBackend.supports("dpo")
    assert TinkerBackend.supports("sft")


def test_available_needs_the_sdk_on_the_path_not_just_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """#9 — credentials alone made select() pick tinker, which then raised ModuleNotFoundError."""
    assert TinkerBackend.available()

    hide(monkeypatch, "tinker")

    assert not TinkerBackend.available()


def test_available_falls_back_to_the_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert TinkerBackend.available()
    monkeypatch.delenv("TINKER_API_KEY")
    assert not TinkerBackend.available()
    (tmp_path / "tinker.env").write_text('# a comment\nTINKER_API_KEY="sk-from-file"\n')
    assert TinkerBackend.available()


def test_from_settings_loads_the_key_out_of_the_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TINKER_API_KEY")
    (tmp_path / "tinker.env").write_text('# tinker\n\nTINKER_API_KEY="sk-from-file"\n')
    load.cache_clear()

    assert TinkerBackend.from_settings().settings.api_key.get_secret_value() == "sk-from-file"


async def test_sft_runs_cross_entropy_and_one_optim_step_per_step(
    service: FakeService, converged: dict[str, object], tmp_path: Path
) -> None:
    checkpoint = await TinkerBackend.from_settings().train(
        spec(corpus(tmp_path, method="sft")), sink=sink(tmp_path), work_dir=tmp_path / "run"
    )

    training = service.clients[0]
    assert len(service.clients) == 1
    assert (training.base_model, training.rank, training.toggles) == ("Qwen/Qwen3-8B", 16, (True, True, False))
    assert [loss_fn for _, loss_fn in training.forward_backward] == ["cross_entropy"] * 3
    assert [len(batch) for batch, _ in training.forward_backward] == [2, 2, 2]
    assert training.optim == [FakeAdamParams(learning_rate=2e-4)] * 3
    assert training.saved == ["watcher-sft-3"]
    assert [completion_of(datum) for batch, _ in training.forward_backward for datum in batch] == ["yes"] * 6
    assert (checkpoint.method, checkpoint.step, checkpoint.backend) == ("sft", 3, "tinker")


async def test_sft_journals_loss_tokens_and_running_spend(
    service: FakeService, converged: dict[str, object], tmp_path: Path
) -> None:
    run = sink(tmp_path)

    checkpoint = await TinkerBackend.from_settings().train(
        spec(corpus(tmp_path, method="sft")), sink=run, work_dir=tmp_path / "run"
    )

    records = load_journal(run.path)
    tokens = [tinker.token_count(batch) for batch, _ in service.clients[0].forward_backward]
    assert [record["step"] for record in records] == [1, 2, 3]
    assert [record["loss"] for record in records] == [-LOGPROB] * 3
    assert [record["tokens"] for record in records] == tokens
    assert records[-1]["cost_usd"] == pytest.approx(sum(tokens) / 1e6 * 0.40)
    assert checkpoint.train_cost_usd == pytest.approx(sum(tokens) / 1e6 * 0.40)


async def test_over_long_examples_never_reach_a_batch(
    service: FakeService, converged: dict[str, object], tmp_path: Path
) -> None:
    path = tmp_path / "sft.jsonl"
    path.write_text(
        "".join(
            f"{json.dumps(row)}\n"
            for row in (
                {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "ok"}]},
                {"messages": [{"role": "user", "content": "q" * 200}, {"role": "assistant", "content": "ok"}]},
            )
        )
    )
    request = spec(LocalJsonlRef(path=path), hyperparams=Hyperparams(steps=2, batch_size=1, max_seq_len=32))

    await TinkerBackend.from_settings().train(request, sink=sink(tmp_path), work_dir=tmp_path / "run")

    lengths = {datum.model_input.length for batch, _ in service.clients[0].forward_backward for datum in batch}
    assert lengths == {len("<user>q<assistant>ok") - 1}


async def test_the_spend_cap_aborts_before_any_billable_call(
    service: FakeService, converged: dict[str, object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ATHOME_TRAIN_TINKER_SPEND_CAP_USD", "0.000001")
    load.cache_clear()
    request = spec(corpus(tmp_path, method="sft"), hyperparams=Hyperparams(steps=1000, batch_size=4))

    with pytest.raises(SpendExceeded, match="exceeds cap"):
        await TinkerBackend.from_settings().train(request, sink=sink(tmp_path), work_dir=tmp_path / "run")

    assert service.clients == []


async def test_the_spec_cap_overrides_the_configured_one(
    service: FakeService, converged: dict[str, object], tmp_path: Path
) -> None:
    request = spec(corpus(tmp_path, method="sft"), max_usd=1e-9)

    with pytest.raises(SpendExceeded, match=r"exceeds cap \$0\.0000"):
        await TinkerBackend.from_settings().train(request, sink=sink(tmp_path), work_dir=tmp_path / "run")

    assert service.clients == []


async def test_a_zero_cap_spends_nothing_rather_than_the_configured_sixty_dollars(
    service: FakeService, converged: dict[str, object], tmp_path: Path
) -> None:
    """#4 — `spec.max_usd or settings.spend_cap_usd` read a 0.0 cap as unset and authorized $60."""
    assert load(tinker.TinkerSettings).spend_cap_usd == 60.0
    request = spec(corpus(tmp_path, method="sft"), max_usd=0.0)

    with pytest.raises(SpendExceeded, match=r"exceeds cap \$0\.0000"):
        await TinkerBackend.from_settings().train(request, sink=sink(tmp_path), work_dir=tmp_path / "run")

    assert service.clients == []
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    ("lora", "match"),
    [
        pytest.param(LoraSpec(alpha=64), "alpha=64", id="a-non-default-alpha-tinker-never-sees"),
        pytest.param(LoraSpec(dropout=0.1), "dropout=0.1", id="a-dropout-tinker-never-sees"),
        pytest.param(
            LoraSpec(target_modules=("self_attn.q_proj",)), "cannot narrow", id="a-target-list-tinker-cannot-narrow-to"
        ),
        pytest.param(LoraSpec(train_unembed=True), "unembedding", id="an-unembedding-lora-nothing-can-fuse"),
    ],
)
async def test_tinker_refuses_a_shape_it_cannot_train_before_spending(
    service: FakeService, converged: dict[str, object], tmp_path: Path, lora: LoraSpec, match: str
) -> None:
    """#6 — these were accepted, trained as something else, then discarded by the converter."""
    request = spec(corpus(tmp_path, method="sft"), lora=lora)

    with pytest.raises(UnsupportedLoraShape, match=match):
        await TinkerBackend.from_settings().train(request, sink=sink(tmp_path), work_dir=tmp_path / "run")

    assert service.clients == []


@pytest.mark.parametrize(
    ("lora", "trained"),
    [
        pytest.param(LoraSpec(), STD_MODULES, id="the-default-shape"),
        pytest.param(LoraSpec(rank=8, train_mlp=False), STD_MODULES[:4], id="mlp-off-narrows-to-attention"),
        pytest.param(LoraSpec(train_attn=False), STD_MODULES[4:], id="attn-off-narrows-to-mlp"),
    ],
)
def test_tinker_lora_is_exactly_what_the_toggles_select(lora: LoraSpec, trained: tuple[str, ...]) -> None:
    """#6 — the modules tinker trains for a shape it accepts, which the archive then reports back."""
    assert tinker_lora(lora) == trained


async def test_the_toggles_reach_the_trainer_and_no_requested_shape_reaches_the_fuse(
    service: FakeService, converged: dict[str, object], tmp_path: Path
) -> None:
    """#6 — the shape trained must BE the shape fused: tinker gets the toggles, the converter gets the archive."""
    request = spec(corpus(tmp_path, method="sft"), lora=LoraSpec(rank=8, train_mlp=False))

    await TinkerBackend.from_settings().train(request, sink=sink(tmp_path), work_dir=tmp_path / "run")

    training = service.clients[0]
    assert (training.rank, training.toggles) == (8, (False, True, False))
    assert converged["convert_from"] == tmp_path / "run" / "peft"
    assert "lora" not in inspect.signature(sidecar.convert_peft_to_mlx).parameters


async def test_dpo_scores_against_a_frozen_reference_client(
    service: FakeService, converged: dict[str, object], torch_present: None, tmp_path: Path
) -> None:
    request = spec(corpus(tmp_path, method="dpo"), method="dpo")

    checkpoint = await TinkerBackend.from_settings().train(request, sink=sink(tmp_path), work_dir=tmp_path / "run")

    reference, policy = service.clients
    assert (reference.optim, reference.forward_backward, reference.custom) == ([], [], [])
    assert reference.forward == [tinker.pair_datums(await pairs_of(request))]
    assert len(policy.custom) == 3
    assert {call.loss_type_input for call in policy.custom} == {"logprobs"}
    assert [len(call.datums) for call in policy.custom] == [4, 4, 4]
    assert policy.optim == [FakeAdamParams(learning_rate=2e-4)] * 3
    assert policy.forward_backward == []
    assert (checkpoint.method, checkpoint.backend) == ("dpo", "tinker")


async def test_dpo_interleaves_chosen_and_rejected_with_their_reference_logprobs(
    service: FakeService, converged: dict[str, object], torch_present: None, tmp_path: Path
) -> None:
    request = spec(
        corpus(tmp_path, method="dpo"),
        method="dpo",
        hyperparams=Hyperparams(steps=1, batch_size=2, max_seq_len=64),
    )

    await TinkerBackend.from_settings().train(request, sink=sink(tmp_path), work_dir=tmp_path / "run")

    call = service.clients[1].custom[0]
    assert [completion_of(datum) for datum in call.datums] == ["yes", "no", "yes", "no"]
    assert call.loss_fn.keywords["reference"] == pytest.approx(
        [LOGPROB * sum(tinker.weights_of(datum)) for datum in call.datums]
    )
    assert call.loss_fn.keywords["beta"] == BETA


async def test_dpo_journals_the_custom_loss_metrics(
    service: FakeService, converged: dict[str, object], torch_present: None, tmp_path: Path
) -> None:
    run = sink(tmp_path)

    await TinkerBackend.from_settings().train(
        spec(corpus(tmp_path, method="dpo"), method="dpo"), sink=run, work_dir=tmp_path / "run"
    )

    records = load_journal(run.path)
    assert [record["step"] for record in records] == [1, 2, 3]
    assert all(record["method"] == "dpo" for record in records)
    assert all(record["loss"] == CANNED["loss"] and record["margin"] == CANNED["margin"] for record in records)
    assert all(record["accuracy"] == CANNED["accuracy"] for record in records)


async def test_dpo_charges_the_reference_pass_and_both_custom_passes(
    service: FakeService, converged: dict[str, object], torch_present: None, tmp_path: Path
) -> None:
    checkpoint = await TinkerBackend.from_settings().train(
        spec(corpus(tmp_path, method="dpo"), method="dpo"), sink=sink(tmp_path), work_dir=tmp_path / "run"
    )

    reference, policy = service.clients
    billed = tinker.token_count(reference.forward[0]) + 2 * sum(
        tinker.token_count(call.datums) for call in policy.custom
    )
    assert checkpoint.train_cost_usd == pytest.approx(billed / 1e6 * 0.40)


async def test_dpo_without_torch_names_the_extra_that_installs_it(
    service: FakeService, converged: dict[str, object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hide(monkeypatch, "torch")
    request = spec(corpus(tmp_path, method="dpo"), method="dpo")

    with pytest.raises(TorchRequired) as raised:
        await TinkerBackend.from_settings().train(request, sink=sink(tmp_path), work_dir=tmp_path / "run")

    assert str(raised.value) == TORCH_HINT
    assert "experiment-at-home[train-dpo]" in str(raised.value)
    assert service.clients == []


async def test_sft_runs_without_torch(
    service: FakeService, converged: dict[str, object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hide(monkeypatch, "torch")

    checkpoint = await TinkerBackend.from_settings().train(
        spec(corpus(tmp_path, method="sft")), sink=sink(tmp_path), work_dir=tmp_path / "run"
    )

    assert len(service.clients[0].forward_backward) == 3
    assert checkpoint.method == "sft"


def datum_of(length: int) -> FakeDatum:
    return FakeDatum(
        model_input=FakeModelInput([1] * length),
        loss_fn_inputs={
            "target_tokens": FakeTensor([1] * length, "int64", [length]),
            "weights": FakeTensor([1.0] * length, "float32", [length]),
        },
    )


def test_dpo_loss_is_the_log_sigmoid_of_the_reference_corrected_margin() -> None:
    torch = pytest.importorskip("torch")
    logprobs = [torch.tensor([-1.0], requires_grad=True), torch.tensor([-2.0], requires_grad=True)]

    loss, metrics = dpo_loss([datum_of(1), datum_of(1)], logprobs, reference=[-1.5, -1.5], beta=0.1)

    margin = 0.1 * ((-1.0 - -1.5) - (-2.0 - -1.5))
    assert margin == pytest.approx(0.1)
    assert loss.item() == pytest.approx(math.log1p(math.exp(-margin)), rel=1e-5)
    assert metrics["margin"] == pytest.approx(margin, rel=1e-5)
    assert metrics["accuracy"] == 1.0


def test_a_better_than_reference_chosen_lowers_the_dpo_loss() -> None:
    torch = pytest.importorskip("torch")

    def loss_for(chosen: float) -> float:
        logprobs = [torch.tensor([chosen], requires_grad=True), torch.tensor([-2.0], requires_grad=True)]
        return dpo_loss([datum_of(1), datum_of(1)], logprobs, reference=[-1.5, -1.5], beta=0.1)[0].item()

    assert loss_for(-0.5) < loss_for(-1.0) < loss_for(-1.6)
    assert loss_for(-1.5) == pytest.approx(math.log1p(math.exp(-0.1 * 0.5)), rel=1e-5)


def test_the_surrogate_weights_push_chosen_up_and_rejected_down() -> None:
    torch = pytest.importorskip("torch")
    logprobs = [torch.tensor([-1.0, -1.0], requires_grad=True), torch.tensor([-1.0, -1.0], requires_grad=True)]

    loss, metrics = dpo_loss([datum_of(2), datum_of(2)], logprobs, reference=[-1.0, -1.0], beta=BETA)
    loss.backward()

    assert loss.item() == pytest.approx(-math.log(0.5), rel=1e-5)
    assert metrics["accuracy"] == 0.0
    chosen, rejected = (-logprob.grad for logprob in logprobs)
    assert all(weight > 0 for weight in chosen.tolist())
    assert all(weight < 0 for weight in rejected.tolist())
    assert chosen.tolist() == pytest.approx([-weight for weight in rejected.tolist()])


async def test_the_checkpoint_fuses_the_downloaded_adapter_into_mlx(
    service: FakeService, converged: dict[str, object], tmp_path: Path
) -> None:
    checkpoint = await TinkerBackend.from_settings().train(
        spec(corpus(tmp_path, method="sft")), sink=sink(tmp_path), work_dir=tmp_path / "run"
    )

    run = tmp_path / "run"
    assert converged["tinker_path"] == "tinker://run/watcher-sft-3"
    assert converged["peft"] == run / "peft"
    assert converged["convert_from"] == run / "peft"
    assert converged["fuse_from"] == run / "adapter"
    assert checkpoint.mlx_path == run / "mlx"
    assert checkpoint.adapter_dir == run / "adapter"
    assert checkpoint.base == BASE_MODELS["qwen3-8b"]


async def test_a_base_with_no_mlx_lm_counterpart_aborts_before_the_service_client(
    service: FakeService, tmp_path: Path
) -> None:
    request = spec(corpus(tmp_path, method="sft"), base=BASE_MODELS["qwen3.5-4b"])

    with pytest.raises(UnservableBase, match="mlx-lm LoRA counterpart"):
        await TinkerBackend.from_settings().train(request, sink=sink(tmp_path), work_dir=tmp_path / "run")

    assert service.clients == []


async def test_download_adapter_unpacks_the_signed_archive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    member = tmp_path / "adapter_model.safetensors"
    member.write_bytes(b"weights")
    archive = tmp_path / "checkpoint.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(member, arcname="adapter_model.safetensors")

    class FakeRest:
        async def get_checkpoint_archive_url_from_tinker_path_async(self, tinker_path: str) -> FakeUrl:
            assert tinker_path == "tinker://run/step3"
            return FakeUrl("https://tinker.example/archive.tar")

    class RestService:
        def create_rest_client(self) -> FakeRest:
            return FakeRest()

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://tinker.example/archive.tar"
        return httpx.Response(200, content=archive.read_bytes())

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real(transport=httpx.MockTransport(handler), **kwargs))

    out = await download_adapter(RestService(), "tinker://run/step3", tmp_path / "peft")

    assert (out / "adapter_model.safetensors").read_bytes() == b"weights"
    assert not (out / "checkpoint.tar").exists()
