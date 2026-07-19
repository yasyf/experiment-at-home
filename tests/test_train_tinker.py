from __future__ import annotations

import dataclasses
import importlib.machinery
import importlib.util
import inspect
import json
import math
import tarfile
from typing import TYPE_CHECKING

import anyio
import httpx
import pytest

from athome.config import load
from athome.llm.spend import SpendExceeded, SpendGuard
from athome.progress import RunSink, load_journal
from athome.train import data, sidecar, tinker
from athome.train.spec import (
    BASE_MODELS,
    STD_MODULES,
    Adapter,
    CheckpointPolicy,
    EvalRow,
    Hyperparams,
    InsufficientData,
    LocalJsonlRef,
    LoraSpec,
    OverlongEvalRows,
    SavedCheckpoint,
    TinkerModelId,
    TrainSpec,
)
from athome.train.tinker import (
    BETA,
    TORCH_HINT,
    TinkerBackend,
    TorchRequired,
    UnservableBase,
    UnsupportedLoraShape,
    download_adapter,
    dpo_loss,
    score_sequence,
    tinker_lora,
    tinker_model,
)
from tests import tinker_fakes
from tests.tinker_fakes import (
    CANNED,
    LOGPROB,
    Boom,
    FakeAdamParams,
    FakeModelInput,
    FakeSamplingParams,
    FakeService,
    FakeTokenizer,
    FakeUrl,
    install_sdk,
    make_datum,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path
    from types import ModuleType

    from athome.train.data import TinkerPreference
    from athome.train.spec import Method


@dataclasses.dataclass(frozen=True, slots=True)
class FakeSampledSequence:
    tokens: list[int]
    stop_reason: str = "length"
    logprobs: list[float] | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class FakeSampleResponse:
    sequences: list[FakeSampledSequence]


@dataclasses.dataclass(slots=True)
class FakeSampling:
    outputs: dict[tuple[int, ...], list[float | None]] = dataclasses.field(default_factory=dict)
    generated: dict[tuple[int, ...], list[int]] = dataclasses.field(default_factory=dict)
    calls: list[tuple[int, ...]] = dataclasses.field(default_factory=list)
    finished: list[tuple[int, ...]] = dataclasses.field(default_factory=list)
    sample_calls: list[tuple[tuple[int, ...], int, FakeSamplingParams]] = dataclasses.field(default_factory=list)
    slow: tuple[int, ...] | None = None
    release: anyio.Event | None = None

    async def compute_logprobs_async(self, prompt: FakeModelInput) -> list[float | None]:
        key = tuple(prompt.ids)
        self.calls.append(key)
        if self.release is not None:
            if key == self.slow:
                await self.release.wait()
            else:
                self.release.set()
        self.finished.append(key)
        return self.outputs.get(key, [None, *([LOGPROB] * (len(key) - 1))])

    async def sample_async(
        self, prompt: FakeModelInput, num_samples: int, sampling_params: FakeSamplingParams
    ) -> FakeSampleResponse:
        key = tuple(prompt.ids)
        self.sample_calls.append((key, num_samples, sampling_params))
        if self.release is not None:
            if key == self.slow:
                await self.release.wait()
            else:
                self.release.set()
        self.finished.append(key)
        return FakeSampleResponse(
            [FakeSampledSequence(tokens=self.generated.get(key, [ord("x")])) for _ in range(num_samples)]
        )


@dataclasses.dataclass(slots=True)
class FakeSamplingService(FakeService):
    sampling: FakeSampling = dataclasses.field(default_factory=FakeSampling)
    sampling_paths: list[str | None] = dataclasses.field(default_factory=list)
    sampling_base_models: list[str | None] = dataclasses.field(default_factory=list)

    async def create_sampling_client_async(
        self, model_path: str | None = None, base_model: str | None = None
    ) -> FakeSampling:
        self.sampling_paths.append(model_path)
        self.sampling_base_models.append(base_model)
        return self.sampling


@pytest.fixture(autouse=True)
def sdk(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return install_sdk(monkeypatch)


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
def sampling_service(sdk: ModuleType) -> FakeSamplingService:
    instance = FakeSamplingService("sk-tinker-test")
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


def budget(max_usd: float | None = 60.0) -> SpendGuard:
    return SpendGuard(max_usd=max_usd)


def step_records(path: Path) -> list[dict[str, object]]:
    return [record for record in load_journal(path) if record.get("event") != "checkpoint"]


def checkpoint_events(path: Path) -> list[dict[str, object]]:
    return [record for record in load_journal(path) if record.get("event") == "checkpoint"]


async def pairs_of(request: TrainSpec) -> list[TinkerPreference]:
    return [
        data.render_tinker_dpo(example, request.base.mlx)
        for example in await data.normalize(request.dataset, method="dpo")
    ]


def completion_of(datum: object) -> str:
    return bytes(
        token
        for token, weight in zip(datum.loss_fn_inputs["target_tokens"].data, tinker.weights_of(datum), strict=True)
        if weight
    ).decode()


def hide(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    """Make one module un-importable to ``find_spec``, leaving every other lookup real."""
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args: None if name == missing else real(name, *args))


def show(monkeypatch: pytest.MonkeyPatch, present: str) -> None:
    """Make one module importable to ``find_spec`` even where it is not installed.

    The inverse of :func:`hide`, so a probe that turns on a package's presence can be driven both
    ways in a run that does not have it — the free-threaded job installs no torch.
    """
    real = importlib.util.find_spec
    module_spec = importlib.machinery.ModuleSpec(present, None)
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *args: module_spec if name == present else real(name, *args)
    )


def test_supports_sft_always_and_the_name_is_stable() -> None:
    assert TinkerBackend.supports("sft")
    assert TinkerBackend.name == "tinker"


def test_supports_dpo_only_where_torch_can_back_the_custom_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    """#9 — `supports("dpo")` claimed True without torch, so select() picked tinker and then died."""
    show(monkeypatch, "torch")

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


def test_cost_bills_each_token_class_at_its_own_rate() -> None:
    backend = TinkerBackend.from_settings()
    model = TinkerModelId("Qwen/Qwen3-8B")

    assert backend.cost(model=model, prefill=2_000_000) == pytest.approx(0.39)
    assert backend.cost(model=model, sample=500_000) == pytest.approx(0.30)
    assert backend.cost(model=model, train=250_000) == pytest.approx(0.11)
    assert backend.cost(model=model, prefill=2_000_000, sample=500_000, train=250_000) == pytest.approx(0.80)


def test_cost_prices_each_base_model_off_its_own_sheet() -> None:
    backend = TinkerBackend.from_settings()

    assert backend.cost(model=TinkerModelId("Qwen/Qwen3.5-4B"), prefill=1_000_000) == pytest.approx(0.33)
    assert backend.cost(model=TinkerModelId("Qwen/Qwen3.5-4B"), train=1_000_000) == pytest.approx(0.737)
    assert backend.cost(model=TinkerModelId("Qwen/Qwen3.5-9B"), prefill=1_000_000) == pytest.approx(0.66)
    assert backend.cost(model=TinkerModelId("Qwen/Qwen3.5-9B"), train=1_000_000) == pytest.approx(1.463)
    assert backend.cost(model=TinkerModelId("Qwen/Qwen3.6-35B-A3B"), train=1_000_000) == pytest.approx(1.177)


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
    assert training.saves == [("watcher-sft-3", None)]
    assert [completion_of(datum) for batch, _ in training.forward_backward for datum in batch] == ["yes"] * 6
    assert (checkpoint.method, checkpoint.step, checkpoint.backend) == ("sft", 3, "tinker")


async def test_sft_journals_loss_tokens_and_running_spend(
    service: FakeService, converged: dict[str, object], tmp_path: Path
) -> None:
    run = sink(tmp_path)

    checkpoint = await TinkerBackend.from_settings().train(
        spec(corpus(tmp_path, method="sft")), sink=run, work_dir=tmp_path / "run"
    )

    records = step_records(run.path)
    tokens = [tinker.token_count(batch) for batch, _ in service.clients[0].forward_backward]
    assert [record["step"] for record in records] == [1, 2, 3]
    assert [record["loss"] for record in records] == [-LOGPROB] * 3
    assert [record["tokens"] for record in records] == tokens
    assert records[-1]["cost_usd"] == pytest.approx(sum(tokens) / 1e6 * 0.44)
    assert checkpoint.train_cost_usd == pytest.approx(sum(tokens) / 1e6 * 0.44)


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

    report = await TinkerBackend.from_settings().fit(request, sink=sink(tmp_path), budget=budget())

    lengths = {datum.model_input.length for batch, _ in service.clients[0].forward_backward for datum in batch}
    assert lengths == {len("<user>q<assistant>ok") - 1}
    assert report.dropped == 1


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


async def test_fit_refuses_before_spend_when_its_projection_alone_exceeds_the_envelope(
    service: FakeService, tmp_path: Path
) -> None:
    request = spec(corpus(tmp_path, method="sft"), hyperparams=Hyperparams(steps=1000, batch_size=4))

    with pytest.raises(SpendExceeded, match="exceeds cap"):
        await TinkerBackend.from_settings().fit(request, sink=sink(tmp_path), budget=SpendGuard(max_usd=1e-6))

    assert service.clients == []


async def test_fit_ignores_the_spec_cap_in_favor_of_the_envelope(service: FakeService, tmp_path: Path) -> None:
    """fit reads only the passed envelope: a spec cap that would refuse under train is never consulted."""
    envelope = SpendGuard(max_usd=60.0)
    request = spec(corpus(tmp_path, method="sft"), max_usd=1e-9)

    report = await TinkerBackend.from_settings().fit(request, sink=sink(tmp_path), budget=envelope)

    assert len(report.steps) == 3
    assert report.train_cost_usd == pytest.approx(envelope.spent)
    assert envelope.spent > 1e-9


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


async def test_an_under_filled_pool_aborts_before_the_guard_and_client(
    service: FakeService, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The InsufficientData floor precedes the spend guard and any client, and carries the count."""
    checked: list[float] = []

    async def spy_check(self: SpendGuard, projected: float) -> None:
        checked.append(projected)

    monkeypatch.setattr(SpendGuard, "check", spy_check)
    request = spec(
        corpus(tmp_path, method="sft", rows=2), hyperparams=Hyperparams(steps=3, batch_size=4, max_seq_len=64)
    )

    with pytest.raises(InsufficientData) as raised:
        await TinkerBackend.from_settings().fit(request, sink=sink(tmp_path), budget=budget())

    assert raised.value.count == 2
    assert service.clients == []
    assert checked == []


async def test_an_overlong_eval_row_fails_the_whole_set_up_front(service: FakeService, tmp_path: Path) -> None:
    rows = (
        EvalRow(tokens=tuple(range(100)), weights=(0.0,) * 99 + (1.0,)),
        EvalRow(tokens=(1, 2, 3), weights=(0.0, 0.0, 1.0)),
    )
    request = spec(corpus(tmp_path, method="sft"), hyperparams=Hyperparams(steps=3, batch_size=2, max_seq_len=32))

    with pytest.raises(OverlongEvalRows) as raised:
        await TinkerBackend.from_settings().fit(request, sink=sink(tmp_path), budget=budget(), eval_rows=rows)

    assert raised.value.count == 1
    assert service.clients == []


async def test_fit_snapshots_at_the_cadence_with_names_ttls_and_eval_scores(
    service: FakeService, tmp_path: Path
) -> None:
    rows = (EvalRow(tokens=(10, 20, 30), weights=(0.0, 0.0, 1.0)),)
    request = spec(
        corpus(tmp_path, method="sft", rows=8), hyperparams=Hyperparams(steps=4, batch_size=2, max_seq_len=64)
    )

    report = await TinkerBackend.from_settings().fit(
        request, sink=sink(tmp_path), budget=budget(), checkpoints=CheckpointPolicy(at=(0.5,)), eval_rows=rows
    )

    assert service.clients[0].saves == [("watcher-step00002", 604_800), ("watcher-sft-4", None)]
    assert [(checkpoint.step, checkpoint.final) for checkpoint in report.checkpoints] == [(2, False), (4, True)]
    assert report.final.step == 4
    assert all(checkpoint.scores is not None and len(checkpoint.scores) == 1 for checkpoint in report.checkpoints)
    assert report.final.scores[0].logprob == pytest.approx(LOGPROB)
    assert report.final.scores[0].weight == 1.0


async def test_the_sink_interleaves_step_records_and_checkpoint_events_in_order(
    service: FakeService, tmp_path: Path
) -> None:
    run = sink(tmp_path)
    request = spec(
        corpus(tmp_path, method="sft", rows=8), hyperparams=Hyperparams(steps=4, batch_size=2, max_seq_len=64)
    )

    await TinkerBackend.from_settings().fit(request, sink=run, budget=budget(), checkpoints=CheckpointPolicy(at=(0.5,)))

    kinds = ["checkpoint" if record.get("event") == "checkpoint" else "step" for record in load_journal(run.path)]
    assert kinds == ["step", "step", "checkpoint", "step", "step", "checkpoint"]
    costs = [record["cost_usd"] for record in step_records(run.path)]
    assert costs == sorted(costs)
    assert len(set(costs)) == len(costs)
    assert [event["step"] for event in checkpoint_events(run.path)] == [2, 4]
    assert [event["final"] for event in checkpoint_events(run.path)] == [False, True]


async def test_the_report_orders_checkpoints_and_totals_the_spend(service: FakeService, tmp_path: Path) -> None:
    request = spec(
        corpus(tmp_path, method="sft", rows=8), hyperparams=Hyperparams(steps=4, batch_size=2, max_seq_len=64)
    )

    report = await TinkerBackend.from_settings().fit(
        request, sink=sink(tmp_path), budget=budget(), checkpoints=CheckpointPolicy(at=(0.25, 0.75))
    )

    assert [checkpoint.step for checkpoint in report.checkpoints] == [1, 3, 4]
    assert report.final is report.checkpoints[-1]
    assert report.checkpoints[-1].final
    assert not any(checkpoint.final for checkpoint in report.checkpoints[:-1])
    assert report.train_cost_usd == pytest.approx(sum(record.tokens for record in report.steps) / 1e6 * 0.44)
    assert (report.method, report.dropped, len(report.steps)) == ("sft", 0, 4)


async def test_eval_prefill_is_billed_into_the_run_cost(service: FakeService, tmp_path: Path) -> None:
    rows = (EvalRow(tokens=(1, 2, 3, 4, 5), weights=(0.0, 0.0, 0.0, 0.0, 1.0)),)
    request = spec(
        corpus(tmp_path, method="sft", rows=8), hyperparams=Hyperparams(steps=2, batch_size=2, max_seq_len=64)
    )

    report = await TinkerBackend.from_settings().fit(
        request, sink=sink(tmp_path), budget=budget(), checkpoints=CheckpointPolicy(at=(0.5,)), eval_rows=rows
    )

    train_tokens = sum(record.tokens for record in report.steps)
    eval_tokens = 4 * len(report.checkpoints)
    assert report.train_cost_usd == pytest.approx((train_tokens * 0.44 + eval_tokens * 0.195) / 1e6)


async def test_a_mid_stream_failure_leaves_the_sink_with_only_the_drained_steps(
    sdk: ModuleType, tmp_path: Path
) -> None:
    instance = FakeService("sk-tinker-test", fail_fb=3)
    sdk.ServiceClient = lambda api_key: instance
    run = sink(tmp_path)
    envelope = budget()
    request = spec(
        corpus(tmp_path, method="sft", rows=8), hyperparams=Hyperparams(steps=5, batch_size=2, max_seq_len=64)
    )

    with pytest.raises(Boom):
        await TinkerBackend.from_settings().fit(request, sink=run, budget=envelope)

    drained = step_records(run.path)
    assert [record["step"] for record in drained] == [1, 2]
    assert checkpoint_events(run.path) == []
    assert envelope.reserved == pytest.approx(0.0)
    assert envelope.spent == pytest.approx(sum(record["tokens"] for record in drained) / 1e6 * 0.44)
    await envelope.check(envelope.max_usd - envelope.spent)


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

    records = step_records(run.path)
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
    prefilled = tinker.token_count(reference.forward[0])
    trained = 2 * sum(tinker.token_count(call.datums) for call in policy.custom)
    assert checkpoint.train_cost_usd == pytest.approx((prefilled * 0.195 + trained * 0.44) / 1e6)


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


def test_dpo_loss_is_the_log_sigmoid_of_the_reference_corrected_margin() -> None:
    torch = pytest.importorskip("torch")
    logprobs = [torch.tensor([-1.0], requires_grad=True), torch.tensor([-2.0], requires_grad=True)]

    loss, metrics = dpo_loss([make_datum(1), make_datum(1)], logprobs, reference=[-1.5, -1.5], beta=0.1)

    margin = 0.1 * ((-1.0 - -1.5) - (-2.0 - -1.5))
    assert margin == pytest.approx(0.1)
    assert loss.item() == pytest.approx(math.log1p(math.exp(-margin)), rel=1e-5)
    assert metrics["margin"] == pytest.approx(margin, rel=1e-5)
    assert metrics["accuracy"] == 1.0


def test_a_better_than_reference_chosen_lowers_the_dpo_loss() -> None:
    torch = pytest.importorskip("torch")

    def loss_for(chosen: float) -> float:
        logprobs = [torch.tensor([chosen], requires_grad=True), torch.tensor([-2.0], requires_grad=True)]
        return dpo_loss([make_datum(1), make_datum(1)], logprobs, reference=[-1.5, -1.5], beta=0.1)[0].item()

    assert loss_for(-0.5) < loss_for(-1.0) < loss_for(-1.6)
    assert loss_for(-1.5) == pytest.approx(math.log1p(math.exp(-0.1 * 0.5)), rel=1e-5)


def test_the_surrogate_weights_push_chosen_up_and_rejected_down() -> None:
    torch = pytest.importorskip("torch")
    logprobs = [torch.tensor([-1.0, -1.0], requires_grad=True), torch.tensor([-1.0, -1.0], requires_grad=True)]

    loss, metrics = dpo_loss([make_datum(2), make_datum(2)], logprobs, reference=[-1.0, -1.0], beta=BETA)
    loss.backward()

    assert loss.item() == pytest.approx(-math.log(0.5), rel=1e-5)
    assert metrics["accuracy"] == 0.0
    chosen, rejected = (-logprob.grad for logprob in logprobs)
    assert all(weight > 0 for weight in chosen.tolist())
    assert all(weight < 0 for weight in rejected.tolist())
    assert chosen.tolist() == pytest.approx([-weight for weight in rejected.tolist()])


def test_score_sequence_carries_the_exact_fractional_weight_mass() -> None:
    scored = score_sequence([0.0, -1.0, -2.0], [0.0, 0.25, 0.25])

    assert scored.weight == 0.5


async def test_materialize_converts_any_saved_checkpoint_without_fusing(
    service: FakeService, converged: dict[str, object], tmp_path: Path
) -> None:
    request = spec(corpus(tmp_path, method="sft"))
    saved = SavedCheckpoint(step=2, path="tinker://run/watcher-step00002", final=False, scores=None)

    adapter = await TinkerBackend.from_settings().materialize(saved, request, work_dir=tmp_path / "run", cost=1.25)

    assert adapter == Adapter(
        step=2,
        adapter_dir=tmp_path / "run" / "adapter",
        train_cost_usd=1.25,
        sampler_path="tinker://run/watcher-step00002",
    )
    assert converged["tinker_path"] == saved.path
    assert converged["peft"] == tmp_path / "run" / "peft"
    assert converged["convert_from"] == tmp_path / "run" / "peft"
    assert "fuse_from" not in converged


async def test_fuse_preserves_the_materialized_adapter_step(converged: dict[str, object], tmp_path: Path) -> None:
    request = spec(corpus(tmp_path, method="sft"))
    adapter = Adapter(
        step=2, adapter_dir=tmp_path / "adapter", train_cost_usd=1.25, sampler_path="tinker://run/watcher-step2"
    )

    checkpoint = await TinkerBackend.from_settings().fuse(adapter, request, work_dir=tmp_path / "run")

    assert checkpoint.step == adapter.step
    assert checkpoint.adapter_dir == adapter.adapter_dir
    assert checkpoint.mlx_path == tmp_path / "run" / "mlx"
    assert checkpoint.train_cost_usd == adapter.train_cost_usd
    assert checkpoint.sampler_path == adapter.sampler_path
    assert converged["fuse_from"] == adapter.adapter_dir


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
    assert checkpoint.sampler_path == "tinker://run/watcher-sft-3"
    assert checkpoint.base == BASE_MODELS["qwen3-8b"]


async def test_score_preserves_row_order_and_reduces_the_shifted_token_positions(
    sampling_service: FakeSamplingService,
) -> None:
    rows = (
        EvalRow(tokens=(1, 2, 3, 4), weights=(7.0, 0.0, 2.0, 1.0)),
        EvalRow(tokens=(9, 8, 7), weights=(4.0, 1.0, 3.0)),
    )
    first, second = (row.tokens for row in rows)
    sampling_service.sampling.outputs = {
        first: [None, -0.1, -0.2, -0.3],
        second: [None, -0.4, -0.5],
    }
    sampling_service.sampling.slow = first
    sampling_service.sampling.release = anyio.Event()

    scores = await TinkerBackend.from_settings().score(
        "tinker://run/step2", rows, base=BASE_MODELS["qwen3-8b"], budget=budget()
    )

    assert sampling_service.sampling.finished == [second, first]
    assert sampling_service.sampling_paths == ["tinker://run/step2"]
    assert [score.logprob for score in scores] == pytest.approx([-0.7, -1.9])
    assert [score.weight for score in scores] == [3.0, 4.0]


async def test_fit_and_post_hoc_score_have_identical_weighted_reductions(
    sampling_service: FakeSamplingService, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def positioned_logprobs(
        datums: Sequence[tinker_fakes.FakeDatum],
    ) -> list[dict[str, tinker_fakes.FakeTensor]]:
        outputs = []
        for datum in datums:
            length = len(datum.loss_fn_inputs["target_tokens"].data)
            outputs.append(
                {
                    "logprobs": tinker_fakes.FakeTensor(
                        data=[-(index + 1) / 10 for index in range(length)],
                        dtype="float32",
                        shape=[length],
                    )
                }
            )
        return outputs

    monkeypatch.setattr(tinker_fakes, "logprobs_for", positioned_logprobs)
    row = EvalRow(tokens=(10, 20, 30, 40), weights=(9.0, 0.0, 2.0, 1.0))
    sampling_service.sampling.outputs[row.tokens] = [None, -0.1, -0.2, -0.3]
    backend = TinkerBackend.from_settings()
    request = spec(corpus(tmp_path, method="sft"))

    report = await backend.fit(request, sink=sink(tmp_path), budget=budget(), eval_rows=(row,))
    scores = await backend.score(report.final.path, (row,), base=request.base, budget=budget())

    assert report.final.scores is not None
    assert scores == report.final.scores
    assert scores[0].logprob == pytest.approx(-0.7)
    assert scores[0].weight == 3.0


async def test_a_shared_envelope_draws_fit_actuals_down_against_a_later_score(
    sampling_service: FakeSamplingService, tmp_path: Path
) -> None:
    backend = TinkerBackend.from_settings()
    request = spec(corpus(tmp_path, method="sft"))
    rows = (EvalRow(tokens=(1, 2, 3), weights=(0.0, 0.0, 1.0)),)

    probe = SpendGuard(max_usd=None)
    report = await backend.fit(request, sink=sink(tmp_path), budget=probe)
    fit_spent = probe.spent
    score_projection = backend.cost(model=tinker_model(request.base), prefill=3, sample=1)
    assert fit_spent > 0.0

    shared = SpendGuard(max_usd=fit_spent + score_projection / 2)
    await backend.fit(request, sink=sink(tmp_path), budget=shared)
    with pytest.raises(SpendExceeded):
        await backend.score(report.final.path, rows, base=request.base, budget=shared)

    fresh = SpendGuard(max_usd=fit_spent + score_projection / 2)
    assert len(await backend.score(report.final.path, rows, base=request.base, budget=fresh)) == 1


async def test_a_transient_score_failure_on_a_shared_envelope_releases_its_reservation(
    sampling_service: FakeSamplingService, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend = TinkerBackend.from_settings()
    request = spec(corpus(tmp_path, method="sft"))
    rows = (EvalRow(tokens=(1, 2, 3), weights=(0.0, 0.0, 1.0)),)

    shared = SpendGuard(max_usd=None)
    report = await backend.fit(request, sink=sink(tmp_path), budget=shared)
    fit_spent = shared.spent
    assert fit_spent > 0.0
    assert shared.reserved == pytest.approx(0.0)

    fail = {"on": True}
    real_client = FakeSamplingService.create_sampling_client_async

    async def flaky_client(instance: FakeSamplingService, **kwargs: object) -> FakeSampling:
        if fail["on"]:
            raise Boom("sampling client unavailable")
        return await real_client(instance, **kwargs)

    monkeypatch.setattr(FakeSamplingService, "create_sampling_client_async", flaky_client)

    with pytest.raises(Boom):
        await backend.score(report.final.path, rows, base=request.base, budget=shared)
    assert shared.reserved == pytest.approx(0.0)
    assert shared.spent == pytest.approx(fit_spent)

    fail["on"] = False
    retry = await backend.score(report.final.path, rows, base=request.base, budget=shared)
    assert len(retry) == 1
    assert shared.reserved == pytest.approx(0.0)
    assert shared.spent == pytest.approx(
        fit_spent + backend.cost(model=tinker_model(request.base), prefill=3, sample=1)
    )


async def test_a_transient_sample_failure_on_a_shared_envelope_releases_its_reservation(
    sampling_service: FakeSamplingService, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend = TinkerBackend.from_settings()
    request = spec(corpus(tmp_path, method="sft"))
    prompts = [[{"role": "user", "content": "one"}]]

    shared = SpendGuard(max_usd=None)
    report = await backend.fit(request, sink=sink(tmp_path), budget=shared)
    fit_spent = shared.spent
    assert fit_spent > 0.0

    fail = {"on": True}
    real_sample = FakeSampling.sample_async

    async def flaky_sample(
        instance: FakeSampling, prompt: FakeModelInput, num_samples: int, sampling_params: FakeSamplingParams
    ) -> FakeSampleResponse:
        if fail["on"]:
            raise Boom("sampler unavailable")
        return await real_sample(instance, prompt, num_samples, sampling_params)

    monkeypatch.setattr(FakeSampling, "sample_async", flaky_sample)

    with pytest.raises(BaseExceptionGroup) as raised:
        await backend.sample(
            report.final.path, prompts, base=request.base, budget=shared, max_tokens=8, temperature=0.7
        )
    assert raised.group_contains(Boom)
    assert shared.reserved == pytest.approx(0.0)
    assert shared.spent == pytest.approx(fit_spent)

    fail["on"] = False
    retry = await backend.sample(
        report.final.path, prompts, base=request.base, budget=shared, max_tokens=8, temperature=0.7
    )
    assert len(retry) == 1
    assert shared.reserved == pytest.approx(0.0)
    assert shared.spent == pytest.approx(fit_spent + retry[0].usd)


async def test_cancelling_a_sample_mid_gather_still_releases_its_reservation(
    sampling_service: FakeSamplingService,
) -> None:
    backend = TinkerBackend.from_settings()
    prompts = [[{"role": "user", "content": "hang"}]]
    sampling_service.sampling.slow = prompt_ids(prompts[0])
    sampling_service.sampling.release = anyio.Event()

    shared = SpendGuard(max_usd=None)
    with anyio.move_on_after(0.1):
        await backend.sample(None, prompts, base=BASE_MODELS["qwen3-8b"], budget=shared, max_tokens=8, temperature=0.5)

    assert not sampling_service.sampling.finished
    assert shared.reserved == pytest.approx(0.0)
    assert shared.spent == pytest.approx(0.0)


async def test_score_projects_full_prefill_plus_one_sample_token_per_row(
    sampling_service: FakeSamplingService,
) -> None:
    seen: dict[str, object] = {}

    class RecordingGuard:
        async def check(self, projected: float) -> None:
            seen["checked"] = projected

        async def record(self, reserved: float, actual: float) -> None:
            seen["recorded"] = (reserved, actual)

    rows = (
        EvalRow(tokens=(1, 2, 3), weights=(0.0, 0.0, 1.0)),
        EvalRow(tokens=(4, 5), weights=(0.0, 1.0)),
    )
    backend = TinkerBackend.from_settings()

    await backend.score("tinker://run/final", rows, base=BASE_MODELS["qwen3-8b"], budget=RecordingGuard())

    projected = backend.cost(model=TinkerModelId("Qwen/Qwen3-8B"), prefill=5, sample=2)
    assert seen["checked"] == pytest.approx(projected)
    assert seen["recorded"] == pytest.approx((projected, projected))
    assert sampling_service.sampling.calls == [row.tokens for row in rows]


async def test_score_rejects_an_over_cap_projection_before_creating_a_service_client(
    sdk: ModuleType,
) -> None:
    service_calls: list[str] = []

    def create_service(api_key: str) -> FakeSamplingService:
        service_calls.append(api_key)
        return FakeSamplingService(api_key)

    sdk.ServiceClient = create_service
    rows = (EvalRow(tokens=(1, 2, 3), weights=(0.0, 0.0, 1.0)),)

    with pytest.raises(SpendExceeded):
        await TinkerBackend.from_settings().score(
            "tinker://run/final", rows, base=BASE_MODELS["qwen3-8b"], budget=SpendGuard(max_usd=0.0)
        )

    assert service_calls == []


def prompt_ids(prompt: list[dict[str, str]]) -> tuple[int, ...]:
    return tuple(data.chat_ids(prompt, BASE_MODELS["qwen3-8b"].mlx, add_generation_prompt=True))


async def test_sample_preserves_prompt_order_and_seeds_each_prompt_by_index(
    sampling_service: FakeSamplingService,
) -> None:
    prompts = [[{"role": "user", "content": "one"}], [{"role": "user", "content": "two"}]]
    sampling_service.sampling.generated = {
        prompt_ids(prompts[0]): [ord("A"), ord("B")],
        prompt_ids(prompts[1]): [ord("C")],
    }
    sampling_service.sampling.slow = prompt_ids(prompts[0])
    sampling_service.sampling.release = anyio.Event()

    sampled = await TinkerBackend.from_settings().sample(
        "tinker://run/step2",
        prompts,
        base=BASE_MODELS["qwen3-8b"],
        budget=budget(),
        max_tokens=8,
        temperature=0.7,
        seed=100,
    )

    assert sampling_service.sampling.finished == [prompt_ids(prompts[1]), prompt_ids(prompts[0])]
    assert [sequence.text for sequence in sampled] == ["AB", "C"]
    assert [sequence.sampled_tokens for sequence in sampled] == [2, 1]
    assert {key: params.seed for key, _, params in sampling_service.sampling.sample_calls} == {
        prompt_ids(prompts[0]): 100,
        prompt_ids(prompts[1]): 101,
    }
    assert {params.temperature for _, _, params in sampling_service.sampling.sample_calls} == {0.7}
    assert {params.max_tokens for _, _, params in sampling_service.sampling.sample_calls} == {8}
    assert {num_samples for _, num_samples, _ in sampling_service.sampling.sample_calls} == {1}


async def test_sample_leaves_the_seed_unset_when_none_is_given(sampling_service: FakeSamplingService) -> None:
    prompts = [[{"role": "user", "content": "hi"}]]

    await TinkerBackend.from_settings().sample(
        "tinker://run/step2", prompts, base=BASE_MODELS["qwen3-8b"], budget=budget(), max_tokens=4, temperature=0.5
    )

    assert [params.seed for _, _, params in sampling_service.sampling.sample_calls] == [None]


async def test_sample_with_no_path_routes_to_a_base_model_client(sampling_service: FakeSamplingService) -> None:
    prompts = [[{"role": "user", "content": "hi"}]]

    await TinkerBackend.from_settings().sample(
        None, prompts, base=BASE_MODELS["qwen3-8b"], budget=budget(), max_tokens=4, temperature=0.5
    )

    assert sampling_service.sampling_paths == [None]
    assert sampling_service.sampling_base_models == ["Qwen/Qwen3-8B"]


async def test_sample_with_a_path_routes_to_a_model_path_client(sampling_service: FakeSamplingService) -> None:
    prompts = [[{"role": "user", "content": "hi"}]]

    await TinkerBackend.from_settings().sample(
        "tinker://run/step2", prompts, base=BASE_MODELS["qwen3-8b"], budget=budget(), max_tokens=4, temperature=0.5
    )

    assert sampling_service.sampling_paths == ["tinker://run/step2"]
    assert sampling_service.sampling_base_models == [None]


async def test_sample_rejects_an_over_cap_projection_before_creating_a_service_client(sdk: ModuleType) -> None:
    service_calls: list[str] = []

    def create_service(api_key: str) -> FakeSamplingService:
        service_calls.append(api_key)
        return FakeSamplingService(api_key)

    sdk.ServiceClient = create_service
    prompts = [[{"role": "user", "content": "hi"}]]

    with pytest.raises(SpendExceeded):
        await TinkerBackend.from_settings().sample(
            None, prompts, base=BASE_MODELS["qwen3-8b"], budget=SpendGuard(max_usd=0.0), max_tokens=4, temperature=0.5
        )

    assert service_calls == []


async def test_sample_records_the_actual_short_output_cost_below_the_projection(
    sampling_service: FakeSamplingService,
) -> None:
    seen: dict[str, object] = {}

    class RecordingGuard:
        async def check(self, projected: float) -> None:
            seen["checked"] = projected

        async def record(self, reserved: float, actual: float) -> None:
            seen["recorded"] = (reserved, actual)

    prompts = [[{"role": "user", "content": "one"}], [{"role": "user", "content": "two"}]]
    sampling_service.sampling.generated = {
        prompt_ids(prompts[0]): [ord("A"), ord("B")],
        prompt_ids(prompts[1]): [ord("C")],
    }
    backend = TinkerBackend.from_settings()

    sampled = await backend.sample(
        None, prompts, base=BASE_MODELS["qwen3-8b"], budget=RecordingGuard(), max_tokens=10, temperature=0.7
    )

    model = TinkerModelId("Qwen/Qwen3-8B")
    projected = backend.cost(model=model, prefill=sum(len(prompt_ids(prompt)) for prompt in prompts), sample=20)
    actual = sum(sequence.usd for sequence in sampled)
    assert seen["checked"] == pytest.approx(projected)
    assert seen["recorded"] == pytest.approx((projected, actual))
    assert actual < projected
    assert sampled[0].usd == pytest.approx(backend.cost(model=model, prefill=len(prompt_ids(prompts[0])), sample=2))
    assert sampled[1].usd == pytest.approx(backend.cost(model=model, prefill=len(prompt_ids(prompts[1])), sample=1))
    assert [sequence.prompt_tokens for sequence in sampled] == [len(prompt_ids(prompt)) for prompt in prompts]


async def test_sample_runs_for_a_hosted_only_base_with_a_tinker_id(sampling_service: FakeSamplingService) -> None:
    prompts = [[{"role": "user", "content": "hi"}]]

    sampled = await TinkerBackend.from_settings().sample(
        None, prompts, base=BASE_MODELS["qwen3.5-4b"], budget=budget(), max_tokens=4, temperature=0.5
    )

    assert sampling_service.sampling_base_models == ["Qwen/Qwen3.5-4B"]
    assert len(sampled) == 1


def test_tinker_model_returns_the_id_for_a_hosted_only_base() -> None:
    base = BASE_MODELS["qwen3.5-4b"]
    assert not base.serves_locally

    assert tinker_model(base) == TinkerModelId("Qwen/Qwen3.5-4B")


def test_tinker_model_refuses_a_base_with_no_tinker_id() -> None:
    base = dataclasses.replace(BASE_MODELS["qwen3-8b"], tinker=None)

    with pytest.raises(UnservableBase, match="no Tinker base model"):
        tinker_model(base)


async def test_fit_runs_for_a_hosted_only_base_with_a_tinker_id(service: FakeService, tmp_path: Path) -> None:
    request = spec(corpus(tmp_path, method="sft"), base=BASE_MODELS["qwen3.5-4b"])

    report = await TinkerBackend.from_settings().fit(request, sink=sink(tmp_path), budget=budget())

    assert service.clients[0].base_model == "Qwen/Qwen3.5-4B"
    assert (report.final.step, len(report.steps)) == (3, 3)


async def test_score_runs_for_a_hosted_only_base_with_a_tinker_id(sampling_service: FakeSamplingService) -> None:
    rows = (EvalRow(tokens=(1, 2, 3), weights=(0.0, 1.0, 2.0)),)

    scores = await TinkerBackend.from_settings().score(
        "tinker://run/step2", rows, base=BASE_MODELS["qwen3.5-4b"], budget=budget()
    )

    assert sampling_service.sampling_paths == ["tinker://run/step2"]
    assert len(scores) == 1


async def test_train_refuses_a_hosted_only_base_before_any_billable_call(service: FakeService, tmp_path: Path) -> None:
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
