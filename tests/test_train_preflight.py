from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from athome.bakeoff import Arm, BakeoffSpec
from athome.train import PreflightFailure, preflight
from athome.train.backend import NoBackendAvailable
from athome.train.data import DpoExample, SftExample
from athome.train.local import LocalBackend
from athome.train.modal import ModalTrainBackend
from athome.train.spec import (
    BASE_MODELS,
    Hyperparams,
    LocalJsonlRef,
    LocalTrainSettings,
    ModalTrainSettings,
    TinkerSettings,
    TrainSettings,
    TrainSpec,
)
from athome.train.tinker import TinkerBackend

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from athome.train.backend import TrainBackend
    from athome.train.spec import Method

probes = import_module("athome.train.preflight")
ARM = Arm(name="base", base_url="http://127.0.0.1:8400/v1", model="base")
SFT = SftExample(
    prompt=({"role": "user", "content": "question"},),
    completion=({"role": "assistant", "content": "answer"},),
    id="sft",
)
DPO = DpoExample(
    prompt=({"role": "user", "content": "question"},),
    chosen=({"role": "assistant", "content": "answer"},),
    rejected=({"role": "assistant", "content": "wrong"},),
    id="dpo",
)


async def task(client: AsyncOpenAI, item: object) -> dict[str, object]:
    raise AssertionError("preflight does not run the evaluation task")


def evaluation(*, arms: tuple[Arm, ...] = (ARM,), primary_metric: str = "exact") -> BakeoffSpec:
    return BakeoffSpec(task=task, corpus=("item",), arms=arms, primary_metric=primary_metric)


def spec(*, method: Method = "sft", max_usd: float | None = None) -> TrainSpec:
    return TrainSpec(
        name="probe",
        base=BASE_MODELS["qwen3-8b"],
        dataset=LocalJsonlRef(path=Path("corpus.jsonl")),
        hyperparams=Hyperparams(steps=10),
        method=method,
        max_usd=max_usd,
    )


def local() -> LocalBackend:
    return LocalBackend(LocalTrainSettings())


def modal() -> ModalTrainBackend:
    return ModalTrainBackend(ModalTrainSettings())


def tinker() -> TinkerBackend:
    return TinkerBackend(TinkerSettings(TINKER_API_KEY="secret"))


def choose(monkeypatch: pytest.MonkeyPatch, backend: TrainBackend) -> None:
    monkeypatch.setattr(probes, "select", lambda spec, settings: backend)


def normalized(monkeypatch: pytest.MonkeyPatch, *examples: SftExample | DpoExample) -> None:
    async def fake_normalize(source: object, *, method: Method) -> list[SftExample | DpoExample]:
        return list(examples)

    monkeypatch.setattr(probes, "normalize", fake_normalize)


@pytest.mark.parametrize(
    "message",
    (
        "backend 'modal' cannot run method 'sft' (available=False, supports=True)",
        "backend 'local' cannot run method 'dpo' (available=True, supports=False)",
    ),
)
async def test_backend_selection_failure_is_a_preflight_failure(monkeypatch: pytest.MonkeyPatch, message: str) -> None:
    def unavailable(spec: TrainSpec, settings: TrainSettings) -> TrainBackend:
        raise NoBackendAvailable(message)

    monkeypatch.setattr(probes, "select", unavailable)

    with pytest.raises(PreflightFailure, match=message.replace("(", r"\(").replace(")", r"\)")) as caught:
        await preflight(spec(), evaluation=evaluation(), settings=TrainSettings())

    assert isinstance(caught.value.__cause__, NoBackendAvailable)


@pytest.mark.parametrize(
    ("backend", "method", "example", "expected"),
    (
        pytest.param(local(), "sft", SFT, "mlx", id="local-sft"),
        pytest.param(tinker(), "sft", SFT, "tinker-sft", id="tinker-sft"),
        pytest.param(tinker(), "dpo", DPO, "tinker-dpo", id="tinker-dpo"),
        pytest.param(modal(), "sft", SFT, "trl-sft", id="modal-sft"),
        pytest.param(modal(), "dpo", DPO, "trl-dpo", id="modal-dpo"),
    ),
)
async def test_selected_backend_uses_its_training_renderer(
    monkeypatch: pytest.MonkeyPatch,
    backend: TrainBackend,
    method: Method,
    example: SftExample | DpoExample,
    expected: str,
) -> None:
    rendered: list[str] = []
    choose(monkeypatch, backend)
    normalized(monkeypatch, example)
    monkeypatch.setattr(probes, "render_mlx_jsonl", lambda examples, path, **options: rendered.append("mlx"))
    monkeypatch.setattr(probes, "render_tinker_sft", lambda example, model: rendered.append("tinker-sft"))
    monkeypatch.setattr(probes, "render_tinker_dpo", lambda example, model: rendered.append("tinker-dpo"))
    monkeypatch.setattr(probes, "render_trl", lambda examples, *, method: rendered.append(f"trl-{method}"))

    report = await preflight(spec(method=method), evaluation=evaluation(), settings=TrainSettings())

    assert rendered == [expected]
    assert report.checks[1] == "dataset renders: 1/1 examples sampled"


async def test_dataset_probe_samples_at_most_sixteen_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    rendered: list[str] = []
    choose(monkeypatch, local())
    normalized(
        monkeypatch, *(SftExample(prompt=SFT.prompt, completion=SFT.completion, id=str(index)) for index in range(18))
    )
    monkeypatch.setattr(
        probes,
        "render_mlx_jsonl",
        lambda examples, path, **options: rendered.append(examples[0].id),
    )

    report = await preflight(spec(), evaluation=evaluation(), settings=TrainSettings())

    assert rendered == [str(index) for index in range(16)]
    assert report.checks[1] == "dataset renders: 16/18 examples sampled"


@pytest.mark.parametrize(
    ("method", "backend"),
    (
        pytest.param("sft", local(), id="sft"),
        pytest.param("dpo", tinker(), id="dpo"),
    ),
)
async def test_empty_dataset_is_a_preflight_failure(
    monkeypatch: pytest.MonkeyPatch, method: Method, backend: TrainBackend
) -> None:
    choose(monkeypatch, backend)
    normalized(monkeypatch)

    with pytest.raises(PreflightFailure) as caught:
        await preflight(spec(method=method), evaluation=evaluation(), settings=TrainSettings())

    assert str(caught.value) == "dataset is empty after normalize"


async def test_renderer_failure_names_the_sampled_row_index(monkeypatch: pytest.MonkeyPatch) -> None:
    choose(monkeypatch, tinker())
    normalized(
        monkeypatch,
        *(SftExample(prompt=SFT.prompt, completion=SFT.completion, id=str(index)) for index in range(4)),
    )

    def render(example: SftExample, model: str) -> None:
        if example.id == "2":
            raise ValueError("invalid chat template")

    monkeypatch.setattr(probes, "render_tinker_sft", render)

    with pytest.raises(PreflightFailure, match="dataset row 2 failed to render: invalid chat template"):
        await preflight(spec(), evaluation=evaluation(), settings=TrainSettings())


async def test_modal_projection_over_the_effective_cap_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    modal_events: list[str] = []

    class ForbiddenApp:
        def __init__(self, *args: object, **kwargs: object) -> None:
            modal_events.append("app")

        def run(self) -> object:
            modal_events.append("run")
            raise AssertionError("app.run must not be reached before preflight passes")

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(App=ForbiddenApp))
    choose(monkeypatch, modal())
    normalized(monkeypatch, SFT)
    monkeypatch.setattr(probes, "render_trl", lambda examples, *, method: None)

    with pytest.raises(PreflightFailure, match=r"modal projected cost \$.* exceeds cap \$0.0000"):
        await preflight(spec(max_usd=0.0), evaluation=evaluation(), settings=TrainSettings())

    assert modal_events == []


async def test_modal_projection_is_skipped_for_a_non_modal_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    choose(monkeypatch, local())
    normalized(monkeypatch, SFT)

    report = await preflight(spec(), evaluation=evaluation(), settings=TrainSettings())

    assert report.checks[2] == "modal cost projection: skipped (backend != modal)"


@pytest.mark.parametrize(
    ("bakeoff", "message"),
    (
        pytest.param(evaluation(arms=()), "evaluation requires at least one arm", id="no-arms"),
        pytest.param(evaluation(primary_metric=""), "evaluation requires a primary metric", id="no-primary-metric"),
    ),
)
async def test_evaluation_sanity_is_mandatory(
    monkeypatch: pytest.MonkeyPatch, bakeoff: BakeoffSpec, message: str
) -> None:
    choose(monkeypatch, local())
    normalized(monkeypatch, SFT)

    with pytest.raises(PreflightFailure, match=message):
        await preflight(spec(), evaluation=bakeoff, settings=TrainSettings())


async def test_inaccessible_baseline_store_is_a_preflight_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    choose(monkeypatch, local())
    normalized(monkeypatch, SFT)

    with pytest.raises(PreflightFailure, match=f"baseline store inaccessible at {tmp_path}"):
        await preflight(spec(), evaluation=evaluation(), settings=TrainSettings(baseline_root=tmp_path))
