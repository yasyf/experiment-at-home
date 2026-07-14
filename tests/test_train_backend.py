from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pytest

from athome.train import backend
from athome.train.backend import NoBackendAvailable, TrainBackend, select
from athome.train.spec import BASE_MODELS, Hyperparams, LocalJsonlRef, TrainSettings, TrainSpec

if TYPE_CHECKING:
    from athome.progress import RunSink
    from athome.train.spec import BackendName, Checkpoint, Method


def fake_backend(backend_name: BackendName, *, present: bool, methods: frozenset[Method]) -> type[TrainBackend]:
    class Fake:
        name: ClassVar[BackendName] = backend_name

        @staticmethod
        def available() -> bool:
            return present

        @staticmethod
        def supports(method: Method) -> bool:
            return method in methods

        @classmethod
        def from_settings(cls) -> Fake:
            return cls()

        async def train(self, spec: TrainSpec, *, sink: RunSink) -> Checkpoint:
            raise AssertionError("select must not train")

    return Fake


def install(monkeypatch: pytest.MonkeyPatch, *backends: type[TrainBackend]) -> None:
    monkeypatch.setattr(backend, "backends", lambda: backends)


def tinker(*, present: bool = True) -> type[TrainBackend]:
    return fake_backend("tinker", present=present, methods=frozenset({"sft", "dpo"}))


def local(*, present: bool = True) -> type[TrainBackend]:
    return fake_backend("local", present=present, methods=frozenset({"sft"}))


def modal(*, present: bool = True) -> type[TrainBackend]:
    return fake_backend("modal", present=present, methods=frozenset({"sft", "dpo"}))


def spec(**overrides: object) -> TrainSpec:
    return TrainSpec(
        name="watcher",
        base=BASE_MODELS["qwen3-8b"],
        dataset=LocalJsonlRef(path=Path("corpus.jsonl")),
        hyperparams=Hyperparams(steps=10),
        **overrides,
    )


def test_selects_tinker_first_when_everything_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, tinker(), local(), modal())
    assert select(spec(), TrainSettings()).name == "tinker"


def test_falls_through_to_the_next_available_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, tinker(present=False), local(), modal())
    assert select(spec(), TrainSettings()).name == "local"


def test_dpo_skips_the_sft_only_local_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, tinker(present=False), local(), modal())
    assert select(spec(method="dpo"), TrainSettings()).name == "modal"


def test_the_spec_override_pins_the_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, tinker(), local(), modal())
    assert select(spec(backend="modal"), TrainSettings()).name == "modal"


def test_the_settings_override_applies_when_the_spec_names_no_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, tinker(), local(), modal())
    assert select(spec(), TrainSettings(backend="local")).name == "local"


def test_the_spec_override_beats_the_settings_override(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, tinker(), local(), modal())
    assert select(spec(backend="tinker"), TrainSettings(backend="local")).name == "tinker"


def test_an_override_that_cannot_run_the_method_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, tinker(), local(), modal())
    with pytest.raises(NoBackendAvailable, match="local"):
        select(spec(method="dpo", backend="local"), TrainSettings())


def test_an_unavailable_override_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, tinker(present=False), local(), modal())
    with pytest.raises(NoBackendAvailable, match="tinker"):
        select(spec(backend="tinker"), TrainSettings())


def test_no_available_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, tinker(present=False), local(present=False), modal(present=False))
    with pytest.raises(NoBackendAvailable, match="sft"):
        select(spec(), TrainSettings())


def test_selection_never_constructs_the_backends_it_passes_over(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed: list[str] = []

    class Counting:
        name: ClassVar[BackendName] = "local"

        @staticmethod
        def available() -> bool:
            return True

        @staticmethod
        def supports(method: Method) -> bool:
            return True

        @classmethod
        def from_settings(cls) -> Counting:
            constructed.append(cls.name)
            return cls()

        async def train(self, spec: TrainSpec, *, sink: RunSink) -> Checkpoint:
            raise AssertionError("select must not train")

    install(monkeypatch, tinker(), Counting)
    assert select(spec(), TrainSettings()).name == "tinker"
    assert constructed == []


def test_a_backend_satisfies_the_protocol_at_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(tinker().from_settings(), TrainBackend)
