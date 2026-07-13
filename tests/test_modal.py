from __future__ import annotations

import importlib.metadata
import sys
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING

import anyio
import pytest

from athome.config import load
from athome.modal import (
    ModalSettings,
    ModalWorkerTransport,
    ParityMismatch,
    ServiceSpec,
    fingerprint_for,
    image_recipe,
    parity_mismatches,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

VERSIONS = {"onnxruntime": "1.19.0", "rapidocr": "1.4.0"}
SPEC = ServiceSpec("ocr-paddle", ("onnxruntime", "rapidocr"), {"upscale": 2})
MATCHING_FINGERPRINT = {"onnxruntime": "1.19.0", "rapidocr": "1.4.0", "upscale": 2}
CLASS_NAME = "PaddleRemote"


class ModalUnreachable(Exception):
    """Stand-in for a modal auth / NotFound failure raised by an unhydratable handle."""


class FakeInstance:
    def __init__(
        self,
        *,
        fingerprint: dict[str, object],
        methods: dict[str, Callable[..., object]] | None = None,
        fingerprint_error: Exception | None = None,
    ) -> None:
        self.fp = fingerprint
        self.fp_error = fingerprint_error
        self.methods = methods or {}
        self.fingerprint_count = 0
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def __getattr__(self, name: str) -> SimpleNamespace:
        if name.startswith("__"):
            raise AttributeError(name)

        async def aio(*args: object) -> object:
            await anyio.sleep(0)
            if name == "fingerprint":
                self.fingerprint_count += 1
                if self.fp_error is not None:
                    raise self.fp_error
                return self.fp
            self.calls.append((name, args))
            return self.methods[name](*args)

        return SimpleNamespace(remote=SimpleNamespace(aio=aio))


def install_fake_modal(monkeypatch: pytest.MonkeyPatch, instance: FakeInstance, record: dict[str, object]) -> None:
    class Handle:
        def __call__(self) -> FakeInstance:
            return instance

    class Cls:
        @staticmethod
        def from_name(app_name: str, tag: str) -> Handle:
            record["from_name"] = (app_name, tag)
            return Handle()

    module = ModuleType("modal")
    module.Cls = Cls
    monkeypatch.setitem(sys.modules, "modal", module)


@pytest.fixture
def pinned_versions(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(importlib.metadata, "version", VERSIONS.__getitem__)
    yield


def test_fingerprint_for_folds_versions_and_params(pinned_versions: None) -> None:
    assert fingerprint_for(SPEC) == MATCHING_FINGERPRINT


@pytest.mark.parametrize(
    ("remote", "expected_substrings"),
    [
        pytest.param(MATCHING_FINGERPRINT, [], id="exact-match"),
        pytest.param({**MATCHING_FINGERPRINT, "rapidocr": "1.3.0"}, ["rapidocr"], id="version-skew"),
        pytest.param({"onnxruntime": "1.19.0", "rapidocr": "1.4.0"}, ["upscale"], id="param-missing"),
        pytest.param({**MATCHING_FINGERPRINT, "upscale": 4}, ["upscale"], id="param-skew"),
    ],
)
def test_parity_mismatches(pinned_versions: None, remote: dict[str, object], expected_substrings: list[str]) -> None:
    mismatches = parity_mismatches(SPEC, remote)
    assert len(mismatches) == len(expected_substrings)
    for substring in expected_substrings:
        assert any(substring in line for line in mismatches)


async def test_first_call_asserts_parity_before_ready(pinned_versions: None, monkeypatch: pytest.MonkeyPatch) -> None:
    instance = FakeInstance(fingerprint=MATCHING_FINGERPRINT, methods={"tokens": lambda payload: ("ok", payload)})
    record: dict[str, object] = {}
    install_fake_modal(monkeypatch, instance, record)
    transport = ModalWorkerTransport(SPEC, CLASS_NAME)

    assert transport.remote is None
    assert await transport.call("tokens", b"img") == ("ok", b"img")

    assert transport.remote is instance
    assert instance.fingerprint_count == 1
    assert instance.calls == [("tokens", (b"img",))]
    assert record["from_name"] == ("athome-ocr-paddle", CLASS_NAME)


async def test_parity_mismatch_raises_and_stays_unready(pinned_versions: None, monkeypatch: pytest.MonkeyPatch) -> None:
    instance = FakeInstance(
        fingerprint={**MATCHING_FINGERPRINT, "rapidocr": "1.3.0"},
        methods={"tokens": lambda payload: payload},
    )
    install_fake_modal(monkeypatch, instance, {})
    transport = ModalWorkerTransport(SPEC, CLASS_NAME)

    with pytest.raises(ParityMismatch) as excinfo:
        await transport.call("tokens", b"img")

    assert "rapidocr" in str(excinfo.value)
    assert "athome-ocr-paddle" in str(excinfo.value)
    assert transport.remote is None
    assert instance.calls == []


async def test_unreachable_raises_with_no_fallback(pinned_versions: None, monkeypatch: pytest.MonkeyPatch) -> None:
    instance = FakeInstance(
        fingerprint=MATCHING_FINGERPRINT,
        methods={"tokens": lambda payload: payload},
        fingerprint_error=ModalUnreachable("no token / undeployed app"),
    )
    install_fake_modal(monkeypatch, instance, {})
    transport = ModalWorkerTransport(SPEC, CLASS_NAME)

    with pytest.raises(ModalUnreachable):
        await transport.call("tokens", b"img")

    assert transport.remote is None
    assert instance.calls == []


async def test_hydrates_once_across_calls(pinned_versions: None, monkeypatch: pytest.MonkeyPatch) -> None:
    instance = FakeInstance(fingerprint=MATCHING_FINGERPRINT, methods={"tokens": lambda payload: payload * 2})
    install_fake_modal(monkeypatch, instance, {})
    transport = ModalWorkerTransport(SPEC, CLASS_NAME)

    assert await transport.call("tokens", "a") == "aa"
    assert await transport.call("tokens", "b") == "bb"
    assert instance.fingerprint_count == 1


async def test_concurrent_first_calls_hydrate_once(pinned_versions: None, monkeypatch: pytest.MonkeyPatch) -> None:
    instance = FakeInstance(fingerprint=MATCHING_FINGERPRINT, methods={"tokens": lambda payload: payload})
    install_fake_modal(monkeypatch, instance, {})
    transport = ModalWorkerTransport(SPEC, CLASS_NAME)
    results: dict[int, object] = {}

    async def one(index: int) -> None:
        results[index] = await transport.call("tokens", index)

    async with anyio.create_task_group() as group:
        for index in range(8):
            group.start_soon(one, index)

    assert instance.fingerprint_count == 1
    assert results == {index: index for index in range(8)}


async def test_aclose_drops_remote(pinned_versions: None, monkeypatch: pytest.MonkeyPatch) -> None:
    instance = FakeInstance(fingerprint=MATCHING_FINGERPRINT, methods={"tokens": lambda payload: payload})
    install_fake_modal(monkeypatch, instance, {})
    transport = ModalWorkerTransport(SPEC, CLASS_NAME)

    await transport.call("tokens", b"img")
    assert transport.remote is instance
    await transport.aclose()
    assert transport.remote is None


def test_app_name_default_prefix() -> None:
    assert ModalWorkerTransport(SPEC, CLASS_NAME).app_name == "athome-ocr-paddle"


def test_app_name_honors_settings_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATHOME_MODAL_APP_PREFIX", "acme")
    load.cache_clear()
    assert load(ModalSettings).app_prefix == "acme"
    assert ModalWorkerTransport(SPEC, CLASS_NAME).app_name == "acme-ocr-paddle"


def test_image_recipe_builds_baked_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    log: list[tuple[object, ...]] = []

    class ImageChain:
        def debian_slim(self, *, python_version: str) -> ImageChain:
            log.append(("debian_slim", python_version))
            return self

        def uv_sync(self) -> ImageChain:
            log.append(("uv_sync",))
            return self

        def add_local_python_source(self, source: str, *, copy: bool) -> ImageChain:
            log.append(("add_local_python_source", source, copy))
            return self

        def run_function(self, fn: Callable[[], None]) -> ImageChain:
            log.append(("run_function", fn))
            return self

    chain = ImageChain()
    module = ModuleType("modal")
    module.Image = chain
    monkeypatch.setitem(sys.modules, "modal", module)

    def download_models() -> None: ...

    result = image_recipe(SPEC, python="3.13", local_source="ppocr_worker", download_models=download_models)

    assert result is chain
    assert log == [
        ("debian_slim", "3.13"),
        ("uv_sync",),
        ("add_local_python_source", "ppocr_worker", True),
        ("run_function", download_models),
    ]


@pytest.mark.live
async def test_live_deployed_toy_app() -> None:
    spec = ServiceSpec("echo", ("modal",), {})
    transport = ModalWorkerTransport(spec, "Echo")
    try:
        assert await transport.call("echo", b"ping") == b"ping"
    finally:
        await transport.aclose()
