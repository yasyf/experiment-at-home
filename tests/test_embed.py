from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING

import anyio
import httpx
import numpy as np
import pytest

from athome import embed
from athome.config import load
from athome.embed import ApiBackend, EmbedError, EmbedIndex, LocalBackend, VoyageEmbedBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence


class FakeBackend:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.array([self.vectors[text] for text in texts], dtype=np.float32)


class SlowBackend:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        await anyio.sleep(0.05)
        return np.array([self.vectors[text] for text in texts], dtype=np.float32)


def mock_embed_client(monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]) -> None:
    monkeypatch.setattr(
        embed,
        "embed_client",
        lambda base_url: httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(handler)),
    )


async def test_upsert_reembeds_only_changed_digest() -> None:
    backend = FakeBackend({"foo": [1.0, 0.0], "bar": [0.0, 1.0], "baz": [1.0, 1.0]})
    index = EmbedIndex("ns", backend)
    await index.upsert({"x": "foo", "y": "bar"})
    await index.upsert({"x": "foo", "y": "baz"})
    assert backend.calls == [["foo", "bar"], ["baz"]]


async def test_upsert_new_id_appends_and_embeds_only_it() -> None:
    backend = FakeBackend({"foo": [1.0, 0.0], "bar": [0.0, 1.0], "qux": [1.0, 1.0]})
    index = EmbedIndex("ns", backend)
    await index.upsert({"x": "foo", "y": "bar"})
    await index.upsert({"x": "foo", "y": "bar", "z": "qux"})
    assert backend.calls == [["foo", "bar"], ["qux"]]
    assert (await index.matrix()).shape == (3, 2)


async def test_concurrent_upserts_to_one_namespace_both_survive() -> None:
    backend = SlowBackend({"foo": [1.0, 0.0], "bar": [0.0, 1.0]})
    async with anyio.create_task_group() as group:
        group.start_soon(EmbedIndex("shared", backend).upsert, {"a": "foo"})
        group.start_soon(EmbedIndex("shared", backend).upsert, {"b": "bar"})
    matrix = await EmbedIndex("shared", FakeBackend({})).matrix()
    assert matrix.shape == (2, 2)


def test_embed_index_documents_single_writer_contract() -> None:
    doc = (EmbedIndex.__doc__ or "").lower()
    assert "single writer per namespace" in doc
    assert "does not span processes" in doc


async def test_matrix_persists_across_instances_without_reembedding() -> None:
    await EmbedIndex("ns", FakeBackend({"foo": [1.0, 0.0, 0.0], "bar": [0.0, 1.0, 0.0]})).upsert(
        {"a": "foo", "b": "bar"}
    )
    fresh_backend = FakeBackend({})
    fresh = EmbedIndex("ns", fresh_backend)
    matrix = await fresh.matrix()
    assert matrix.shape == (2, 3)
    assert matrix[0].tolist() == [1.0, 0.0, 0.0]
    await fresh.upsert({"a": "foo", "b": "bar"})
    assert fresh_backend.calls == []


async def test_mmr_relevance_versus_diversity() -> None:
    backend = FakeBackend({"a": [1.0, 0.0, 0.0], "c": [0.9, 0.1, 0.0], "b": [0.0, 1.0, 0.0]})
    index = EmbedIndex("mmr", backend)
    await index.upsert({"a": "a", "c": "c", "b": "b"})
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert index.mmr(query, k=2, lambda_=1.0) == ["a", "c"]
    assert index.mmr(query, k=2, lambda_=0.3) == ["a", "b"]


async def test_mmr_empty_index_returns_empty() -> None:
    index = EmbedIndex("empty", FakeBackend({}))
    await index.matrix()
    assert index.mmr(np.array([1.0, 0.0], dtype=np.float32), k=4) == []


async def test_api_backend_posts_openai_payload_and_sorts_by_index(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={"data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]},
        )

    mock_embed_client(monkeypatch, handler)
    result = await ApiBackend("http://host/v1", "text-embed-3").embed(["p", "q"])
    assert result.shape == (2, 2)
    assert result[0].tolist() == [1.0, 0.0]
    assert result[1].tolist() == [0.0, 1.0]
    request = seen[0]
    assert request.url.path == "/v1/embeddings"
    assert json.loads(request.content) == {"model": "text-embed-3", "input": ["p", "q"]}
    assert request.headers["authorization"] == "Bearer local"


async def test_api_backend_raises_embed_error_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_embed_client(monkeypatch, lambda request: httpx.Response(500))
    with pytest.raises(EmbedError):
        await ApiBackend("http://host/v1", "m").embed(["p"])


@pytest.mark.live
async def test_local_backend_embeds() -> None:
    result = await LocalBackend().embed(["hello world"])
    assert result.shape[0] == 1
    assert result.dtype == np.dtype("float32")


class VoyageRecorder:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self.input_types: list[str] = []
        self.models: list[str] = []
        self.vectors: dict[str, list[float]] = {}
        self.client_kwargs: dict[str, object] = {}
        self.fail = False
        self.delay: Callable[[int], float] = lambda index: 0.0


@pytest.fixture
def fake_voyage(monkeypatch: pytest.MonkeyPatch) -> Iterator[VoyageRecorder]:
    monkeypatch.setenv("VOYAGE_API_KEY", "vk")
    load.cache_clear()
    recorder = VoyageRecorder()

    class VoyageError(Exception):
        pass

    class AsyncClient:
        def __init__(self, **kwargs: object) -> None:
            recorder.client_kwargs = kwargs

        async def embed(self, texts: list[str], *, model: str, input_type: str) -> SimpleNamespace:
            index = len(recorder.batches)
            recorder.batches.append(list(texts))
            recorder.input_types.append(input_type)
            recorder.models.append(model)
            await anyio.sleep(recorder.delay(index))
            if recorder.fail:
                raise VoyageError("voyage is down")
            return SimpleNamespace(embeddings=[recorder.vectors[text] for text in texts])

    module = ModuleType("voyageai")
    module.AsyncClient = AsyncClient
    error_module = ModuleType("voyageai.error")
    error_module.VoyageError = VoyageError
    module.error = error_module
    monkeypatch.setitem(sys.modules, "voyageai", module)
    monkeypatch.setitem(sys.modules, "voyageai.error", error_module)
    yield recorder


async def test_voyage_batches_by_item_count(fake_voyage: VoyageRecorder) -> None:
    fake_voyage.vectors = {f"t{n}": [float(n), 0.0] for n in range(257)}
    await VoyageEmbedBackend.from_settings().embed([f"t{n}" for n in range(257)])
    assert [len(batch) for batch in fake_voyage.batches] == [256, 1]


async def test_voyage_batches_by_char_budget(fake_voyage: VoyageRecorder) -> None:
    texts = ["a" * 100_000, "b" * 100_000, "c" * 100_000]
    fake_voyage.vectors = {text: [1.0, 0.0] for text in texts}
    await VoyageEmbedBackend.from_settings().embed(texts)
    assert [len(batch) for batch in fake_voyage.batches] == [2, 1]


async def test_voyage_reassembles_vectors_in_input_order(
    fake_voyage: VoyageRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ATHOME_EMBED_VOYAGE_BATCH_TEXTS", "1")
    load.cache_clear()
    texts = ["p", "q", "r", "s"]
    fake_voyage.vectors = {
        "p": [1.0, 0.0, 0.0, 0.0],
        "q": [0.0, 1.0, 0.0, 0.0],
        "r": [0.0, 0.0, 1.0, 0.0],
        "s": [0.0, 0.0, 0.0, 1.0],
    }
    fake_voyage.delay = lambda index: 0.02 * (len(texts) - index)  # later batches finish first
    result = await VoyageEmbedBackend.from_settings(normalize=False).embed(texts)
    assert len(fake_voyage.batches) == len(texts)  # batch_texts=1 => one batch per text
    assert result.tolist() == [fake_voyage.vectors[text] for text in texts]


@pytest.mark.parametrize("input_type", ["query", "document"])
async def test_voyage_input_type_reaches_client(fake_voyage: VoyageRecorder, input_type: str) -> None:
    fake_voyage.vectors = {"hi": [1.0, 0.0]}
    await VoyageEmbedBackend.from_settings(input_type=input_type).embed(["hi"])
    assert fake_voyage.input_types == [input_type]
    assert fake_voyage.models == ["voyage-4-large"]


@pytest.mark.parametrize(
    "normalize, expected",
    [(False, [3.0, 4.0]), (True, [0.6, 0.8])],
    ids=["raw", "unit-norm"],
)
async def test_voyage_normalize_flag(fake_voyage: VoyageRecorder, normalize: bool, expected: list[float]) -> None:
    fake_voyage.vectors = {"a": [3.0, 4.0]}
    result = await VoyageEmbedBackend.from_settings(normalize=normalize).embed(["a"])
    assert result.dtype == np.dtype("float32")
    assert np.allclose(result, [expected])


async def test_voyage_wraps_client_failure_in_embed_error(fake_voyage: VoyageRecorder) -> None:
    fake_voyage.vectors = {"a": [1.0, 0.0]}
    fake_voyage.fail = True
    with pytest.raises(EmbedError):
        await VoyageEmbedBackend.from_settings().embed(["a"])
