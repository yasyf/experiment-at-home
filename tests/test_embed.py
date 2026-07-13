from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import numpy as np
import pytest

from athome import embed
from athome.embed import ApiBackend, EmbedError, EmbedIndex, LocalBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class FakeBackend:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(list(texts))
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
