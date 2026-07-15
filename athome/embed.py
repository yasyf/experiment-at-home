from __future__ import annotations

import functools
import hashlib
import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

import httpx
from anyio import Lock, Semaphore, create_task_group, to_thread
from pydantic import Field, SecretStr

from athome.cache import Cache
from athome.config import SectionSettings, load
from athome.errors import AthomeError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np
    from sentence_transformers import SentenceTransformer

INDEX_VERSION = 1
INDEX_KEY = "index"
EMBED_TIMEOUT_S = 60.0
NAMESPACE_LOCKS: dict[str, Lock] = {}


def namespace_lock(namespace: str) -> Lock:
    if namespace not in NAMESPACE_LOCKS:
        NAMESPACE_LOCKS[namespace] = Lock()
    return NAMESPACE_LOCKS[namespace]


class EmbedError(AthomeError):
    """An embedding backend or index operation failed."""


class EmbedSettings(SectionSettings):
    """The ``[embed]`` section: the bearer key for OpenAI-compatible embedding endpoints."""

    section: ClassVar[tuple[str, ...]] = ("embed",)
    api_key: str = "local"


class VoyageSettings(SectionSettings):
    """The ``[embed.voyage]`` section: the Voyage AI key, model, and batching budgets.

    Bound to ``ATHOME_EMBED_VOYAGE_*``; ``api_key`` reads the canonical ``VOYAGE_API_KEY``
    so a consumer's existing convention works unchanged. ``batch_texts`` and ``batch_chars``
    cap each request by item count and total characters, and ``concurrency`` bounds the
    in-flight requests per :meth:`VoyageEmbedBackend.embed` call.
    """

    section: ClassVar[tuple[str, ...]] = ("embed", "voyage")
    api_key: SecretStr = Field(validation_alias="VOYAGE_API_KEY")
    model: str = "voyage-4-large"
    max_retries: int = 3
    timeout: float = 120
    batch_texts: int = 256
    batch_chars: int = 240_000
    concurrency: int = 16


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def embed_client(base_url: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=base_url, timeout=EMBED_TIMEOUT_S)


@functools.cache
def load_transformer(model: str) -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model)


async def read_index(cache: Cache) -> dict[str, tuple[str, np.ndarray]]:
    import numpy as np

    if (data := await cache.get_bytes(cache.key(INDEX_KEY))) is None:
        return {}
    npz = np.load(io.BytesIO(data), allow_pickle=False)
    return {
        str(entry_id): (str(digest), npz["vectors"][row])
        for row, (entry_id, digest) in enumerate(zip(npz["ids"].tolist(), npz["digests"].tolist(), strict=True))
    }


async def write_index(cache: Cache, entries: Mapping[str, tuple[str, np.ndarray]]) -> None:
    import numpy as np

    ids = list(entries)
    buffer = io.BytesIO()
    np.savez(
        buffer,
        ids=np.array(ids, dtype=np.str_),
        digests=np.array([entries[entry_id][0] for entry_id in ids], dtype=np.str_),
        vectors=stack_vectors(entries),
    )
    await cache.put_bytes(cache.key(INDEX_KEY), buffer.getvalue())


def stack_vectors(entries: Mapping[str, tuple[str, np.ndarray]]) -> np.ndarray:
    import numpy as np

    if not entries:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack([vector for _, vector in entries.values()]).astype(np.float32)


class EmbedBackend(Protocol):
    """A batch text-embedding backend producing an ``(n, dim)`` float32 matrix."""

    async def embed(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class ApiBackend:
    """An OpenAI-compatible ``/embeddings`` endpoint as an embedding backend.

    Example:
        >>> await ApiBackend("http://127.0.0.1:8400/v1", "text-embedding-3-small").embed(["hi"])
    """

    base_url: str
    model: str

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        import numpy as np

        async with embed_client(self.base_url) as client:
            response = await client.post(
                f"{self.base_url.rstrip('/')}/embeddings",
                json={"model": self.model, "input": list(texts)},
                headers={"Authorization": f"Bearer {load(EmbedSettings).api_key}"},
            )
        if response.is_error:
            raise EmbedError(f"embeddings endpoint {self.base_url} returned {response.status_code}")
        rows = sorted(response.json()["data"], key=lambda row: row["index"])
        return np.asarray([row["embedding"] for row in rows], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class LocalBackend:
    """A local sentence-transformers model as an embedding backend (lazy import).

    Example:
        >>> await LocalBackend().embed(["hi"])
    """

    model: str = "all-MiniLM-L6-v2"

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        import numpy as np

        return np.asarray(
            await to_thread.run_sync(
                lambda: load_transformer(self.model).encode(
                    list(texts), normalize_embeddings=True, convert_to_numpy=True
                )
            ),
            dtype=np.float32,
        )


def pack_batches(texts: Sequence[str], *, max_texts: int, max_chars: int) -> list[list[str]]:
    batches: list[list[str]] = []
    chars = 0
    for text in texts:
        if batches and len(batches[-1]) < max_texts and chars + len(text) <= max_chars:
            batches[-1].append(text)
            chars += len(text)
        else:
            batches.append([text])
            chars = len(text)
    return batches


@dataclass(frozen=True, slots=True)
class VoyageEmbedBackend:
    """The Voyage AI ``/embeddings`` API as an embedding backend (lazy import, ``embed-voyage`` extra).

    Packs ``texts`` into dual-budget batches — each capped by both ``settings.batch_texts`` items and
    ``settings.batch_chars`` characters — embeds the batches concurrently under ``settings.concurrency``,
    and reassembles the vectors in the original input order. ``input_type`` and ``normalize`` are
    construction state: build one instance for documents and a separate one for queries.

    Example:
        >>> await VoyageEmbedBackend.from_settings(input_type="query").embed(["hi"])
    """

    settings: VoyageSettings
    input_type: Literal["query", "document"] = "document"
    normalize: bool = True

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        import numpy as np
        from voyageai import AsyncClient
        from voyageai.error import VoyageError

        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        client = AsyncClient(
            api_key=self.settings.api_key.get_secret_value(),
            max_retries=self.settings.max_retries,
            timeout=self.settings.timeout,
        )
        batches = pack_batches(texts, max_texts=self.settings.batch_texts, max_chars=self.settings.batch_chars)
        blocks: list[np.ndarray | None] = [None] * len(batches)
        limiter = Semaphore(self.settings.concurrency)

        async def embed_batch(index: int, batch: list[str]) -> None:
            async with limiter:
                try:
                    result = await client.embed(batch, model=self.settings.model, input_type=self.input_type)
                except VoyageError as error:
                    raise EmbedError(f"voyage embeddings failed: {error}") from error
            blocks[index] = np.asarray(result.embeddings, dtype=np.float32)

        try:
            async with create_task_group() as group:
                for index, batch in enumerate(batches):
                    group.start_soon(embed_batch, index, batch)
        except BaseExceptionGroup as failures:
            raise failures.exceptions[0]
        matrix = np.concatenate(blocks)
        if not self.normalize:
            return matrix
        return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)

    @classmethod
    def from_settings(
        cls, *, input_type: Literal["query", "document"] = "document", normalize: bool = True
    ) -> VoyageEmbedBackend:
        """Build a backend from :class:`VoyageSettings`, loaded from the config file and environment."""
        return cls(load(VoyageSettings), input_type=input_type, normalize=normalize)


@dataclass(frozen=True, slots=True)
class LoadedIndex:
    ids: tuple[str, ...]
    matrix: np.ndarray


@dataclass(frozen=True, slots=True)
class EmbedIndex:
    """A content-digest-keyed incremental embedding matrix persisted under ``cache_root``.

    ``upsert`` re-embeds only the ids whose text digest changed since the last pass; the
    matrix is persisted as an ``.npz`` blob through :class:`athome.cache.Cache` atomic writes.
    Call :meth:`upsert` or :meth:`matrix` before :meth:`mmr`, which reranks the loaded matrix.

    Concurrency contract: :meth:`upsert` serialises its read-modify-write per namespace with an
    in-process :class:`anyio.Lock`, so concurrent upserts within one process never lose updates.
    That lock does not span processes, so the contract is a single writer per namespace across
    processes; a second writing process can still clobber an update. A cross-process file lock is
    intentionally out of scope — run one writer per namespace.

    Example:
        >>> index = EmbedIndex("exemplars", LocalBackend())
        >>> await index.upsert({"a": "first", "b": "second"})
        >>> index.mmr(query_vec, k=8)
    """

    namespace: str
    backend: EmbedBackend
    _loaded: list[LoadedIndex] = field(default_factory=list, init=False, repr=False, compare=False)

    async def upsert(self, items: Mapping[str, str]) -> None:
        """Insert or update ``id -> text`` entries, re-embedding only the changed digests.

        The read-modify-write is serialised per namespace by an in-process :class:`anyio.Lock`,
        so concurrent upserts within one process never lose updates. The lock does not span
        processes: keep a single writer per namespace across processes.
        """
        import numpy as np

        async with namespace_lock(self.namespace):
            cache = Cache.open(self.namespace, version=INDEX_VERSION)
            entries = await read_index(cache)
            digests = {entry_id: text_digest(text) for entry_id, text in items.items()}
            pending = {
                entry_id: text
                for entry_id, text in items.items()
                if (prior := entries.get(entry_id)) is None or prior[0] != digests[entry_id]
            }
            if pending:
                vectors = await self.backend.embed(list(pending.values()))
                for entry_id, vector in zip(pending, vectors, strict=True):
                    entries[entry_id] = (digests[entry_id], np.asarray(vector, dtype=np.float32))
            await write_index(cache, entries)
            self._loaded[:] = [snapshot(entries)]

    async def matrix(self) -> np.ndarray:
        """Load and return the full ``(n, dim)`` float32 embedding matrix."""
        self._loaded[:] = [snapshot(await read_index(Cache.open(self.namespace, version=INDEX_VERSION)))]
        return self._loaded[0].matrix

    def mmr(self, query_vec: np.ndarray, *, k: int, lambda_: float = 0.5) -> list[str]:
        """Rerank the loaded matrix against ``query_vec`` by maximal marginal relevance, returning ids.

        Greedily picks the id maximizing ``lambda_ * sim(query, id) - (1 - lambda_) * max sim(id, picked)``:
        higher ``lambda_`` favors relevance, lower favors diversity among the ``k`` returned ids.
        """
        import numpy as np

        ids, matrix = (loaded := self._loaded[0]).ids, loaded.matrix
        if matrix.shape[0] == 0 or k <= 0:
            return []
        query = (query := np.asarray(query_vec, dtype=np.float32)) / max(float(np.linalg.norm(query)), 1e-12)
        unit = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        sims = unit @ query
        pool = np.argsort(-sims).tolist()
        picked: list[int] = []
        while pool and len(picked) < k:
            if not picked:
                best = pool[0]
            else:
                scores = lambda_ * sims[pool] - (1 - lambda_) * (unit[pool] @ unit[picked].T).max(axis=1)
                best = pool[int(np.argmax(scores))]
            picked.append(best)
            pool.remove(best)
        return [ids[index] for index in picked]


def snapshot(entries: Mapping[str, tuple[str, np.ndarray]]) -> LoadedIndex:
    return LoadedIndex(tuple(entries), stack_vectors(entries))
