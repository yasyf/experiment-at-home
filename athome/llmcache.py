from __future__ import annotations

import json
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from random import uniform
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import anyio
import httpx

from athome.config import AthomeSettings, SectionSettings, load
from athome.errors import AthomeError
from athome.store import Store

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key TEXT PRIMARY KEY,
    status INTEGER NOT NULL,
    headers TEXT NOT NULL,
    body BLOB NOT NULL,
    created_at TEXT NOT NULL
);
"""
SKIP_HEADERS = frozenset({"content-length", "content-encoding", "transfer-encoding", "connection"})
DEFAULT_PORTS = {"http": 80, "https": 443}
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
RETRY_MAX_ATTEMPTS = 3
RETRY_BASE_DELAY = 0.5
RETRY_MAX_DELAY = 30.0
DB_PATH = ("llmcache", "llmcache.db")


class LlmCacheMode(StrEnum):
    """How :class:`CachingTransport` treats each request against the store."""

    RECORD = "record"
    REPLAY = "replay"
    REPLAY_OR_RECORD = "replay_or_record"
    BYPASS = "bypass"


class LlmCacheSettings(SectionSettings):
    """The ``[llmcache]`` section: default cache mode and key schema version."""

    section = ("llmcache",)
    mode: LlmCacheMode = LlmCacheMode.BYPASS
    schema_version: int = 1


class LlmCacheMiss(AthomeError):
    """Raised in :attr:`LlmCacheMode.REPLAY` when a request has no recorded response."""


def cache_key(request: httpx.Request, *, schema_version: int) -> str:
    """Content address for ``request``; request headers are intentionally excluded from the key (donor semantics)."""
    return sha256(
        b"\x00".join(
            (
                str(schema_version).encode(),
                request.method.encode(),
                request.url.scheme.encode(),
                request.url.host.encode(),
                str(request.url.port if request.url.port is not None else DEFAULT_PORTS[request.url.scheme]).encode(),
                request.url.path.encode(),
                urlencode(sorted(request.url.params.multi_items())).encode(),
                request.content,
            )
        )
    ).hexdigest()


def stored_headers(headers: httpx.Headers) -> list[tuple[str, str]]:
    return [(name, value) for name, value in headers.multi_items() if name.lower() not in SKIP_HEADERS]


async def load_response(store: Store, key: str, request: httpx.Request) -> httpx.Response | None:
    row = await store.fetch_one("SELECT status, headers, body FROM responses WHERE key = ?", [key])
    return (
        httpx.Response(row["status"], headers=json.loads(row["headers"]), content=row["body"], request=request)
        if row is not None
        else None
    )


def retry_delay(attempt: int, *, response: httpx.Response | None = None) -> float:
    if response is not None and (after := response.headers.get("retry-after", "")).isdigit():
        return float(after)
    return min(RETRY_MAX_DELAY, RETRY_BASE_DELAY * 2.0**attempt) * uniform(0.5, 1.0)


@dataclass(frozen=True, slots=True)
class RetryTransport(httpx.AsyncBaseTransport):
    """Retries 429/5xx/transport errors with capped exponential backoff (default 3 tries)."""

    inner: httpx.AsyncBaseTransport

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        for attempt in range(RETRY_MAX_ATTEMPTS - 1):
            try:
                response = await self.inner.handle_async_request(request)
            except httpx.TransportError:
                await anyio.sleep(retry_delay(attempt))
                continue
            if response.status_code not in RETRY_STATUSES:
                return response
            await response.aclose()
            await anyio.sleep(retry_delay(attempt, response=response))
        return await self.inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self.inner.aclose()


@dataclass(slots=True)
class CachingTransport(httpx.AsyncBaseTransport):
    """Record/replay layer keyed on sha256(schema_version, method, scheme, host, port, path, sorted-query, body).

    Headers are excluded from the key by design (auth and session noise never perturb
    replay), and only ``2xx`` responses are recorded. Streaming responses are fully
    buffered: the body is read into memory before it is stored. The sqlite store lives
    under ``load(AthomeSettings).cache_root / "llmcache" / "llmcache.db"`` and opens
    lazily on the first request, closing when the transport does.
    """

    inner: httpx.AsyncBaseTransport
    mode: LlmCacheMode
    schema_version: int
    db_path: Path
    lock: anyio.Lock = field(default_factory=anyio.Lock)
    stack: AsyncExitStack = field(default_factory=AsyncExitStack)
    store: Store | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        key = cache_key(request, schema_version=self.schema_version)
        store = await self.opened_store()
        if self.mode is not LlmCacheMode.RECORD and (hit := await load_response(store, key, request)) is not None:
            return hit
        if self.mode is LlmCacheMode.REPLAY:
            raise LlmCacheMiss(f"no cached response for {request.method} {request.url}")
        return await self.fetch_and_store(request, key, store)

    async def fetch_and_store(self, request: httpx.Request, key: str, store: Store) -> httpx.Response:
        response = await self.inner.handle_async_request(request)
        try:
            body = await response.aread()
        finally:
            await response.aclose()
        headers = stored_headers(response.headers)
        if response.status_code // 100 == 2:
            await store.execute(
                "INSERT OR REPLACE INTO responses (key, status, headers, body, created_at) VALUES (?, ?, ?, ?, ?)",
                [key, response.status_code, json.dumps(headers), body, datetime.now(UTC).isoformat()],
            )
        return httpx.Response(response.status_code, headers=headers, content=body, request=request)

    async def opened_store(self) -> Store:
        if (store := self.store) is not None:
            return store
        async with self.lock:
            if (store := self.store) is None:
                store = self.store = await self.stack.enter_async_context(Store.open(self.db_path, schema=SCHEMA))
            return store

    async def aclose(self) -> None:
        await self.stack.aclose()
        await self.inner.aclose()


def transport(*, mode: LlmCacheMode | None = None) -> httpx.AsyncBaseTransport:
    """Build the record-replay transport stack for an ``httpx.AsyncClient``.

    Args:
        mode: The cache mode; ``None`` reads :attr:`LlmCacheSettings.mode`.

    Returns:
        ``CachingTransport(RetryTransport(httpx.AsyncHTTPTransport()))``, or a bare
        ``RetryTransport`` when the mode is :attr:`LlmCacheMode.BYPASS`.

    Example:
        >>> AsyncOpenAI(http_client=httpx.AsyncClient(transport=transport()))
    """
    settings = load(LlmCacheSettings)
    retry = RetryTransport(httpx.AsyncHTTPTransport())
    match mode if mode is not None else settings.mode:
        case LlmCacheMode.BYPASS:
            return retry
        case resolved:
            return CachingTransport(
                inner=retry,
                mode=resolved,
                schema_version=settings.schema_version,
                db_path=load(AthomeSettings).cache_root.joinpath(*DB_PATH),
            )
