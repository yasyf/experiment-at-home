from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import httpx
import pytest

from athome.config import AthomeSettings, load
from athome.llmcache import (
    DB_PATH,
    CachingTransport,
    LlmCacheMiss,
    LlmCacheMode,
    RetryTransport,
    cache_key,
    transport,
)

if TYPE_CHECKING:
    from collections.abc import Callable

CHAT_URL = "https://api.openai.com/v1/chat/completions"
REQUEST_BODY = b'{"model": "gpt-x", "messages": []}'
RESPONSE_BODY = b'{"id": "cmpl-1", "choices": [1, 2, 3]}'


def db_path():
    return load(AthomeSettings).cache_root.joinpath(*DB_PATH)


def counting_origin(response: Callable[[], httpx.Response]) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request.read()
        seen.append(request)
        return response()

    return httpx.MockTransport(handler), seen


async def noop_sleep(_seconds: float) -> None:
    return None


async def test_record_then_replay_roundtrip_is_byte_equal() -> None:
    headers = {"content-type": "application/json", "x-trace": "abc"}
    mock, seen = counting_origin(lambda: httpx.Response(200, headers=headers, content=RESPONSE_BODY))

    recorder = CachingTransport(inner=mock, mode=LlmCacheMode.REPLAY_OR_RECORD, schema_version=1, db_path=db_path())
    async with httpx.AsyncClient(transport=recorder) as client:
        recorded = await client.post(CHAT_URL, content=REQUEST_BODY)
    assert recorded.status_code == 200
    assert recorded.content == RESPONSE_BODY
    assert len(seen) == 1

    replayer = CachingTransport(inner=mock, mode=LlmCacheMode.REPLAY, schema_version=1, db_path=db_path())
    async with httpx.AsyncClient(transport=replayer) as client:
        replayed = await client.post(CHAT_URL, content=REQUEST_BODY)
    assert replayed.status_code == 200
    assert replayed.content == RESPONSE_BODY
    assert replayed.headers["content-type"] == "application/json"
    assert replayed.headers["x-trace"] == "abc"
    assert len(seen) == 1


async def test_replay_miss_raises() -> None:
    mock, _ = counting_origin(lambda: httpx.Response(200, content=RESPONSE_BODY))
    replayer = CachingTransport(inner=mock, mode=LlmCacheMode.REPLAY, schema_version=1, db_path=db_path())
    async with httpx.AsyncClient(transport=replayer) as client:
        with pytest.raises(LlmCacheMiss):
            await client.post(CHAT_URL, content=REQUEST_BODY)


async def test_bypass_returns_retry_and_skips_store() -> None:
    built = transport(mode=LlmCacheMode.BYPASS)
    assert isinstance(built, RetryTransport)
    assert not isinstance(built, CachingTransport)
    assert not db_path().exists()
    await built.aclose()


async def test_transport_default_mode_is_bypass() -> None:
    built = transport()
    assert isinstance(built, RetryTransport)
    assert not isinstance(built, CachingTransport)
    await built.aclose()


async def test_transport_record_mode_builds_caching_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATHOME_LLMCACHE_MODE", "record")
    load.cache_clear()
    built = transport()
    assert isinstance(built, CachingTransport)
    assert built.mode is LlmCacheMode.RECORD
    assert isinstance(built.inner, RetryTransport)
    assert isinstance(built.inner.inner, httpx.AsyncHTTPTransport)
    await built.aclose()


async def test_retry_on_500_then_200(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(anyio, "sleep", noop_sleep)
    statuses = iter([500, 200])
    mock, seen = counting_origin(lambda: httpx.Response(next(statuses), content=b"ok"))
    retrier = RetryTransport(mock)
    async with httpx.AsyncClient(transport=retrier) as client:
        response = await client.get("https://api.openai.com/v1/models")
    assert response.status_code == 200
    assert len(seen) == 2


async def test_retry_on_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(anyio, "sleep", noop_sleep)
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, content=b"ok")

    retrier = RetryTransport(httpx.MockTransport(handler))
    async with httpx.AsyncClient(transport=retrier) as client:
        response = await client.get("https://api.openai.com/v1/models")
    assert response.status_code == 200
    assert attempts["n"] == 2


async def test_retry_exhausts_and_returns_last_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(anyio, "sleep", noop_sleep)
    mock, seen = counting_origin(lambda: httpx.Response(503, content=b"down"))
    retrier = RetryTransport(mock)
    async with httpx.AsyncClient(transport=retrier) as client:
        response = await client.get("https://api.openai.com/v1/models")
    assert response.status_code == 503
    assert len(seen) == 3


async def test_non_2xx_never_recorded() -> None:
    mock, seen = counting_origin(lambda: httpx.Response(404, content=b"nope"))
    recorder = CachingTransport(inner=mock, mode=LlmCacheMode.RECORD, schema_version=1, db_path=db_path())
    async with httpx.AsyncClient(transport=recorder) as client:
        response = await client.post(CHAT_URL, content=REQUEST_BODY)
    assert response.status_code == 404

    replayer = CachingTransport(inner=mock, mode=LlmCacheMode.REPLAY, schema_version=1, db_path=db_path())
    async with httpx.AsyncClient(transport=replayer) as client:
        with pytest.raises(LlmCacheMiss):
            await client.post(CHAT_URL, content=REQUEST_BODY)
    assert len(seen) == 1


async def test_record_mode_serves_prior_recording_without_origin() -> None:
    mock, seen = counting_origin(lambda: httpx.Response(200, content=RESPONSE_BODY))
    recorder = CachingTransport(inner=mock, mode=LlmCacheMode.REPLAY_OR_RECORD, schema_version=1, db_path=db_path())
    async with httpx.AsyncClient(transport=recorder) as client:
        await client.post(CHAT_URL, content=REQUEST_BODY)
        again = await client.post(CHAT_URL, content=REQUEST_BODY)
    assert again.content == RESPONSE_BODY
    assert len(seen) == 1


@pytest.mark.parametrize(
    ("left", "right", "expected_equal"),
    [
        pytest.param(
            httpx.Request("POST", CHAT_URL, headers={"authorization": "Bearer a"}, content=REQUEST_BODY),
            httpx.Request("POST", CHAT_URL, headers={"authorization": "Bearer z", "x-id": "9"}, content=REQUEST_BODY),
            True,
            id="ignores-headers",
        ),
        pytest.param(
            httpx.Request("POST", CHAT_URL, content=b'{"a": 1}'),
            httpx.Request("POST", CHAT_URL, content=b'{"a": 2}'),
            False,
            id="varies-on-body",
        ),
        pytest.param(
            httpx.Request("POST", "https://api.openai.com/v1/x?a=1", content=REQUEST_BODY),
            httpx.Request("POST", "https://api.openai.com/v1/x?a=2", content=REQUEST_BODY),
            False,
            id="varies-on-query",
        ),
        pytest.param(
            httpx.Request("POST", "https://api.openai.com/v1/x?a=1&b=2", content=REQUEST_BODY),
            httpx.Request("POST", "https://api.openai.com/v1/x?b=2&a=1", content=REQUEST_BODY),
            True,
            id="query-order-insensitive",
        ),
    ],
)
def test_cache_key_selectivity(left: httpx.Request, right: httpx.Request, expected_equal: bool) -> None:
    assert (cache_key(left, schema_version=1) == cache_key(right, schema_version=1)) is expected_equal


def test_cache_key_varies_on_schema_version() -> None:
    request = httpx.Request("POST", CHAT_URL, content=REQUEST_BODY)
    assert cache_key(request, schema_version=1) != cache_key(request, schema_version=2)


def test_cache_key_distinguishes_scheme_and_port() -> None:
    keys = {
        cache_key(httpx.Request("GET", url), schema_version=1)
        for url in ("http://h/x", "https://h/x", "https://h:8443/x")
    }
    assert len(keys) == 3


def test_cache_key_collapses_default_port() -> None:
    assert cache_key(httpx.Request("GET", "https://h/x"), schema_version=1) == cache_key(
        httpx.Request("GET", "https://h:443/x"), schema_version=1
    )


def test_cache_key_distinguishes_explicit_zero_port_from_default() -> None:
    assert cache_key(httpx.Request("GET", "http://h:0/x"), schema_version=1) != cache_key(
        httpx.Request("GET", "http://h:80/x"), schema_version=1
    )


async def test_record_mode_refreshes_stored_row_on_rerecord() -> None:
    bodies = iter([b"first", b"second"])
    mock, seen = counting_origin(lambda: httpx.Response(200, content=next(bodies)))
    recorder = CachingTransport(inner=mock, mode=LlmCacheMode.RECORD, schema_version=1, db_path=db_path())
    async with httpx.AsyncClient(transport=recorder) as client:
        await client.post(CHAT_URL, content=REQUEST_BODY)
        await client.post(CHAT_URL, content=REQUEST_BODY)
    assert len(seen) == 2

    replayer = CachingTransport(inner=mock, mode=LlmCacheMode.REPLAY, schema_version=1, db_path=db_path())
    async with httpx.AsyncClient(transport=replayer) as client:
        replayed = await client.post(CHAT_URL, content=REQUEST_BODY)
    assert replayed.content == b"second"


async def test_duplicate_response_headers_survive_replay() -> None:
    dup_headers = [("set-cookie", "a=1"), ("set-cookie", "b=2"), ("content-type", "application/json")]
    mock, _ = counting_origin(lambda: httpx.Response(200, headers=dup_headers, content=RESPONSE_BODY))
    recorder = CachingTransport(inner=mock, mode=LlmCacheMode.RECORD, schema_version=1, db_path=db_path())
    async with httpx.AsyncClient(transport=recorder) as client:
        await client.post(CHAT_URL, content=REQUEST_BODY)

    replayer = CachingTransport(inner=mock, mode=LlmCacheMode.REPLAY, schema_version=1, db_path=db_path())
    async with httpx.AsyncClient(transport=replayer) as client:
        replayed = await client.post(CHAT_URL, content=REQUEST_BODY)
    assert replayed.headers.get_list("set-cookie") == ["a=1", "b=2"]
