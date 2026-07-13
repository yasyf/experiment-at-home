from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import pytest
from click.testing import CliRunner
from loguru import logger
from pydantic import ValidationError

from athome.config import AthomeSettings, load
from athome.llm import batch
from athome.llm.batch import BatchJob, BatchRequest, BatchStatus, BudgetExceeded
from athome.llm.batch.anthropic import ANTHROPIC_VERSION, AnthropicBatchSettings
from athome.llm.batch.state import estimate_batch_usd, request_tokens
from athome.llm.pricing import UnpricedModel, cost
from athome.progress import load_journal

if TYPE_CHECKING:
    from collections.abc import Callable

MODEL_BODY: dict = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 16}


def set_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
    monkeypatch.setenv("GEMINI_API_KEY", "sk-gem")
    load.cache_clear()


def patch_http(
    monkeypatch: pytest.MonkeyPatch, module: object, handler: Callable[[httpx.Request], httpx.Response]
) -> None:
    monkeypatch.setattr(
        module,
        "http_client",
        lambda settings: httpx.AsyncClient(base_url="http://mock", transport=httpx.MockTransport(handler)),
    )


def anthropic_router(
    results: list[dict], *, processing_status: str = "ended", ids: tuple[str, ...] = ("msgbatch_1", "msgbatch_2")
) -> tuple[Callable[[httpx.Request], httpx.Response], list[dict]]:
    submits: list[dict] = []
    ids_iter = iter(ids)

    def handler(request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if method == "POST" and path == "/v1/messages/batches":
            submits.append(json.loads(request.content))
            return httpx.Response(200, json={"id": next(ids_iter), "processing_status": "in_progress"})
        if method == "GET" and path.endswith("/results"):
            return httpx.Response(200, text="\n".join(json.dumps(line) for line in results))
        if method == "GET" and path.startswith("/v1/messages/batches/"):
            return httpx.Response(200, json={"processing_status": processing_status})
        raise AssertionError(f"unexpected {method} {path}")

    return handler, submits


def reqs(*custom_ids: str, body: dict | None = None) -> list[BatchRequest]:
    return [BatchRequest(custom_id=cid, body=dict(body or MODEL_BODY)) for cid in custom_ids]


# --- settings / auth ---------------------------------------------------------


def test_settings_read_key_from_bare_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-alias")
    monkeypatch.delenv("ATHOME_BATCH_ANTHROPIC_API_KEY", raising=False)
    load.cache_clear()
    assert load(AnthropicBatchSettings).api_key == "sk-alias"


def test_settings_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    load.cache_clear()
    with pytest.raises(ValidationError):
        load(AnthropicBatchSettings)


async def test_http_client_sets_auth_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-h")
    load.cache_clear()
    client = batch.anthropic.http_client(load(AnthropicBatchSettings))
    try:
        assert client.headers["x-api-key"] == "sk-h"
        assert client.headers["anthropic-version"] == ANTHROPIC_VERSION
    finally:
        await client.aclose()


# --- estimate / budget -------------------------------------------------------


def test_estimate_is_half_of_full_price() -> None:
    req = BatchRequest("a", {"model": "gpt-4o-mini", "messages": [{"content": "word " * 100}], "max_tokens": 50})
    input_tokens, output_tokens = request_tokens(req.body)
    assert output_tokens == 50
    full = cost("gpt-4o-mini", input_tokens=input_tokens, output_tokens=output_tokens)
    assert estimate_batch_usd([req]) == pytest.approx(full * 0.5)


async def test_submit_aborts_over_budget_before_any_network(monkeypatch: pytest.MonkeyPatch) -> None:
    set_keys(monkeypatch)

    def boom(request: httpx.Request) -> httpx.Response:
        raise AssertionError("over-budget submit must not reach the provider")

    patch_http(monkeypatch, batch.anthropic, boom)
    big = {"model": "claude-opus-4-8", "messages": [{"role": "user", "content": "x" * 8000}], "max_tokens": 2000}
    with pytest.raises(BudgetExceeded, match="exceeds max"):
        await batch.submit(reqs("a", body=big), provider="anthropic", max_usd=0.001)
    assert list((load(AthomeSettings).batches_root).glob("*.jsonl")) == []


async def test_submit_raises_on_unpriced_model(monkeypatch: pytest.MonkeyPatch) -> None:
    set_keys(monkeypatch)
    handler, _ = anthropic_router([])
    patch_http(monkeypatch, batch.anthropic, handler)
    unpriced = reqs("a", body={"model": "gemini-2.0-flash", "max_tokens": 8})
    with pytest.raises(UnpricedModel, match="gemini-2.0-flash"):
        await batch.submit(unpriced, provider="anthropic", max_usd=100.0)


# --- anthropic flow ----------------------------------------------------------


async def test_submit_writes_state_and_collect_maps_by_custom_id(monkeypatch: pytest.MonkeyPatch) -> None:
    set_keys(monkeypatch)
    results = [
        {"custom_id": "a", "result": {"type": "succeeded", "message": {"id": "msg_a"}}},
        {"custom_id": "b", "result": {"type": "succeeded", "message": {"id": "msg_b"}}},
    ]
    handler, submits = anthropic_router(results)
    patch_http(monkeypatch, batch.anthropic, handler)

    job = await batch.submit(reqs("a", "b"), provider="anthropic", max_usd=100.0)
    assert job.provider == "anthropic"
    assert job.provider_batch_id == "msgbatch_1"
    assert job.state_path.name == "anthropic-msgbatch_1.jsonl"
    assert [entry["custom_id"] for entry in submits[0]["requests"]] == ["a", "b"]
    assert submits[0]["requests"][0]["params"] == MODEL_BODY

    collected = await batch.collect(job)
    assert {r.custom_id: r.status for r in collected} == {"a": BatchStatus.COMPLETED, "b": BatchStatus.COMPLETED}
    assert {r.custom_id: (r.body or {})["id"] for r in collected} == {"a": "msg_a", "b": "msg_b"}
    records = [r for r in load_journal(job.state_path) if r.get("event") == "result"]
    assert {r["custom_id"]: r["status"] for r in records} == {"a": "completed", "b": "completed"}


async def test_status_reports_running_before_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    set_keys(monkeypatch)
    handler, _ = anthropic_router([], processing_status="in_progress")
    patch_http(monkeypatch, batch.anthropic, handler)
    job = await batch.submit(reqs("a"), provider="anthropic", max_usd=100.0)
    assert await batch.status(job) == BatchStatus.RUNNING
    assert await batch.collect(job) == []


async def test_collect_records_partial_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    set_keys(monkeypatch)
    results = [
        {"custom_id": "ok", "result": {"type": "succeeded", "message": {"id": "m"}}},
        {"custom_id": "bad", "result": {"type": "errored", "error": {"type": "overloaded"}}},
    ]
    handler, submits = anthropic_router(results)
    patch_http(monkeypatch, batch.anthropic, handler)
    job = await batch.submit(reqs("ok", "bad"), provider="anthropic", max_usd=100.0)
    collected = {r.custom_id: r for r in await batch.collect(job)}
    assert collected["ok"].status == BatchStatus.COMPLETED and collected["ok"].body == {"id": "m"}
    assert collected["bad"].status == BatchStatus.FAILED and collected["bad"].body is None
    records = [r for r in load_journal(job.state_path) if r.get("event") == "result"]
    assert {r["custom_id"]: r["status"] for r in records} == {"ok": "completed", "bad": "failed"}
    assert not any(r.get("failed") for r in load_journal(job.state_path))
    assert len(submits) == 1


async def test_expired_item_resubmitted_with_fresh_id(monkeypatch: pytest.MonkeyPatch) -> None:
    set_keys(monkeypatch)
    results = [
        {"custom_id": "ok", "result": {"type": "succeeded", "message": {"id": "m"}}},
        {"custom_id": "gone", "result": {"type": "expired"}},
    ]
    handler, submits = anthropic_router(results)
    patch_http(monkeypatch, batch.anthropic, handler)
    logs: list[str] = []
    handle = logger.add(logs.append, level="INFO")

    job = await batch.submit(reqs("ok", "gone"), provider="anthropic", max_usd=100.0)
    collected = {r.custom_id: r.status for r in await batch.collect(job)}
    logger.remove(handle)

    assert collected == {"ok": BatchStatus.COMPLETED, "gone": BatchStatus.EXPIRED}
    assert len(submits) == 2
    resubmitted = submits[1]["requests"]
    assert len(resubmitted) == 1
    fresh = resubmitted[0]["custom_id"]
    assert fresh != "gone" and fresh.startswith("gone")
    assert resubmitted[0]["params"] == MODEL_BODY
    retries = [r for r in load_journal(job.state_path) if r.get("event") == "retry"]
    assert len(retries) == 1
    assert retries[0]["old_custom_id"] == "gone" and retries[0]["new_custom_id"] == fresh
    assert not any(r.get("failed") for r in load_journal(job.state_path))
    assert any("resubmitting expired gone" in message for message in logs)


async def test_recollect_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    set_keys(monkeypatch)
    results = [
        {"custom_id": "gone", "result": {"type": "expired"}},
        {"custom_id": "ok", "result": {"type": "succeeded", "message": {"id": "m"}}},
    ]
    handler, submits = anthropic_router(results)
    patch_http(monkeypatch, batch.anthropic, handler)

    job = await batch.submit(reqs("gone", "ok"), provider="anthropic", max_usd=100.0)
    await batch.collect(job)
    resumed = BatchJob.open(job.state_path)
    assert resumed.provider_batch_id == "msgbatch_1"
    await batch.collect(resumed)

    records = load_journal(job.state_path)
    assert len([r for r in records if r.get("event") == "result"]) == 2
    assert len([r for r in records if r.get("event") == "retry"]) == 1
    assert len(submits) == 2


# --- openai flow -------------------------------------------------------------


async def test_openai_submit_uploads_file_and_collects(monkeypatch: pytest.MonkeyPatch) -> None:
    set_keys(monkeypatch)
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if method == "POST" and path == "/v1/files":
            captured["upload"] = request.content
            return httpx.Response(200, json={"id": "file-in"})
        if method == "POST" and path == "/v1/batches":
            captured["input_file_id"] = json.loads(request.content)["input_file_id"]
            return httpx.Response(200, json={"id": "batch_1", "status": "validating"})
        if method == "GET" and path == "/v1/batches/batch_1":
            return httpx.Response(
                200, json={"status": "completed", "output_file_id": "file-out", "error_file_id": "file-err"}
            )
        if method == "GET" and path == "/v1/files/file-out/content":
            success = {"custom_id": "ok", "response": {"status_code": 200, "body": {"id": "r"}}, "error": None}
            return httpx.Response(200, text=json.dumps(success))
        if method == "GET" and path == "/v1/files/file-err/content":
            return httpx.Response(
                200, text=json.dumps({"custom_id": "bad", "response": None, "error": {"code": "server_error"}})
            )
        raise AssertionError(f"unexpected {method} {path}")

    patch_http(monkeypatch, batch.openai, handler)
    job = await batch.submit(reqs("ok", "bad"), provider="openai", max_usd=100.0)
    assert job.provider_batch_id == "batch_1"
    assert job.state_path.name == "openai-batch_1.jsonl"
    assert captured["input_file_id"] == "file-in"
    assert b'"custom_id": "ok"' in captured["upload"] and b'"custom_id": "bad"' in captured["upload"]

    collected = {r.custom_id: r for r in await batch.collect(job)}
    assert collected["ok"].status == BatchStatus.COMPLETED and collected["ok"].body == {"id": "r"}
    assert collected["bad"].status == BatchStatus.FAILED and collected["bad"].body is None


async def test_openai_batch_expired_resubmits_from_error_file(monkeypatch: pytest.MonkeyPatch) -> None:
    set_keys(monkeypatch)
    file_ids = iter(["file-in-1", "file-in-2"])
    batch_ids = iter(["batch_1", "batch_2"])
    uploads: list[bytes] = []
    creates: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if method == "POST" and path == "/v1/files":
            uploads.append(request.content)
            return httpx.Response(200, json={"id": next(file_ids)})
        if method == "POST" and path == "/v1/batches":
            creates.append(json.loads(request.content))
            return httpx.Response(200, json={"id": next(batch_ids), "status": "validating"})
        if method == "GET" and path == "/v1/batches/batch_1":
            return httpx.Response(200, json={"status": "expired", "error_file_id": "file-err"})
        if method == "GET" and path == "/v1/files/file-err/content":
            return httpx.Response(
                200, text=json.dumps({"custom_id": "gone", "response": None, "error": {"code": "batch_expired"}})
            )
        raise AssertionError(f"unexpected {method} {path}")

    patch_http(monkeypatch, batch.openai, handler)
    job = await batch.submit(reqs("gone"), provider="openai", max_usd=100.0)
    collected = await batch.collect(job)
    assert [(r.custom_id, r.status) for r in collected] == [("gone", BatchStatus.EXPIRED)]
    assert len(creates) == 2
    assert b"gone::retry::" in uploads[1]
    retries = [r for r in load_journal(job.state_path) if r.get("event") == "retry"]
    assert len(retries) == 1 and retries[0]["old_custom_id"] == "gone"


# --- gemini flow -------------------------------------------------------------


async def test_gemini_submit_and_collect_maps_by_key(monkeypatch: pytest.MonkeyPatch) -> None:
    set_keys(monkeypatch)
    submits: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if method == "POST" and path == "/v1beta/batches":
            submits.append(json.loads(request.content))
            return httpx.Response(200, json={"name": "batches/one"})
        if method == "GET" and path == "/v1beta/batches/one":
            return httpx.Response(
                200,
                json={
                    "metadata": {"state": "JOB_STATE_SUCCEEDED"},
                    "response": {
                        "inlinedResponses": {
                            "inlinedResponses": [
                                {"metadata": {"key": "a"}, "response": {"text": "hi"}},
                                {"metadata": {"key": "b"}, "error": {"code": 13, "message": "internal"}},
                            ]
                        }
                    },
                },
            )
        raise AssertionError(f"unexpected {method} {path}")

    patch_http(monkeypatch, batch.gemini, handler)
    job = await batch.submit(reqs("a", "b"), provider="gemini", max_usd=100.0)
    assert job.provider_batch_id == "batches/one"
    assert job.state_path.name == "gemini-batches_one.jsonl"
    keys = [entry["metadata"]["key"] for entry in submits[0]["batch"]["input_config"]["requests"]["requests"]]
    assert keys == ["a", "b"]

    collected = {r.custom_id: r for r in await batch.collect(job)}
    assert collected["a"].status == BatchStatus.COMPLETED and collected["a"].body == {"text": "hi"}
    assert collected["b"].status == BatchStatus.FAILED and collected["b"].body is None


async def test_gemini_batch_expired_synthesizes_and_resubmits(monkeypatch: pytest.MonkeyPatch) -> None:
    set_keys(monkeypatch)
    names = iter(["batches/one", "batches/two"])
    submits: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if method == "POST" and path == "/v1beta/batches":
            submits.append(json.loads(request.content))
            return httpx.Response(200, json={"name": next(names)})
        if method == "GET" and path == "/v1beta/batches/one":
            return httpx.Response(200, json={"metadata": {"state": "JOB_STATE_EXPIRED"}})
        raise AssertionError(f"unexpected {method} {path}")

    patch_http(monkeypatch, batch.gemini, handler)
    job = await batch.submit(reqs("x", "y"), provider="gemini", max_usd=100.0)
    collected = {r.custom_id: r.status for r in await batch.collect(job)}
    assert collected == {"x": BatchStatus.EXPIRED, "y": BatchStatus.EXPIRED}
    assert len(submits) == 2
    resubmitted = submits[1]["batch"]["input_config"]["requests"]["requests"]
    fresh_keys = sorted(entry["metadata"]["key"] for entry in resubmitted)
    assert len(fresh_keys) == 2
    assert fresh_keys[0].startswith("x::retry::") and fresh_keys[1].startswith("y::retry::")
    retries = {r["old_custom_id"] for r in load_journal(job.state_path) if r.get("event") == "retry"}
    assert retries == {"x", "y"}


# --- cli ---------------------------------------------------------------------


def test_cli_submit_status_collect(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    set_keys(monkeypatch)
    results = [{"custom_id": "a", "result": {"type": "succeeded", "message": {"id": "m"}}}]
    handler, _ = anthropic_router(results)
    patch_http(monkeypatch, batch.anthropic, handler)
    requests_file = tmp_path / "reqs.jsonl"
    requests_file.write_text(json.dumps({"custom_id": "a", "body": MODEL_BODY}) + "\n")
    runner = CliRunner()

    submitted = runner.invoke(
        batch.cli, ["submit", str(requests_file), "--provider", "anthropic", "--max-usd", "100", "--json"]
    )
    assert submitted.exit_code == 0, submitted.output
    payload = json.loads(submitted.output)
    assert payload["batch_id"] == "msgbatch_1"

    status = runner.invoke(batch.cli, ["status", payload["state_path"], "--json"])
    assert status.exit_code == 0, status.output
    assert json.loads(status.output)["status"] == "completed"

    collected = runner.invoke(batch.cli, ["collect", payload["state_path"], "--json"])
    assert collected.exit_code == 0, collected.output
    assert json.loads(collected.output) == [{"custom_id": "a", "status": "completed"}]
