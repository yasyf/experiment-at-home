from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import httpx
from pydantic import Field

from athome.config import SectionSettings, load
from athome.llm.batch.state import (
    BatchError,
    BatchRequest,
    BatchResult,
    BatchStatus,
    estimate_batch_usd,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

GEMINI_BASE = "https://generativelanguage.googleapis.com"
DISPLAY_NAME = "athome-batch"
REQUEST_TIMEOUT = 60.0


class GeminiBatchSettings(SectionSettings):
    """The ``[batch.gemini]`` section: the Gemini key, read from ``GEMINI_API_KEY``."""

    section: ClassVar[tuple[str, ...]] = ("batch", "gemini")
    api_key: str = Field(validation_alias="GEMINI_API_KEY")


def http_client(settings: GeminiBatchSettings) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=GEMINI_BASE,
        headers={"x-goog-api-key": settings.api_key},
        timeout=REQUEST_TIMEOUT,
    )


def batch_status(state: str) -> BatchStatus:
    match state:
        case "JOB_STATE_PENDING" | "JOB_STATE_QUEUED":
            return BatchStatus.PENDING
        case "JOB_STATE_RUNNING":
            return BatchStatus.RUNNING
        case "JOB_STATE_SUCCEEDED":
            return BatchStatus.COMPLETED
        case "JOB_STATE_FAILED" | "JOB_STATE_CANCELLED":
            return BatchStatus.FAILED
        case "JOB_STATE_EXPIRED":
            return BatchStatus.EXPIRED
        case other:
            raise BatchError(f"unknown gemini job state: {other}")


def inlined_responses(batch: dict[str, object]) -> list[dict[str, object]]:
    return (((batch.get("response") or {}).get("inlinedResponses") or {}).get("inlinedResponses")) or []


def item_result(item: dict[str, object]) -> BatchResult:
    custom_id = item["metadata"]["key"]
    if "response" in item:
        return BatchResult(custom_id=custom_id, body=item["response"], status=BatchStatus.COMPLETED)
    return BatchResult(custom_id=custom_id, body=None, status=BatchStatus.FAILED)


@dataclass(frozen=True, slots=True)
class GeminiBatch:
    """Gemini async Batch adapter: inline requests keyed by ``custom_id`` via per-item metadata."""

    async def submit(self, reqs: Sequence[BatchRequest]) -> str:
        payload = {
            "batch": {
                "display_name": DISPLAY_NAME,
                "input_config": {
                    "requests": {
                        "requests": [{"metadata": {"key": req.custom_id}, "request": req.body} for req in reqs]
                    }
                },
            }
        }
        async with http_client(load(GeminiBatchSettings)) as client:
            response = await client.post("/v1beta/batches", json=payload)
            response.raise_for_status()
            return response.json()["name"]

    async def poll(self, batch_id: str) -> BatchStatus:
        async with http_client(load(GeminiBatchSettings)) as client:
            response = await client.get(f"/v1beta/{batch_id}")
            response.raise_for_status()
            return batch_status(response.json()["metadata"]["state"])

    async def collect(self, batch_id: str) -> list[BatchResult]:
        async with http_client(load(GeminiBatchSettings)) as client:
            response = await client.get(f"/v1beta/{batch_id}")
            response.raise_for_status()
            return [item_result(item) for item in inlined_responses(response.json())]

    def estimate_usd(self, reqs: Sequence[BatchRequest]) -> float:
        return estimate_batch_usd(reqs)
