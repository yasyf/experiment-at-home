from __future__ import annotations

import json
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

ANTHROPIC_BASE = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
REQUEST_TIMEOUT = 60.0


class AnthropicBatchSettings(SectionSettings):
    """The ``[batch.anthropic]`` section: the Anthropic key, read from ``ANTHROPIC_API_KEY``."""

    section: ClassVar[tuple[str, ...]] = ("batch", "anthropic")
    api_key: str = Field(validation_alias="ANTHROPIC_API_KEY")


def http_client(settings: AnthropicBatchSettings) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=ANTHROPIC_BASE,
        headers={"x-api-key": settings.api_key, "anthropic-version": ANTHROPIC_VERSION},
        timeout=REQUEST_TIMEOUT,
    )


def batch_status(processing_status: str) -> BatchStatus:
    match processing_status:
        case "in_progress" | "canceling":
            return BatchStatus.RUNNING
        case "ended":
            return BatchStatus.COMPLETED
        case other:
            raise BatchError(f"unknown anthropic processing_status: {other}")


def item_result(line: dict[str, object]) -> BatchResult:
    result = line["result"]
    match result["type"]:
        case "succeeded":
            return BatchResult(custom_id=line["custom_id"], body=result["message"], status=BatchStatus.COMPLETED)
        case "expired":
            return BatchResult(custom_id=line["custom_id"], body=None, status=BatchStatus.EXPIRED)
        case "errored" | "canceled":
            return BatchResult(custom_id=line["custom_id"], body=None, status=BatchStatus.FAILED)
        case other:
            raise BatchError(f"unknown anthropic result type: {other}")


@dataclass(frozen=True, slots=True)
class AnthropicBatch:
    """Anthropic Message Batches adapter over raw httpx."""

    async def submit(self, reqs: Sequence[BatchRequest]) -> str:
        payload = {"requests": [{"custom_id": req.custom_id, "params": req.body} for req in reqs]}
        async with http_client(load(AnthropicBatchSettings)) as client:
            response = await client.post("/v1/messages/batches", json=payload)
            response.raise_for_status()
            return response.json()["id"]

    async def poll(self, batch_id: str) -> BatchStatus:
        async with http_client(load(AnthropicBatchSettings)) as client:
            response = await client.get(f"/v1/messages/batches/{batch_id}")
            response.raise_for_status()
            return batch_status(response.json()["processing_status"])

    async def collect(self, batch_id: str) -> list[BatchResult]:
        async with http_client(load(AnthropicBatchSettings)) as client:
            response = await client.get(f"/v1/messages/batches/{batch_id}/results")
            response.raise_for_status()
            return [item_result(json.loads(line)) for line in response.text.splitlines() if line]

    def estimate_usd(self, reqs: Sequence[BatchRequest]) -> float:
        return estimate_batch_usd(reqs)
