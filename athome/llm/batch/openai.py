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

OPENAI_BASE = "https://api.openai.com"
COMPLETION_WINDOW = "24h"
CHAT_ENDPOINT = "/v1/chat/completions"
REQUEST_TIMEOUT = 60.0


class OpenAIBatchSettings(SectionSettings):
    """The ``[batch.openai]`` section: the OpenAI key, read from ``OPENAI_API_KEY``."""

    section: ClassVar[tuple[str, ...]] = ("batch", "openai")
    api_key: str = Field(validation_alias="OPENAI_API_KEY")


def http_client(settings: OpenAIBatchSettings) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=OPENAI_BASE,
        headers={"Authorization": f"Bearer {settings.api_key}"},
        timeout=REQUEST_TIMEOUT,
    )


def input_jsonl(reqs: Sequence[BatchRequest]) -> bytes:
    return "\n".join(
        json.dumps({"custom_id": req.custom_id, "method": "POST", "url": CHAT_ENDPOINT, "body": req.body})
        for req in reqs
    ).encode()


def batch_status(status: str) -> BatchStatus:
    match status:
        case "validating":
            return BatchStatus.PENDING
        case "in_progress" | "finalizing" | "cancelling":
            return BatchStatus.RUNNING
        case "completed":
            return BatchStatus.COMPLETED
        case "failed" | "cancelled":
            return BatchStatus.FAILED
        case "expired":
            return BatchStatus.EXPIRED
        case other:
            raise BatchError(f"unknown openai batch status: {other}")


def output_result(line: dict[str, object]) -> BatchResult:
    response = line["response"]
    if response is not None and response["status_code"] == 200:
        return BatchResult(custom_id=line["custom_id"], body=response["body"], status=BatchStatus.COMPLETED)
    return BatchResult(custom_id=line["custom_id"], body=None, status=BatchStatus.FAILED)


def error_result(line: dict[str, object]) -> BatchResult:
    expired = line["error"]["code"] == "batch_expired"
    return BatchResult(
        custom_id=line["custom_id"],
        body=None,
        status=BatchStatus.EXPIRED if expired else BatchStatus.FAILED,
    )


@dataclass(frozen=True, slots=True)
class OpenAIBatch:
    """OpenAI ``/v1/batches`` adapter: file upload, batch create, output/error file collect."""

    async def submit(self, reqs: Sequence[BatchRequest]) -> str:
        async with http_client(load(OpenAIBatchSettings)) as client:
            upload = await client.post(
                "/v1/files",
                data={"purpose": "batch"},
                files={"file": ("batch.jsonl", input_jsonl(reqs), "application/jsonl")},
            )
            upload.raise_for_status()
            response = await client.post(
                "/v1/batches",
                json={
                    "input_file_id": upload.json()["id"],
                    "endpoint": CHAT_ENDPOINT,
                    "completion_window": COMPLETION_WINDOW,
                },
            )
            response.raise_for_status()
            return response.json()["id"]

    async def poll(self, batch_id: str) -> BatchStatus:
        async with http_client(load(OpenAIBatchSettings)) as client:
            response = await client.get(f"/v1/batches/{batch_id}")
            response.raise_for_status()
            return batch_status(response.json()["status"])

    async def collect(self, batch_id: str) -> list[BatchResult]:
        async with http_client(load(OpenAIBatchSettings)) as client:
            response = await client.get(f"/v1/batches/{batch_id}")
            response.raise_for_status()
            batch = response.json()
            sources = ((batch.get("output_file_id"), output_result), (batch.get("error_file_id"), error_result))
            return [
                mapper(json.loads(line))
                for file_id, mapper in sources
                if file_id
                for line in (await self.file_content(client, file_id)).splitlines()
                if line
            ]

    async def file_content(self, client: httpx.AsyncClient, file_id: str) -> str:
        response = await client.get(f"/v1/files/{file_id}/content")
        response.raise_for_status()
        return response.text

    def estimate_usd(self, reqs: Sequence[BatchRequest]) -> float:
        return estimate_batch_usd(reqs)
