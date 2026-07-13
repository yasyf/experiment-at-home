from __future__ import annotations

import asyncio
import fcntl
import math
import os
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, Protocol

from athome.config import AthomeSettings, load
from athome.errors import AthomeError
from athome.llm.pricing import cost
from athome.progress import load_journal

if TYPE_CHECKING:
    from pathlib import Path

    from athome.wire import Wire

SCHEMA_VERSION = 1
CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOKENS = 1024
BATCH_DISCOUNT = 0.5
MAX_TOKEN_KEYS = ("max_tokens", "max_output_tokens", "maxOutputTokens")

type Provider = Literal["anthropic", "openai", "gemini"]


class BatchError(AthomeError):
    """Root of the batch adapter's typed errors."""


class BudgetExceeded(BatchError):
    """Raised when a batch's pre-submit cost estimate exceeds the caller's ``max_usd``."""


class BatchStatus(StrEnum):
    """Provider-agnostic lifecycle state of a batch or a single item within one.

    ``EXPIRED`` is distinct from ``FAILED``: an expired item is *unbilled*, so it is
    safe to resubmit under a fresh ``custom_id`` rather than counted as a failure.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class BatchRequest:
    """One batched call: a stable ``custom_id`` and the provider chat-completion body.

    Example:
        >>> BatchRequest(custom_id="row-7", body={"model": "gpt-4o-mini", "messages": []})
    """

    custom_id: str
    body: dict[str, Wire]


@dataclass(frozen=True, slots=True)
class BatchResult:
    """One item's outcome, correlated back to its request by ``custom_id`` alone.

    ``body`` is the provider response payload on success and ``None`` when the item
    failed or expired.
    """

    custom_id: str
    body: dict[str, Wire] | None
    status: BatchStatus


@dataclass(frozen=True, slots=True)
class BatchJob:
    """A submitted batch's provider, remote id, and the JSONL state file that owns it.

    The state file under ``batches_root`` is the sole idempotency and resume layer:
    :meth:`open` reconstructs a job from it so a later process (or the daily collect
    agent) can poll and collect without any in-memory handle.

    Example:
        >>> job = await submit(reqs, provider="anthropic", max_usd=5.0)
        >>> job = BatchJob.open(job.state_path)
    """

    provider: Provider
    provider_batch_id: str
    state_path: Path
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def open(cls, state_path: Path) -> BatchJob:
        intent = intent_record(state_path)
        if (batch_id := submitted_batch_id(state_path)) is None:
            raise BatchError(
                f"dangling submit attempt in {state_path}: intent journaled but no batch_id. "
                "A batch may already be running at the provider — reconcile it there before resubmitting."
            )
        return cls(
            provider=intent["provider"],
            provider_batch_id=batch_id,
            state_path=state_path,
            schema_version=int(intent["schema_version"]),
        )


class BatchProvider(Protocol):
    async def submit(self, reqs: Sequence[BatchRequest]) -> str: ...
    async def poll(self, batch_id: str) -> BatchStatus: ...
    async def collect(self, batch_id: str) -> list[BatchResult]: ...
    def estimate_usd(self, reqs: Sequence[BatchRequest]) -> float: ...


def batches_root() -> Path:
    return load(AthomeSettings).batches_root


def new_attempt_id() -> str:
    return uuid.uuid4().hex


def state_path_for(provider: Provider, attempt_id: str) -> Path:
    return batches_root() / f"{provider}-{attempt_id}.jsonl"


def lock_path(state_path: Path) -> Path:
    return state_path.with_name(f"{state_path.name}.lock")


@asynccontextmanager
async def state_lock(state_path: Path) -> AsyncIterator[None]:
    fd = os.open(lock_path(state_path), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        await asyncio.to_thread(fcntl.flock, fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


async def fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        await asyncio.to_thread(os.fsync, fd)
    finally:
        os.close(fd)


def fresh_custom_id(custom_id: str) -> str:
    return f"{custom_id}::retry::{uuid.uuid4().hex[:8]}"


def check_max_usd(max_usd: float) -> None:
    if not (math.isfinite(max_usd) and max_usd > 0):
        raise BatchError(f"max_usd must be finite and > 0, got {max_usd!r}")


def check_unique_custom_ids(reqs: Sequence[BatchRequest]) -> None:
    ids = [req.custom_id for req in reqs]
    if len(ids) != len(set(ids)):
        raise BatchError(f"duplicate custom_id(s): {sorted({cid for cid in ids if ids.count(cid) > 1})}")


def collect_text(value: object) -> Iterator[str]:
    match value:
        case str():
            yield value
        case Mapping():
            for item in value.values():
                yield from collect_text(item)
        case list() | tuple():
            for item in value:
                yield from collect_text(item)


def request_tokens(body: Mapping[str, Wire]) -> tuple[int, int]:
    return (
        sum(len(text) for text in collect_text(body)) // CHARS_PER_TOKEN,
        next((int(body[key]) for key in MAX_TOKEN_KEYS if key in body), DEFAULT_MAX_TOKENS),
    )


def estimate_request_usd(req: BatchRequest) -> float:
    input_tokens, output_tokens = request_tokens(req.body)
    return cost(str(req.body["model"]), input_tokens=input_tokens, output_tokens=output_tokens) * BATCH_DISCOUNT


def estimate_batch_usd(reqs: Sequence[BatchRequest]) -> float:
    return sum(estimate_request_usd(req) for req in reqs)


def intent_record(state_path: Path) -> dict[str, object]:
    return next(record for record in load_journal(state_path) if record.get("event") == "intent")


def submitted_batch_id(state_path: Path) -> str | None:
    return next(
        (str(record["batch_id"]) for record in load_journal(state_path) if record.get("event") == "submitted"),
        None,
    )


def dangling_attempts(provider: Provider) -> list[Path]:
    return [
        path
        for path in sorted(batches_root().glob(f"{provider}-*.jsonl"))
        if (records := load_journal(path))
        and any(record.get("event") == "intent" for record in records)
        and not any(record.get("event") == "submitted" for record in records)
    ]


def submitted_bodies(state_path: Path) -> dict[str, dict[str, Wire]]:
    return {entry["custom_id"]: entry["body"] for entry in intent_record(state_path)["requests"]}


def collected_ids(state_path: Path) -> set[str]:
    return {record["custom_id"] for record in load_journal(state_path) if record.get("event") == "result"}


def retried_ids(state_path: Path) -> set[str]:
    return {record["old_custom_id"] for record in load_journal(state_path) if record.get("event") == "retry_intent"}


def retry_records(state_path: Path) -> list[dict[str, object]]:
    return [record for record in load_journal(state_path) if record.get("event") == "retry"]


def retry_roots(state_path: Path) -> dict[str, str]:
    return {str(record["new_custom_id"]): str(record["root_custom_id"]) for record in retry_records(state_path)}
