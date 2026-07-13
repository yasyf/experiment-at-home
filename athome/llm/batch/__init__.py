from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import click
from loguru import logger

from athome.cli import coro, emit, json_option
from athome.llm.batch.anthropic import AnthropicBatch
from athome.llm.batch.gemini import GeminiBatch
from athome.llm.batch.openai import OpenAIBatch
from athome.llm.batch.state import (
    SCHEMA_VERSION,
    BatchError,
    BatchJob,
    BatchProvider,
    BatchRequest,
    BatchResult,
    BatchStatus,
    BudgetExceeded,
    Provider,
    collected_ids,
    fresh_custom_id,
    now_iso,
    retried_ids,
    state_path_for,
    submitted_bodies,
)
from athome.progress import RunSink

if TYPE_CHECKING:
    from collections.abc import Sequence

PROVIDER_CHOICE = click.Choice(["anthropic", "openai", "gemini"])


def provider_for(provider: Provider) -> BatchProvider:
    match provider:
        case "anthropic":
            return AnthropicBatch()
        case "openai":
            return OpenAIBatch()
        case "gemini":
            return GeminiBatch()


async def submit(reqs: Sequence[BatchRequest], *, provider: Provider, max_usd: float) -> BatchJob:
    """Submit ``reqs`` to ``provider``, refusing to spend more than ``max_usd``.

    The pre-submit estimate is checked before any network call: an over-budget batch
    raises :class:`BudgetExceeded` and never reaches the provider. On success the
    request bodies and remote batch id are journaled to a JSONL state file — the sole
    idempotency and resume layer — and the resulting :class:`BatchJob` is returned.

    Args:
        reqs: The requests to batch, each with its own ``custom_id`` correlation key.
        provider: Which provider's batch REST to drive.
        max_usd: The hard cap on the estimated batch cost.

    Returns:
        The submitted job, backed by its state file under ``batches_root``.

    Raises:
        BudgetExceeded: The estimated cost exceeds ``max_usd``.
    """
    adapter = provider_for(provider)
    if (estimate := adapter.estimate_usd(reqs)) > max_usd:
        raise BudgetExceeded(f"estimated ${estimate:.4f} exceeds max ${max_usd:.4f}")
    batch_id = await adapter.submit(reqs)
    path = state_path_for(provider, batch_id)
    await RunSink.open(path).append(
        {
            "event": "submitted",
            "schema_version": SCHEMA_VERSION,
            "provider": provider,
            "batch_id": batch_id,
            "submitted_at": now_iso(),
            "requests": [{"custom_id": req.custom_id, "body": req.body} for req in reqs],
        }
    )
    return BatchJob(provider=provider, provider_batch_id=batch_id, state_path=path)


async def status(job: BatchJob) -> BatchStatus:
    """Poll ``job``'s remote batch and return its current :class:`BatchStatus`."""
    return await provider_for(job.provider).poll(job.provider_batch_id)


async def collect(job: BatchJob) -> list[BatchResult]:
    """Collect ``job``'s results, resubmitting any expired items under fresh ids.

    Returns an empty list until the batch has completed or expired. Completed and
    failed items are journaled to the state file keyed by ``custom_id``; a batch-level
    expiry synthesizes an ``EXPIRED`` result for every item the provider did not
    return. Every expired item is resubmitted as a fresh single-item batch under a new
    ``custom_id`` (logged as a retry, recorded in state, and never counted as a
    failure). Re-collecting is idempotent: already-journaled items are not rewritten
    and already-retried items are not resubmitted.
    """
    adapter = provider_for(job.provider)
    batch_state = await adapter.poll(job.provider_batch_id)
    if batch_state not in (BatchStatus.COMPLETED, BatchStatus.EXPIRED):
        return []
    results = await adapter.collect(job.provider_batch_id)
    bodies = submitted_bodies(job.state_path)
    if batch_state is BatchStatus.EXPIRED:
        returned = {result.custom_id for result in results}
        results = results + [
            BatchResult(custom_id, None, BatchStatus.EXPIRED) for custom_id in bodies if custom_id not in returned
        ]
    sink = RunSink.open(job.state_path)
    already = collected_ids(job.state_path)
    for result in results:
        if result.custom_id not in already:
            await sink.append(
                {"event": "result", "custom_id": result.custom_id, "status": result.status, "body": result.body}
            )
    await resubmit_expired(adapter, job, results, sink, bodies)
    return results


async def resubmit_expired(
    adapter: BatchProvider,
    job: BatchJob,
    results: Sequence[BatchResult],
    sink: RunSink,
    bodies: dict[str, dict],
) -> None:
    retried = retried_ids(job.state_path)
    expired = [r for r in results if r.status is BatchStatus.EXPIRED and r.custom_id not in retried]
    if not expired:
        return
    fresh = [BatchRequest(custom_id=fresh_custom_id(r.custom_id), body=bodies[r.custom_id]) for r in expired]
    batch_id = await adapter.submit(fresh)
    for old, new in zip(expired, fresh, strict=True):
        logger.info(
            "batch {}: resubmitting expired {} as {} in {}",
            job.provider_batch_id,
            old.custom_id,
            new.custom_id,
            batch_id,
        )
        await sink.append(
            {"event": "retry", "old_custom_id": old.custom_id, "new_custom_id": new.custom_id, "batch_id": batch_id}
        )


def load_requests(path: Path) -> list[BatchRequest]:
    return [
        BatchRequest(custom_id=record["custom_id"], body=record["body"])
        for line in path.read_text().splitlines()
        if line
        for record in [json.loads(line)]
    ]


@click.group("batch")
def cli() -> None:
    """Submit, poll, and collect provider batch jobs (50%-off async inference)."""


@cli.command("submit")
@click.argument("requests_file", type=click.Path(exists=True, path_type=Path))
@click.option("--provider", type=PROVIDER_CHOICE, required=True, help="Batch provider to submit to.")
@click.option("--max-usd", type=float, required=True, help="Abort before submit if the estimate exceeds this cap.")
@json_option
@coro
async def submit_command(requests_file: Path, provider: str, max_usd: float, as_json: bool) -> None:
    """Submit the JSONL REQUESTS_FILE (one {custom_id, body} per line) as a batch."""
    job = await submit(load_requests(requests_file), provider=provider, max_usd=max_usd)
    emit(
        {"provider": job.provider, "batch_id": job.provider_batch_id, "state_path": str(job.state_path)},
        as_json=as_json,
    )


@cli.command("status")
@click.argument("state_file", type=click.Path(exists=True, path_type=Path))
@json_option
@coro
async def status_command(state_file: Path, as_json: bool) -> None:
    """Print the current status of the batch tracked by STATE_FILE."""
    job = BatchJob.open(state_file)
    emit({"batch_id": job.provider_batch_id, "status": await status(job)}, as_json=as_json)


@cli.command("collect")
@click.argument("state_file", type=click.Path(exists=True, path_type=Path))
@json_option
@coro
async def collect_command(state_file: Path, as_json: bool) -> None:
    """Collect results for the batch tracked by STATE_FILE, resubmitting expired items."""
    results = await collect(BatchJob.open(state_file))
    emit([{"custom_id": r.custom_id, "status": r.status} for r in results], as_json=as_json)
