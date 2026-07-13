from __future__ import annotations

from functools import cache
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Literal, overload

from pydantic import BaseModel

from athome.config import SectionSettings, load
from athome.llm.pricing import cost
from athome.llm.spend import SpendGuard
from athome.llm.telemetry import CallLog, CallRecord

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from athome.serve import Recipe

TierName = Literal["small", "medium", "large"]

TIER_PRICE_MODEL: dict[TierName, str] = {
    "small": "claude-haiku-4-5",
    "medium": "claude-sonnet-5",
    "large": "claude-opus-4-8",
}
OUTPUT_ALLOWANCE_TOKENS = 512


class LlmSettings(SectionSettings):
    """The ``[llm]`` section: the process-wide spend cap and an optional telemetry sink."""

    section = ("llm",)
    max_usd: float = 10.0
    telemetry_sink: Path | None = None


@cache
def default_log() -> CallLog:
    """The process-wide :class:`CallLog`, built once from :class:`LlmSettings`."""
    return CallLog(sink=load(LlmSettings).telemetry_sink)


@cache
def default_guard() -> SpendGuard:
    """The process-wide :class:`SpendGuard`, capped at ``LlmSettings.max_usd``."""
    return SpendGuard(max_usd=load(LlmSettings).max_usd)


async def metered[R](
    price_model: str,
    prompt: str,
    invoke: Callable[[], Awaitable[R]],
    render: Callable[[R], str],
) -> R:
    """Reserve spend, invoke, then reconcile spend and log the call, priced as ``price_model``.

    Lane calls surface no served model (spawnllm returns only the completion), so every
    :class:`~athome.llm.telemetry.CallRecord` carries ``served_model=None`` and
    ``system_fingerprint=None`` — telemetry's drift detector is inert for lane calls.
    """
    guard = default_guard()
    input_tokens = len(prompt) // 4
    projected = cost(price_model, input_tokens=input_tokens, output_tokens=OUTPUT_ALLOWANCE_TOKENS)
    await guard.check(projected)
    started = perf_counter()
    try:
        result = await invoke()
    except BaseException:
        await guard.record(projected, 0.0)
        raise
    output_tokens = len(render(result)) // 4
    spent = cost(price_model, input_tokens=input_tokens, output_tokens=output_tokens)
    await guard.record(projected, spent)
    default_log().add(
        CallRecord(
            model=price_model,
            latency_s=perf_counter() - started,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=spent,
            served_model=None,
            system_fingerprint=None,
        )
    )
    return result


async def small(prompt: str, *, model: TierName = "small", timeout: int = 180) -> str:
    """Run a cheap interactive completion through spawnllm, metered and spend-guarded.

    Args:
        prompt: The user prompt.
        model: The abstract tier, priced via :data:`TIER_PRICE_MODEL`.
        timeout: Seconds before the backend process is killed.

    Returns:
        The model's text response.
    """
    from spawnllm import call

    return await metered(
        TIER_PRICE_MODEL[model],
        prompt,
        lambda: call(prompt, model=model, timeout=timeout),
        lambda result: result,
    )


async def extract[T: BaseModel](prompt: str, schema: type[T], *, model: TierName = "large", timeout: int = 180) -> T:
    """Run a structured completion validated to ``schema``, metered and spend-guarded.

    Args:
        prompt: The user prompt.
        schema: The Pydantic model the structured output is validated against.
        model: The abstract tier, priced via :data:`TIER_PRICE_MODEL`.
        timeout: Seconds before the backend process is killed.

    Returns:
        The validated ``schema`` instance.
    """
    from spawnllm import extract as run_extract

    return await metered(
        TIER_PRICE_MODEL[model],
        prompt,
        lambda: run_extract(prompt, schema, model=model, timeout=timeout),
        lambda result: result.model_dump_json(),
    )


@overload
async def local(prompt: str, *, schema: None = None, recipe: Recipe = "rapid-mlx") -> str: ...
@overload
async def local[T: BaseModel](prompt: str, *, schema: type[T], recipe: Recipe = "rapid-mlx") -> T: ...
async def local[T: BaseModel](prompt: str, *, schema: type[T] | None = None, recipe: Recipe = "rapid-mlx") -> str | T:
    """Drive a local recipe's OpenAI endpoint through spawnllm, cache-wrapped and metered.

    The recipe server is ensured healthy, then reached through spawnllm's
    OpenAI-endpoint backend over the record/replay transport. Local inference is
    priced as the ``small`` tier: the served model carries no :data:`PRICES` entry
    and the compute is effectively free.

    Args:
        prompt: The user prompt.
        schema: When given, the Pydantic model the output is validated against;
            ``None`` returns raw text.
        recipe: The managed-server recipe to reach.

    Returns:
        The text response, or the validated ``schema`` instance when ``schema`` is given.
    """
    from spawnllm import OpenAiEndpointBackend, call
    from spawnllm import extract as run_extract

    from athome import llmcache
    from athome.serve import ManagedServer, settings_for

    handle = await ManagedServer(recipe).ensure()
    backend = OpenAiEndpointBackend(handle.base_url, settings_for(recipe).model, transport=llmcache.transport())
    if schema is None:
        return await metered(
            TIER_PRICE_MODEL["small"], prompt, lambda: call(prompt, backend=backend), lambda result: result
        )
    return await metered(
        TIER_PRICE_MODEL["small"],
        prompt,
        lambda: run_extract(prompt, schema, backend=backend),
        lambda result: result.model_dump_json(),
    )
