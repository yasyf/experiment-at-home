from __future__ import annotations

import json
from typing import TYPE_CHECKING

import anyio
import pytest
from loguru import logger

from athome.llm.pricing import PRICES, Price, UnpricedModel, cost
from athome.llm.spend import InvalidBudget, SpendExceeded, SpendGuard
from athome.llm.telemetry import CallLog, CallRecord

if TYPE_CHECKING:
    from pathlib import Path


def make_record(**overrides: object) -> CallRecord:
    base = {
        "model": "gpt-x",
        "latency_s": 1.0,
        "input_tokens": 10,
        "output_tokens": 5,
        "cost_usd": 0.001,
        "served_model": "gpt-x-2026",
        "system_fingerprint": "fp_1",
    }
    return CallRecord(**(base | overrides))


@pytest.mark.parametrize(
    ("model", "input_tokens", "output_tokens", "expected"),
    [
        pytest.param("claude-opus-4-8", 1_000_000, 1_000_000, 30.0, id="opus-one-million-each"),
        pytest.param("claude-opus-4-8", 500_000, 200_000, 7.5, id="opus-partial-mtok"),
        pytest.param("claude-fable-5", 1_000_000, 0, 10.0, id="fable-input-only"),
        pytest.param("gpt-4o-mini", 0, 1_000_000, 0.6, id="gpt4omini-output-only"),
        pytest.param("claude-haiku-4-5", 0, 0, 0.0, id="zero-tokens"),
    ],
)
def test_cost_computes_usd(model: str, input_tokens: int, output_tokens: int, expected: float) -> None:
    assert cost(model, input_tokens=input_tokens, output_tokens=output_tokens) == pytest.approx(expected)


def test_cost_raises_on_unpriced_model() -> None:
    with pytest.raises(UnpricedModel, match="mystery-model"):
        cost("mystery-model", input_tokens=1, output_tokens=1)


def test_prices_are_price_instances() -> None:
    assert all(isinstance(price, Price) for price in PRICES.values())
    assert PRICES["claude-opus-4-8"] == Price(5.0, 25.0)


async def test_spend_guard_allows_under_cap() -> None:
    await SpendGuard(max_usd=1.0).check(0.99)


async def test_spend_guard_raises_over_cap() -> None:
    with pytest.raises(SpendExceeded, match=r"exceeds cap"):
        await SpendGuard(max_usd=1.0).check(1.01)


async def test_spend_guard_accumulates_and_then_blocks() -> None:
    guard = SpendGuard(max_usd=1.0)
    await guard.check(0.6)
    await guard.record(0.6, 0.6)
    assert guard.spent == pytest.approx(0.6)
    assert guard.reserved == pytest.approx(0.0)
    with pytest.raises(SpendExceeded):
        await guard.check(0.5)


async def test_an_unbounded_envelope_is_spelled_none_and_never_refuses() -> None:
    guard = SpendGuard(max_usd=None)
    await guard.check(1_000_000.0)
    await guard.check(1_000_000.0)
    assert guard.reserved == pytest.approx(2_000_000.0)


async def test_an_unbounded_envelope_still_accumulates_spend_for_reporting() -> None:
    guard = SpendGuard(max_usd=None)
    await guard.check(5.0)
    await guard.record(5.0, 4.2)
    assert guard.spent == pytest.approx(4.2)
    assert guard.reserved == pytest.approx(0.0)


async def test_external_spend_debits_share_the_envelope_with_hosted_reservations() -> None:
    guard = SpendGuard(max_usd=1.0)
    await guard.check(0.4)
    await guard.record(0.4, 0.4)
    await guard.check(0.3)
    await guard.record(0.3, 0.3)
    assert guard.spent == pytest.approx(0.7)
    with pytest.raises(SpendExceeded):
        await guard.check(0.4)
    await guard.record(0.0, 0.2)
    assert guard.spent == pytest.approx(0.9)
    assert guard.reserved == pytest.approx(0.0)


async def test_concurrent_reservations_never_overcommit_a_shared_envelope() -> None:
    guard = SpendGuard(max_usd=1.0)
    outcomes: list[bool] = []

    async def reserve() -> None:
        try:
            await guard.check(0.3)
        except SpendExceeded:
            outcomes.append(False)
        else:
            outcomes.append(True)

    async with anyio.create_task_group() as tg:
        for _ in range(10):
            tg.start_soon(reserve)

    assert sum(outcomes) == 3
    assert guard.reserved == pytest.approx(3 * 0.3)
    assert guard.reserved <= guard.max_usd


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(float("inf"), id="inf"),
        pytest.param(float("-inf"), id="-inf"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(-1.0, id="negative"),
    ],
)
def test_an_invalid_budget_is_refused_at_construction(bad: float) -> None:
    with pytest.raises(InvalidBudget, match="finite and non-negative"):
        SpendGuard(max_usd=bad)


async def test_a_zero_cap_is_legal_and_refuses_all_spend() -> None:
    guard = SpendGuard(max_usd=0.0)
    await guard.check(0.0)
    with pytest.raises(SpendExceeded):
        await guard.check(0.01)


def test_calllog_total_usd_sums_records() -> None:
    log = CallLog()
    log.add(make_record(cost_usd=0.1))
    log.add(make_record(cost_usd=0.25))
    assert log.total_usd == pytest.approx(0.35)
    assert len(log.records) == 2


def test_calllog_writes_jsonl_sink(tmp_path: Path) -> None:
    sink = tmp_path / "calls.jsonl"
    log = CallLog(sink=sink)
    log.add(make_record(cost_usd=0.2))
    log.add(make_record(cost_usd=0.3))
    lines = sink.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["cost_usd"] == 0.2
    assert json.loads(lines[1])["served_model"] == "gpt-x-2026"


def test_calllog_warns_on_served_model_drift() -> None:
    messages: list[str] = []
    handle = logger.add(messages.append, level="WARNING")
    log = CallLog()
    log.add(make_record(served_model="alpha"))
    log.add(make_record(served_model="beta"))
    logger.remove(handle)
    assert any("served-model drift" in str(message) for message in messages)


def test_calllog_warns_on_fingerprint_drift() -> None:
    messages: list[str] = []
    handle = logger.add(messages.append, level="WARNING")
    log = CallLog()
    log.add(make_record(system_fingerprint="fp_1"))
    log.add(make_record(system_fingerprint="fp_2"))
    logger.remove(handle)
    assert any("system-fingerprint drift" in str(message) for message in messages)


def test_calllog_quiet_without_drift() -> None:
    messages: list[str] = []
    handle = logger.add(messages.append, level="WARNING")
    log = CallLog()
    log.add(make_record(served_model="alpha", system_fingerprint="fp_1"))
    log.add(make_record(served_model="alpha", system_fingerprint="fp_1"))
    logger.remove(handle)
    assert messages == []
