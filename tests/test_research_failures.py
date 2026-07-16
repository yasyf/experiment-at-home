from __future__ import annotations

import json
from subprocess import CalledProcessError
from typing import TYPE_CHECKING, Literal

import pytest

from athome.research.driver import MetricShapeError
from athome.research.failures import (
    CandidateFault,
    InfraFailure,
    classify,
    infra_cost,
    infra_events,
    infra_log,
    infra_retries,
    record_infra_event,
)
from athome.research.spec import ImmutableViolation, ProposalTimeout

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "exc, expected",
    [
        pytest.param(InfraFailure("machine trouble"), "infra", id="infra-failure"),
        pytest.param(OSError("disk gone"), "infra", id="oserror"),
        pytest.param(ConnectionResetError("peer reset"), "infra", id="connection-reset-subclass"),
        pytest.param(CalledProcessError(1, ["git", "archive"]), "infra", id="harness-git-subprocess"),
        pytest.param(CandidateFault("git rejected the staged candidate tree"), "candidate", id="candidate-git-fault"),
        pytest.param(ProposalTimeout("hung", cost=1.0), "candidate", id="proposal-timeout"),
        pytest.param(ImmutableViolation("touched score.py"), "candidate", id="immutable-violation"),
        pytest.param(MetricShapeError("wrong shape"), "candidate", id="metric-shape"),
        pytest.param(RuntimeError("harness bug"), "candidate", id="unknown-defaults-to-candidate"),
    ],
)
def test_classify_taxonomy(exc: BaseException, expected: Literal["infra", "candidate"]) -> None:
    assert classify(exc) == expected


@pytest.mark.parametrize(
    "log, expected",
    [
        pytest.param(b"Connection reset by peer", True, id="case-insensitive"),
        pytest.param(b"HTTP 503 Service Unavailable", True, id="503-service"),
        pytest.param(b"OSError: [Errno 28] No space left on device", True, id="enospc"),
        pytest.param(b"429 Too Many Requests: rate limit exceeded", True, id="rate-limit"),
        pytest.param(b"NameError: name 'undefined_symbol' is not defined", False, id="candidate-nameerror"),
        pytest.param(b'{"loss": 0.5}\nloss=999.0\n', False, id="clean-metric-log"),
        pytest.param(b"", False, id="empty-log"),
        pytest.param(b"\xff\xfe garbage \x00 connection refused", True, id="non-utf8-still-scans"),
    ],
)
def test_infra_log_marker_scan(log: bytes, expected: bool) -> None:
    assert infra_log(log) is expected


async def test_record_and_count_infra_events(tmp_path: Path) -> None:
    events = tmp_path / "toy.events.jsonl"
    assert infra_retries(events) == 0 and infra_cost(events) == 0.0  # absent sidecar reads as zero

    await record_infra_event(events, unit=3, attempt=0, reason="OSError('reset')", cost=0.6)
    await record_infra_event(events, unit=3, attempt=1, reason="OSError('reset')", cost=0.0)

    assert infra_retries(events) == 2
    assert infra_cost(events) == pytest.approx(0.6)
    records = [json.loads(line) for line in events.read_text().splitlines()]
    assert [record["attempt"] for record in records] == [0, 1]
    assert all(record["unit"] == 3 and record["reason"] == "OSError('reset')" for record in records)


async def test_torn_sidecar_line_never_loses_a_later_record(tmp_path: Path) -> None:
    # Finding #4: a torn (newline-less) prior line must not swallow the next appends. The writer
    # heals the fragment onto its own line; the reader skips only that malformed line.
    events = tmp_path / "toy.events.jsonl"
    await record_infra_event(events, unit=0, attempt=0, reason="ok", cost=0.6)
    with events.open("a") as handle:
        handle.write('{"unit": 1, "attempt": 0, "reason": "tor')  # a crash mid-write leaves this fragment

    await record_infra_event(events, unit=2, attempt=0, reason="ok", cost=0.6)
    await record_infra_event(events, unit=3, attempt=0, reason="ok", cost=0.6)

    assert sorted(event["unit"] for event in infra_events(events)) == [0, 2, 3]  # fragment skipped, nothing else lost
    assert infra_retries(events) == 3
    assert infra_cost(events) == pytest.approx(1.8)  # 0.6 × 3, no undercount from the tear
