from __future__ import annotations

import json
from subprocess import CalledProcessError
from typing import TYPE_CHECKING, Literal

import pytest

import athome.research.failures as failures
from athome.research.driver import MetricShapeError
from athome.research.failures import (
    AccountingIntegrityError,
    CandidateFault,
    InfraFailure,
    accounting_aborts,
    classify,
    infra_cost,
    infra_events,
    infra_log,
    infra_retries,
    record_accounting_abort,
    record_infra_event,
    safe_describe,
)
from athome.research.spec import ImmutableViolation, ProposalTimeout

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "exc, expected",
    [
        pytest.param(AccountingIntegrityError("unknown spend"), "accounting", id="accounting-integrity"),
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
def test_classify_taxonomy(exc: BaseException, expected: Literal["accounting", "infra", "candidate"]) -> None:
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
    latch = tmp_path / "toy.abort.json"
    events = tmp_path / "toy.events.jsonl"
    assert infra_retries(events) == 0 and infra_cost(events) == 0.0  # absent sidecar reads as zero

    await record_infra_event(events, unit=3, attempt=0, reason="OSError('reset')", cost=0.6, kind="retry", run=None)
    await record_infra_event(events, unit=3, attempt=1, reason="OSError('reset')", cost=0.0, kind="retry", run=None)
    await record_accounting_abort(latch, events, unit=4, reason="unknown spend")

    assert infra_retries(events) == 2
    assert infra_cost(events) == pytest.approx(0.6)
    assert accounting_aborts(events) == 1
    latch_payload = json.loads(latch.read_text())
    assert set(latch_payload) == {"unit", "reason", "ts"}
    assert latch_payload["unit"] == 4
    assert latch_payload["reason"] == "unknown spend"
    assert isinstance(latch_payload["ts"], float) and latch_payload["ts"] > 0
    records = [json.loads(line) for line in events.read_text().splitlines()]
    assert [record.get("attempt") for record in records] == [0, 1, None]
    assert [record["kind"] for record in records] == ["retry", "retry", "accounting_abort"]
    assert all(record["unit"] == 3 and record["reason"] == "OSError('reset')" for record in records[:2])
    assert records[2] == {"unit": 4, "reason": "unknown spend", "kind": "accounting_abort"}


async def test_accounting_abort_writes_latch_before_breadcrumb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    latch = tmp_path / "toy.abort.json"
    events = tmp_path / "toy.events.jsonl"
    observed: list[tuple[Path, dict[str, object], object]] = []

    def observe_append(path: Path, event: dict[str, object], *, kind: object) -> None:
        payload = json.loads(latch.read_text())
        assert payload["unit"] == 7
        assert payload["reason"] == "unrecoverable spend"
        observed.append((path, event, kind))

    monkeypatch.setattr(failures, "append_event", observe_append)

    await record_accounting_abort(latch, events, unit=7, reason="unrecoverable spend")

    assert observed == [(events, {"unit": 7, "reason": "unrecoverable spend"}, "accounting_abort")]


async def test_failed_initial_latch_write_skips_enrichment_entirely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # R3 fix 5: the renderer must never run without a durable latch on disk.
    latch = tmp_path / "toy.abort.json"
    events = tmp_path / "toy.events.jsonl"
    rendered: list[str] = []

    class ProbingError(Exception):
        def __str__(self) -> str:
            rendered.append("str")
            return "boom"

        def __repr__(self) -> str:
            rendered.append("repr")
            return "boom"

    async def fail_write(path: object, text: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(failures, "atomic_write_text", fail_write)

    await record_accounting_abort(latch, events, unit=3, reason=ProbingError("boom"))

    assert rendered == []  # enrichment skipped: no renderer ran without a durable latch
    assert not latch.exists()
    [event] = infra_events(events)
    assert event == {"unit": 3, "reason": "ProbingError", "kind": "accounting_abort"}


async def test_torn_middle_sidecar_line_never_loses_a_later_record(tmp_path: Path) -> None:
    # Finding #4: a torn (newline-less) prior line must not swallow the next appends. The writer
    # heals the fragment onto its own line; the reader skips only that malformed line.
    events = tmp_path / "toy.events.jsonl"
    await record_infra_event(events, unit=0, attempt=0, reason="ok", cost=0.6, kind="retry", run=None)
    with events.open("a") as handle:
        handle.write('{"unit": 1, "attempt": 0, "reason": "tor')  # a crash mid-write leaves this fragment

    await record_infra_event(events, unit=2, attempt=0, reason="ok", cost=0.6, kind="retry", run=None)
    await record_infra_event(events, unit=3, attempt=0, reason="ok", cost=0.6, kind="retry", run=None)

    assert sorted(event["unit"] for event in infra_events(events)) == [0, 2, 3]  # fragment skipped, nothing else lost
    assert infra_retries(events) == 3
    assert infra_cost(events) == pytest.approx(1.8)  # 0.6 × 3, no undercount from the tear


@pytest.mark.parametrize(
    "final_line",
    [
        pytest.param(b'{"unit":1,"attempt":0,"reason":"torn","cost":', id="torn-cost"),
        pytest.param(b'{"unit":1,"attempt":0,"reason":"unknown","cost":0.4,"kind":"legacy"}', id="unknown-kind"),
    ],
)
def test_malformed_final_sidecar_line_fails_closed(tmp_path: Path, final_line: bytes) -> None:
    events = tmp_path / "toy.events.jsonl"
    events.write_bytes(b'{"unit":0,"attempt":0,"reason":"ok","cost":0.2,"kind":"retry"}\n' + final_line)

    with pytest.raises(AccountingIntegrityError, match="malformed final line"):
        infra_events(events)


def test_corrupt_sidecar_lines_are_skipped_independently(tmp_path: Path) -> None:
    events = tmp_path / "toy.events.jsonl"
    events.write_bytes(
        b'{"unit":0,"attempt":0,"reason":"ok","cost":0.2,"kind":"retry"}\n'
        b'{"unit":1,"reason":"torn \xe2\x82\n'
        b"[]\n"
        b'{"unit":2,"attempt":0,"reason":"missing","kind":"retry"}\n'
        b'{"unit":3,"attempt":0,"reason":"bad","cost":"nope","kind":"retry"}\n'
        b'{"unit":4,"attempt":0,"reason":"ok","cost":0.4,"kind":"retry"}\n'
    )

    assert [event["unit"] for event in infra_events(events)] == [0, 2, 3, 4]
    assert infra_cost(events) == pytest.approx(0.6)


def test_missing_and_unknown_sidecar_kinds_are_skipped(tmp_path: Path) -> None:
    events = tmp_path / "toy.events.jsonl"
    events.write_bytes(
        b'{"unit":0,"attempt":0,"reason":"missing","cost":0.6}\n'
        b'{"unit":1,"attempt":0,"reason":"unknown","cost":0.7,"kind":"legacy"}\n'
        b'{"unit":2,"attempt":0,"reason":"retry","cost":0.4,"kind":"retry"}\n'
        b'{"unit":3,"attempt":0,"reason":"cancel","cost":0.2,"kind":"wall_cancel"}\n'
    )

    assert [(event["unit"], event["kind"]) for event in infra_events(events)] == [
        (2, "retry"),
        (3, "wall_cancel"),
    ]
    assert infra_retries(events) == 1
    assert infra_cost(events) == pytest.approx(0.6)


@pytest.mark.parametrize(
    "cost",
    [
        pytest.param(b"NaN", id="nan"),
        pytest.param(b"Infinity", id="infinity"),
        pytest.param(b"-Infinity", id="negative-infinity"),
        pytest.param(b"-0.1", id="negative"),
        pytest.param(b"true", id="true"),
        pytest.param(b"false", id="false"),
        pytest.param(b'"nope"', id="string"),
        pytest.param(b"9" * 1000, id="overflowing-integer"),
        pytest.param(b"9" * 10000, id="json-integer-limit"),
    ],
)
def test_invalid_sidecar_costs_are_skipped(tmp_path: Path, cost: bytes) -> None:
    events = tmp_path / "toy.events.jsonl"
    events.write_bytes(
        b'{"unit":0,"attempt":0,"reason":"invalid","kind":"retry","cost":'
        + cost
        + b'}\n{"unit":1,"attempt":0,"reason":"valid","kind":"retry","cost":0.4}\n'
    )

    assert infra_cost(events) == pytest.approx(0.4)


def test_unreadable_sidecar_cost_uses_accounting_taxonomy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events = tmp_path / "toy.events.jsonl"
    events.write_text('{"kind":"retry","cost":0.4}\n')

    def fail_read_bytes(path: Path) -> bytes:
        raise OSError(f"cannot read {path}")

    monkeypatch.setattr(type(events), "read_bytes", fail_read_bytes)

    with pytest.raises(AccountingIntegrityError):
        infra_cost(events)


class BrokenStr(Exception):
    def __str__(self) -> str:
        raise RuntimeError("broken __str__")


class BrokenStrAndRepr(BrokenStr):
    def __repr__(self) -> str:
        raise RuntimeError("broken __repr__")


class InterruptingStr(Exception):
    def __str__(self) -> str:
        raise KeyboardInterrupt


class InterruptingStrAndRepr(InterruptingStr):
    def __repr__(self) -> str:
        raise SystemExit(1)


def test_safe_describe_is_total_over_broken_renderers() -> None:
    assert safe_describe(ValueError("plain message")) == "plain message"
    assert safe_describe(BrokenStr("boom")) == "BrokenStr('boom')"
    assert safe_describe(BrokenStrAndRepr("boom")) == "<unprintable BrokenStrAndRepr>"
    assert safe_describe(InterruptingStr("boom")) == "InterruptingStr('boom')"
    assert safe_describe(InterruptingStrAndRepr("boom")) == "<unprintable InterruptingStrAndRepr>"


@pytest.mark.parametrize(
    "renderer_error",
    [
        pytest.param(RuntimeError, id="exception-renderer"),
        pytest.param(KeyboardInterrupt, id="base-exception-renderer"),
    ],
)
async def test_abort_latch_is_durable_before_any_renderer_runs(
    tmp_path: Path, renderer_error: type[BaseException]
) -> None:
    latch = tmp_path / "toy.abort.json"
    events = tmp_path / "toy.events.jsonl"
    observed: list[bool] = []

    class ProbingAbort(AccountingIntegrityError):
        def __str__(self) -> str:
            observed.append(latch.exists())
            raise renderer_error("hostile renderer")

    await record_accounting_abort(latch, events, unit=2, reason=ProbingAbort("boom"))

    assert observed and all(observed)  # every render found the latch already on disk
    enriched = json.loads(latch.read_text())
    assert enriched["reason"] == "ProbingAbort" and enriched["detail"] == "ProbingAbort('boom')"
    records = [json.loads(line) for line in events.read_text().splitlines()]
    assert [(record["kind"], record["reason"]) for record in records] == [("accounting_abort", "ProbingAbort")]
