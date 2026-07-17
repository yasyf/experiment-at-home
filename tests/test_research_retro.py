from __future__ import annotations

from typing import TYPE_CHECKING, cast

import anyio
import pytest

from athome.research import judge as judge_mod
from athome.research.journal import CC_NOTES_BIN, CC_NOTES_LABEL, JournalRow, Verdict
from athome.research.nightly import MorningReport
from athome.research.retro import RetroJournal, RetroRecord, RetroVerdict, build_prompt, generate

if TYPE_CHECKING:
    from pathlib import Path

    from spawnllm.backends.base import LlmBackend


def make_row(
    unit: int,
    metric: float | None,
    verdict: Verdict,
    *,
    commit: str = "abc123",
    description: str = "measured experiment",
    resources: dict[str, float] | None = None,
) -> JournalRow:
    return JournalRow(
        unit=unit,
        commit=commit,
        metric=metric,
        verdict=verdict,
        resources=resources if resources is not None else {"wall_s": 1.0},
        description=description,
    )


def make_report(rows: tuple[JournalRow, ...], *, experiment: str = "toy") -> MorningReport:
    kept = tuple(row for row in rows if row.verdict is Verdict.KEEP)
    return MorningReport(
        experiment=experiment,
        units=len(rows),
        kept=len(kept),
        crashes=sum(row.verdict is Verdict.CRASH for row in rows),
        best=kept[-1],
        rows=rows,
    )


def make_verdict() -> RetroVerdict:
    return RetroVerdict(
        outcome="improved",
        summary="The kept metric moved from 1.0 to 0.6.",
        evidence=("Two of three units were kept.",),
        next_steps=("Repeat the strongest configuration.",),
    )


def test_build_prompt_projects_only_numeric_and_fixed_enum_evidence() -> None:
    rows = (
        make_row(0, 1.0, Verdict.KEEP),
        make_row(1, 0.8, Verdict.DISCARD),
        make_row(2, 0.6, Verdict.KEEP),
    )

    prompt = build_prompt(rows, make_report(rows), baseline=1.0, uplift=0.4)

    assert '"baseline": 1.0' in prompt
    assert '"uplift": 0.4' in prompt
    assert '"kept": 2' in prompt
    assert '"metric": 0.6' in prompt
    assert '"verdict": "discard"' in prompt


def test_build_prompt_with_lying_stdout_never_includes_raw_log_text() -> None:
    marker = "<<<RAW-RUN-STDOUT>>>" + ("\x00garbage" * 2_000)
    rows = (
        make_row(
            0,
            1.0,
            Verdict.KEEP,
            commit=marker,
            description=marker,
            resources={marker: 1.0},
        ),
        make_row(1, 0.7, Verdict.KEEP, description=marker),
    )
    report = make_report(rows, experiment=marker)

    assert marker not in build_prompt(rows, report, baseline=1.0, uplift=0.3)


async def test_generate_binds_registry_backend_and_forwards_extract_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawnllm = pytest.importorskip("spawnllm")
    from spawnllm.backends.registry import BACKENDS_BY_NAME

    captured: dict[str, object] = {}
    expected = make_verdict()

    async def fake_extract(
        prompt: str,
        verdict_model: type[RetroVerdict],
        *,
        backend: object,
        model: str,
        timeout: int,
    ) -> RetroVerdict:
        captured.update(
            prompt=prompt,
            verdict_model=verdict_model,
            backend=backend,
            model=model,
            timeout=timeout,
        )
        return expected

    monkeypatch.setattr(spawnllm, "extract", fake_extract)
    rows = (make_row(0, 1.0, Verdict.KEEP), make_row(1, 0.6, Verdict.KEEP))

    result = await generate(
        rows,
        make_report(rows),
        backend="codex",
        tier="small",
        timeout=17,
        label="nightly-retro",
    )

    assert result is expected
    assert captured["verdict_model"] is RetroVerdict
    assert captured["backend"] is BACKENDS_BY_NAME["codex"]
    assert captured["model"] == "small"
    assert captured["timeout"] == 17
    assert '"baseline": 1.0' in cast("str", captured["prompt"])
    assert '"uplift": 0.4' in cast("str", captured["prompt"])


async def test_generate_retries_transient_extract_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    spawnllm = pytest.importorskip("spawnllm")
    monkeypatch.setattr(judge_mod, "BACKOFF_BASE_S", 0.0)
    attempts = {"count": 0}
    expected = make_verdict()

    async def flaky_extract(
        prompt: str,
        verdict_model: type[RetroVerdict],
        *,
        backend: object,
        model: str,
        timeout: int,
    ) -> RetroVerdict:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise spawnllm.BackendCallError("transient")
        return expected

    monkeypatch.setattr(spawnllm, "extract", flaky_extract)
    rows = (make_row(0, 1.0, Verdict.KEEP), make_row(1, 0.6, Verdict.KEEP))

    result = await generate(
        rows,
        make_report(rows),
        backend=cast("LlmBackend", object()),
    )

    assert result is expected
    assert attempts["count"] == 3


def test_retro_record_uses_first_kept_metric_as_baseline() -> None:
    rows = (
        make_row(0, None, Verdict.CRASH),
        make_row(1, 1.0, Verdict.KEEP),
        make_row(2, 0.8, Verdict.DISCARD),
        make_row(3, 0.6, Verdict.KEEP),
    )

    record = RetroRecord.from_report(make_report(rows), make_verdict())

    assert (record.baseline, record.best_metric, record.uplift) == (1.0, 0.6, 0.4)


def test_retro_record_from_record_requires_every_field() -> None:
    with pytest.raises(KeyError, match="best_metric"):
        RetroRecord.from_record(
            {
                "experiment": "toy",
                "baseline": 1.0,
                "uplift": 0.4,
                "verdict": make_verdict().model_dump(mode="json"),
            }
        )


async def test_retro_journal_mirrors_to_cc_notes_by_default_with_exact_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    async def fake(command: list[str], *args: object, **kwargs: object) -> None:
        commands.append(command)

    monkeypatch.setattr(anyio, "run_process", fake)
    record = RetroRecord("toy", 1.0, 0.6, 0.4, make_verdict())
    journal = RetroJournal.open(tmp_path / "retro.jsonl")

    await journal.append(record)

    assert commands == [
        [
            str(CC_NOTES_BIN),
            "note",
            "add",
            "athome retro toy [improved]",
            "--body",
            "\n".join(
                [
                    "baseline 1.0",
                    "best metric 0.6",
                    "uplift 0.4",
                    "summary The kept metric moved from 1.0 to 0.6.",
                    'evidence ["Two of three units were kept."]',
                    'next steps ["Repeat the strongest configuration."]',
                ]
            ),
            "--label",
            CC_NOTES_LABEL,
        ]
    ]
    assert RetroJournal.open(tmp_path / "retro.jsonl", mirror_cc_notes=False).records() == [record]


async def test_retro_record_is_durable_before_cc_notes_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("cc-notes unavailable")

    monkeypatch.setattr(anyio, "run_process", boom)
    path = tmp_path / "retro.jsonl"
    record = RetroRecord("toy", 1.0, 0.6, 0.4, make_verdict())

    with pytest.raises(RuntimeError, match="cc-notes unavailable"):
        await RetroJournal.open(path).append(record)

    assert RetroJournal.open(path, mirror_cc_notes=False).records() == [record]
