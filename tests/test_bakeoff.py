from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio
import pytest
from click.testing import CliRunner

from athome import bakeoff
from athome.bakeoff import Arm, BakeoffSpec, WinnerPicker, cli, run
from athome.config import load

if TYPE_CHECKING:
    from collections.abc import Mapping

LLAMA = Arm(name="llama", base_url="http://127.0.0.1:8402/v1", model="m")
RAPID = Arm(name="rapid", base_url="http://127.0.0.1:8400/v1", model="m")
CORPUS = ("q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8")


@dataclass(slots=True)
class FakeClient:
    base_url: str

    async def close(self) -> None: ...


def use_fake_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bakeoff, "client_for", lambda arm: FakeClient(arm.base_url))


def spec_from(
    outputs: Mapping[str, Mapping[str, object]],
    *,
    arms: tuple[Arm, ...] = (LLAMA, RAPID),
    corpus: tuple[object, ...] = CORPUS,
    primary_metric: str = "exact",
    tiebreak: str | None = None,
) -> BakeoffSpec:
    async def task(client: FakeClient, item: object) -> dict[str, object]:
        return dict(outputs[client.base_url])

    return BakeoffSpec(task=task, corpus=corpus, arms=arms, primary_metric=primary_metric, tiebreak=tiebreak)


async def test_winner_and_gate_pass_when_candidate_beats_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake_clients(monkeypatch)
    board = await run(
        spec_from(
            {
                LLAMA.base_url: {"exact": 0.0, "label": "no", "viable": 1.0},
                RAPID.base_url: {"exact": 1.0, "label": "yes", "viable": 1.0},
            }
        )
    )
    assert board.winner == "rapid"
    assert board.passed_gate is True
    assert tuple(result.arm for result in board.results) == ("rapid", "llama")
    metrics = {result.arm: result.metrics for result in board.results}
    assert metrics["rapid"]["exact"] == 1.0
    assert metrics["llama"]["exact"] == 0.0


async def test_gate_fails_when_candidate_regresses(monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake_clients(monkeypatch)
    board = await run(
        spec_from(
            {
                LLAMA.base_url: {"exact": 1.0, "viable": 1.0},
                RAPID.base_url: {"exact": 0.0, "viable": 1.0},
            }
        )
    )
    assert board.winner == "llama"
    assert board.passed_gate is False


async def test_gate_fails_when_arms_are_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake_clients(monkeypatch)
    board = await run(
        spec_from(
            {
                LLAMA.base_url: {"exact": 0.5, "label": "x", "viable": 1.0},
                RAPID.base_url: {"exact": 0.5, "label": "x", "viable": 1.0},
            }
        )
    )
    assert board.winner == "llama"
    assert board.passed_gate is False
    assert {result.arm: result.metrics["agreement"] for result in board.results} == {"llama": 1.0, "rapid": 1.0}


async def test_per_field_disagreement_measures_against_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake_clients(monkeypatch)
    board = await run(
        spec_from(
            {
                LLAMA.base_url: {"exact": 0.0, "label": "no", "viable": 1.0},
                RAPID.base_url: {"exact": 1.0, "label": "yes", "viable": 1.0},
            }
        )
    )
    disagreement = {result.arm: result.per_field_disagreement for result in board.results}
    assert disagreement["llama"] == {"exact": 0.0, "label": 0.0, "viable": 0.0}
    assert disagreement["rapid"] == {"exact": 1.0, "label": 1.0, "viable": 0.0}


async def test_agreement_metric_counts_matching_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake_clients(monkeypatch)
    board = await run(
        spec_from(
            {
                LLAMA.base_url: {"exact": 0.0, "label": "no", "viable": 1.0},
                RAPID.base_url: {"exact": 1.0, "label": "yes", "viable": 1.0},
            }
        )
    )
    # Per item: viable agrees, exact + label disagree -> 1/3 of cells match.
    assert board.results[0].metrics["agreement"] == pytest.approx(1 / 3)


async def test_hard_constraint_disqualifies_non_viable_winner(monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake_clients(monkeypatch)

    async def task(client: FakeClient, item: object) -> dict[str, object]:
        if client.base_url == RAPID.base_url:
            return {"exact": 1.0, "viable": 1.0 if item != "q3" else 0.0}
        return {"exact": 0.5, "viable": 1.0}

    board = await run(BakeoffSpec(task=task, corpus=CORPUS, arms=(LLAMA, RAPID), primary_metric="exact"))
    assert board.winner == "llama"
    assert board.passed_gate is False
    assert {result.arm for result in board.results} == {"llama", "rapid"}
    assert next(r for r in board.results if r.arm == "rapid").metrics["viable"] == 0.875


async def test_tiebreak_breaks_equal_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake_clients(monkeypatch)
    board = await run(
        spec_from(
            {
                LLAMA.base_url: {"exact": 0.5, "score": 0.1, "viable": 1.0},
                RAPID.base_url: {"exact": 0.5, "score": 0.9, "viable": 1.0},
            },
            tiebreak="score",
        )
    )
    assert board.winner == "rapid"
    assert board.passed_gate is False


async def test_bounded_concurrency_caps_in_flight_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake_clients(monkeypatch)
    monkeypatch.setenv("ATHOME_BAKEOFF_CONCURRENCY", "2")
    load.cache_clear()
    state = {"current": 0, "peak": 0}

    async def task(client: FakeClient, item: object) -> dict[str, object]:
        state["current"] += 1
        state["peak"] = max(state["peak"], state["current"])
        await anyio.sleep(0.01)
        state["current"] -= 1
        return {"exact": 1.0}

    await run(BakeoffSpec(task=task, corpus=tuple(range(6)), arms=(LLAMA,), primary_metric="exact"))
    assert state["peak"] == 2


async def test_task_runs_once_per_corpus_item_per_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake_clients(monkeypatch)
    calls: list[tuple[str, object]] = []

    async def task(client: FakeClient, item: object) -> dict[str, object]:
        calls.append((client.base_url, item))
        return {"exact": 1.0, "viable": 1.0}

    await run(BakeoffSpec(task=task, corpus=CORPUS, arms=(LLAMA, RAPID), primary_metric="exact"))
    assert len(calls) == len(CORPUS) * 2
    assert sum(url == RAPID.base_url for url, _ in calls) == len(CORPUS)


def test_winner_picker_skips_non_viable_arm() -> None:
    spec = spec_from({}, primary_metric="exact")
    from athome.bakeoff import ArmResult

    results = (
        ArmResult(arm="a", metrics={"exact": 0.9, "viable": 0.5}, per_field_disagreement={}),
        ArmResult(arm="b", metrics={"exact": 0.4, "viable": 1.0}, per_field_disagreement={}),
    )
    picked = WinnerPicker.pick(results, spec)
    assert picked is not None and picked.arm == "b"


def register_spec_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: object) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


def test_load_spec_resolves_instance_and_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = spec_from({})
    register_spec_module(monkeypatch, "fake_spec_a", spec=instance, build=lambda: instance)
    assert bakeoff.load_spec("fake_spec_a") is instance
    assert bakeoff.load_spec("fake_spec_a:spec") is instance
    assert bakeoff.load_spec("fake_spec_a:build") is instance


def test_load_spec_rejects_non_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    register_spec_module(monkeypatch, "fake_spec_b", spec=object())
    with pytest.raises(bakeoff.BakeoffError, match="expected a BakeoffSpec"):
        bakeoff.load_spec("fake_spec_b")


def test_cli_run_emits_leaderboard(monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake_clients(monkeypatch)
    spec = spec_from(
        {
            LLAMA.base_url: {"exact": 0.0, "viable": 1.0},
            RAPID.base_url: {"exact": 1.0, "viable": 1.0},
        }
    )
    monkeypatch.setattr(bakeoff, "load_spec", lambda target: spec)
    result = CliRunner().invoke(cli, ["run", "pkg:spec", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["winner"] == "rapid"
    assert payload["passed_gate"] is True
    assert {row["arm"] for row in payload["results"]} == {"llama", "rapid"}


@pytest.mark.live
async def test_live_two_recipe_fidelity_gate() -> None:
    from openai import AsyncOpenAI

    async def extract(client: AsyncOpenAI, item: object) -> dict[str, object]:
        response = await client.chat.completions.create(
            model="local", messages=[{"role": "user", "content": str(item)}]
        )
        return {"exact": 1.0, "text": response.choices[0].message.content, "viable": 1.0}

    board = await run(
        BakeoffSpec(
            task=extract,
            corpus=("2+2=?",),
            arms=(
                Arm(name="llama-server", base_url="http://127.0.0.1:8402/v1", model="local"),
                Arm(name="rapid-mlx", base_url="http://127.0.0.1:8400/v1", model="local"),
            ),
            primary_metric="exact",
        )
    )
    assert board.winner in {"llama-server", "rapid-mlx"}
