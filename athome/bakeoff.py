from __future__ import annotations

import importlib
import random
import statistics
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, ClassVar

import anyio
import click

from athome.cli import coro, emit, json_option
from athome.config import SectionSettings, load
from athome.errors import AthomeError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from openai import AsyncOpenAI

AGREEMENT_METRIC = "agreement"
VIABLE_METRIC = "viable"


class BakeoffError(AthomeError):
    """Raised when a bake-off cannot be assembled or a spec module exposes no spec."""


class BakeoffSettings(SectionSettings):
    """The ``[bakeoff]`` section: corpus concurrency and the statistical-gate parameters."""

    section: ClassVar[tuple[str, ...]] = ("bakeoff",)
    concurrency: int = 8
    permutations: int = 10000
    alpha: float = 0.05
    min_lift: float = 0.0
    seed: int = 0


@dataclass(frozen=True, slots=True)
class Arm:
    """One endpoint under test: a name, an OpenAI-compatible base URL, and a model.

    Example:
        >>> Arm(name="rapid-mlx", base_url="http://127.0.0.1:8400/v1", model="Qwen3-4bit")
    """

    name: str
    base_url: str
    model: str


@dataclass(frozen=True, slots=True)
class BakeoffSpec:
    """A bake-off definition: a per-item task, the shared corpus, the arms, and the ranking keys.

    ``task`` runs one corpus item against one arm's client and returns a flat result dict;
    numeric fields aggregate into per-arm metrics (mean over the corpus), and every field
    feeds cross-arm agreement. ``primary_metric`` names the numeric field to maximise,
    ``tiebreak`` an optional secondary. A ``viable`` field (0/1) acts as a hard constraint:
    an arm not viable on every item is reported but cannot win. ``arms[0]`` is the baseline
    the gate and per-field disagreement measure against.

    Example:
        >>> BakeoffSpec(task=extract, corpus=pages, arms=(llama, rapid), primary_metric="exact")
    """

    task: Callable[[AsyncOpenAI, object], Awaitable[dict[str, object]]]
    corpus: tuple[object, ...]
    arms: tuple[Arm, ...]
    primary_metric: str
    tiebreak: str | None = None


@dataclass(frozen=True, slots=True)
class ArmResult:
    """One arm's aggregated metrics and its per-field disagreement against the baseline arm.

    ``metrics`` holds the mean of each numeric result field plus a computed ``agreement``
    (mean cell-match rate against the other arms). ``per_field_disagreement`` is the fraction
    of corpus items whose value for each field differs from the baseline arm's.
    """

    arm: str
    metrics: dict[str, float]
    per_field_disagreement: dict[str, float]


@dataclass(frozen=True, slots=True)
class Leaderboard:
    """The ranked arm results, the picked winner, and the statistical go/no-go verdict.

    ``winner`` is the best viable arm by ``(primary_metric, tiebreak)`` (empty when none is
    viable). ``passed_gate`` is ``True`` only when the winner is a non-baseline arm whose
    lift over the baseline on the primary metric clears ``min_lift`` and a paired sign-flip
    permutation test at ``alpha``.
    """

    results: tuple[ArmResult, ...]
    winner: str
    passed_gate: bool


def client_for(arm: Arm) -> AsyncOpenAI:
    from openai import AsyncOpenAI

    return AsyncOpenAI(base_url=arm.base_url, api_key="local")


async def run_arm(arm: Arm, spec: BakeoffSpec, *, concurrency: int) -> tuple[dict[str, object], ...]:
    client = client_for(arm)
    outputs: list[dict[str, object] | None] = [None] * len(spec.corpus)
    limiter = anyio.Semaphore(concurrency)

    async def run_item(index: int, item: object) -> None:
        async with limiter:
            outputs[index] = await spec.task(client, item)

    try:
        async with anyio.create_task_group() as group:
            for index, item in enumerate(spec.corpus):
                group.start_soon(run_item, index, item)
    finally:
        await client.close()
    return tuple(outputs)


def mean_by_field(outputs: Sequence[Mapping[str, object]]) -> dict[str, float]:
    fields = {k for output in outputs for k, value in output.items() if isinstance(value, bool | int | float)}
    return {metric: sum(float(o[metric]) for o in outputs if metric in o) / len(outputs) for metric in fields}


def cell_agreement(left: Sequence[Mapping[str, object]], right: Sequence[Mapping[str, object]]) -> float:
    cells = [
        k in left_item and k in right_item and left_item[k] == right_item[k]
        for left_item, right_item in zip(left, right, strict=True)
        for k in left_item.keys() | right_item.keys()
    ]
    return statistics.fmean(cells) if cells else 1.0


def agreement(name: str, outputs: Mapping[str, Sequence[Mapping[str, object]]]) -> float:
    peers = [cell_agreement(outputs[name], other) for peer, other in outputs.items() if peer != name]
    return statistics.fmean(peers) if peers else 1.0


def field_disagreement(
    outputs: Sequence[Mapping[str, object]], baseline: Sequence[Mapping[str, object]]
) -> dict[str, float]:
    pairs = list(zip(outputs, baseline, strict=True))
    return {
        field: statistics.fmean(
            [(field in out) != (field in base) or (field in out and out[field] != base[field]) for out, base in pairs]
        )
        for field in {k for out, base in pairs for k in out.keys() | base.keys()}
    }


def gate_pvalue(diffs: Sequence[float], *, permutations: int, rng: random.Random) -> float:
    observed = statistics.fmean(diffs)
    exceed = sum(
        statistics.fmean([diff if rng.random() < 0.5 else -diff for diff in diffs]) >= observed
        for _ in range(permutations)
    )
    return (exceed + 1) / (permutations + 1)


class WinnerPicker:
    """Picks the best viable arm by primary metric, then tiebreak; a declared ``viable`` field gates entry."""

    @classmethod
    def pick(cls, results: Sequence[ArmResult], spec: BakeoffSpec) -> ArmResult | None:
        declared = cls.viable_declared(results)
        viable = [result for result in results if cls.is_viable(result, declared=declared)]
        return max(viable, key=lambda result: cls.sort_key(result, spec)) if viable else None

    @staticmethod
    def viable_declared(results: Sequence[ArmResult]) -> bool:
        return any(VIABLE_METRIC in result.metrics for result in results)

    @staticmethod
    def is_viable(result: ArmResult, *, declared: bool) -> bool:
        return not declared or result.metrics.get(VIABLE_METRIC, 0.0) >= 1.0

    @staticmethod
    def sort_key(result: ArmResult, spec: BakeoffSpec) -> tuple[float, float]:
        return (
            result.metrics[spec.primary_metric],
            result.metrics[spec.tiebreak] if spec.tiebreak else 0.0,
        )

    @classmethod
    def rank_key(cls, result: ArmResult, spec: BakeoffSpec, *, declared: bool) -> tuple[bool, float, float]:
        return (cls.is_viable(result, declared=declared), *cls.sort_key(result, spec))


def passed_gate(
    winner: ArmResult | None,
    spec: BakeoffSpec,
    outputs: Mapping[str, Sequence[Mapping[str, object]]],
    settings: BakeoffSettings,
) -> bool:
    baseline = spec.arms[0].name
    if winner is None or winner.arm == baseline:
        return False
    diffs = [
        float(win[spec.primary_metric]) - float(base[spec.primary_metric])
        for win, base in zip(outputs[winner.arm], outputs[baseline], strict=True)
    ]
    alpha = settings.alpha / (len(spec.arms) - 1)
    return statistics.fmean(diffs) >= settings.min_lift and (
        gate_pvalue(diffs, permutations=settings.permutations, rng=random.Random(settings.seed)) < alpha
    )


async def run(spec: BakeoffSpec) -> Leaderboard:
    """Run every arm over the shared corpus and rank them into a gated :class:`Leaderboard`.

    Each arm runs the spec's task across the corpus under bounded concurrency, producing
    per-arm metrics (numeric-field means plus cross-arm ``agreement``) and per-field
    disagreement against the baseline arm. :class:`WinnerPicker` selects the winner and a
    paired permutation test on the primary metric decides ``passed_gate``.

    Args:
        spec: The task, corpus, arms, and ranking keys defining the bake-off.

    Returns:
        The ranked results, the winner's name, and the statistical go/no-go verdict.
    """
    settings = load(BakeoffSettings)
    outputs = {arm.name: await run_arm(arm, spec, concurrency=settings.concurrency) for arm in spec.arms}
    baseline = spec.arms[0].name
    results = [
        ArmResult(
            arm=arm.name,
            metrics=mean_by_field(outputs[arm.name]) | {AGREEMENT_METRIC: agreement(arm.name, outputs)},
            per_field_disagreement=field_disagreement(outputs[arm.name], outputs[baseline]),
        )
        for arm in spec.arms
    ]
    declared = WinnerPicker.viable_declared(results)
    ranked = tuple(
        sorted(results, key=lambda result: WinnerPicker.rank_key(result, spec, declared=declared), reverse=True)
    )
    winner = WinnerPicker.pick(ranked, spec)
    return Leaderboard(
        results=ranked,
        winner=winner.arm if winner else "",
        passed_gate=passed_gate(winner, spec, outputs, settings),
    )


def load_spec(target: str) -> BakeoffSpec:
    module_name, _, attr = target.partition(":")
    obj = getattr(importlib.import_module(module_name), attr or "spec")
    spec = obj() if callable(obj) else obj
    if not isinstance(spec, BakeoffSpec):
        raise BakeoffError(f"{target} resolved to {type(spec).__name__}, expected a BakeoffSpec")
    return spec


@click.group("bakeoff")
def cli() -> None:
    """Run A/B/N endpoint bake-offs and print the gated leaderboard."""


@cli.command("run")
@click.argument("spec")
@json_option
@coro
async def run_command(spec: str, as_json: bool) -> None:
    """Run the BakeoffSpec exported by SPEC (a ``module`` or ``module:attr`` path)."""
    emit(asdict(await run(load_spec(spec))), as_json=as_json)
