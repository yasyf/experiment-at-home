from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from athome.errors import AthomeError
from athome.train.backend import NoBackendAvailable, TrainBackend, select
from athome.train.data import (
    DpoExample,
    SftExample,
    normalize,
    render_mlx_jsonl,
    render_tinker_dpo,
    render_tinker_sft,
    render_trl,
)
from athome.train.local import LocalBackend
from athome.train.modal import ModalTrainBackend, projected_usd
from athome.train.spec import TrainSettings, TrainSpec, spend_cap
from athome.train.tinker import TinkerBackend

if TYPE_CHECKING:
    from athome.bakeoff import BakeoffSpec

SAMPLE_SIZE = 16


class PreflightFailure(AthomeError):
    """Raised when a mandatory training probe fails before the backend starts."""


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """The ordered training probes that passed before a backend was allowed to start.

    Attributes:
        checks: Human-readable results in execution order.
    """

    checks: tuple[str, ...]


def render_sft(example: SftExample, backend: TrainBackend, spec: TrainSpec, scratch: Path, *, index: int) -> None:
    match backend:
        case TinkerBackend():
            render_tinker_sft(example, spec.base.mlx)
        case LocalBackend(settings=settings):
            render_mlx_jsonl(
                [example],
                scratch / str(index),
                val_fraction=settings.val_fraction,
                seed=spec.hyperparams.seed,
            )
        case ModalTrainBackend():
            render_trl([example], method="sft")


def render_dpo(example: DpoExample, backend: TrainBackend, spec: TrainSpec) -> None:
    match backend:
        case TinkerBackend():
            render_tinker_dpo(example, spec.base.mlx)
        case ModalTrainBackend():
            render_trl([example], method="dpo")


async def render_dataset(spec: TrainSpec, backend: TrainBackend) -> tuple[int, int]:
    match spec.method:
        case "sft":
            try:
                examples = await normalize(spec.dataset, method="sft")
            except Exception as error:
                raise PreflightFailure(f"dataset normalization failed: {error}") from error
            with TemporaryDirectory() as directory:
                for index, example in enumerate(examples[:SAMPLE_SIZE]):
                    try:
                        render_sft(example, backend, spec, Path(directory), index=index)
                    except Exception as error:
                        raise PreflightFailure(f"dataset row {index} failed to render: {error}") from error
        case "dpo":
            try:
                examples = await normalize(spec.dataset, method="dpo")
            except Exception as error:
                raise PreflightFailure(f"dataset normalization failed: {error}") from error
            for index, example in enumerate(examples[:SAMPLE_SIZE]):
                try:
                    render_dpo(example, backend, spec)
                except Exception as error:
                    raise PreflightFailure(f"dataset row {index} failed to render: {error}") from error
    return len(examples), min(len(examples), SAMPLE_SIZE)


async def preflight(spec: TrainSpec, *, evaluation: BakeoffSpec, settings: TrainSettings) -> PreflightReport:
    """Validate backend, data, budget, and evaluation before training can start.

    Args:
        spec: The fine-tuning request whose backend and dataset are probed.
        evaluation: The bake-off definition that will score the trained artifact.
        settings: The already-loaded training settings used for backend selection.

    Returns:
        The ordered checks that passed.

    Raises:
        PreflightFailure: A backend cannot run the method, a sampled row cannot render,
            a Modal projection crosses its cap, or the evaluation is incomplete.
    """
    checks: list[str] = []
    try:
        backend = select(spec, settings)
    except NoBackendAvailable as error:
        raise PreflightFailure(str(error)) from error
    checks.append(f"backend availability: {backend.name} supports {spec.method}")

    examples, rendered = await render_dataset(spec, backend)
    checks.append(f"dataset renders: {rendered}/{examples} examples sampled")

    match backend:
        case ModalTrainBackend(settings=modal_settings):
            projected = projected_usd(spec, modal_settings)
            cap = spend_cap(spec, modal_settings.spend_cap_usd)
            if not projected <= cap:
                raise PreflightFailure(f"modal projected cost ${projected:.4f} exceeds cap ${cap:.4f}")
            checks.append(f"modal cost projection: ${projected:.4f} <= ${cap:.4f}")
        case _:
            checks.append("modal cost projection: skipped (backend != modal)")

    if not evaluation.arms:
        raise PreflightFailure("evaluation requires at least one arm")
    if not evaluation.primary_metric:
        raise PreflightFailure("evaluation requires a primary metric")
    checks.append(f"evaluation sanity: {len(evaluation.arms)} arms, primary metric {evaluation.primary_metric!r}")
    return PreflightReport(checks=tuple(checks))
