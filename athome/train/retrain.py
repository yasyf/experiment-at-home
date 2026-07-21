from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from athome.train.spec import require_servable

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from athome.llm.spend import SpendGuard
    from athome.progress import RunSink
    from athome.train.gate import GateVerdict
    from athome.train.spec import (
        Adapter,
        CheckpointPolicy,
        EvalRow,
        SavedCheckpoint,
        TrainReport,
        TrainSpec,
    )
    from athome.train.tinker import TinkerBackend


@dataclass(frozen=True, slots=True)
class RetrainOutcome:
    """Everything one :func:`retrain` produced, with no side effect taken on the caller's behalf.

    The consumer decides what to do with the verdict — register, promote, journal, prune — after
    ``retrain`` returns; the orchestrator itself writes to no registry.

    Attributes:
        report: The full training report ``fit`` produced, including every saved checkpoint.
        best: The checkpoint ``select`` argmaxed over ``report.checkpoints``, the one materialized.
        adapter: The servable adapter ``best`` was materialized into.
        served: The scores ``artifact_scorer`` read off the materialized adapter.
        verdict: The gate's decision over ``served``.
    """

    report: TrainReport
    best: SavedCheckpoint
    adapter: Adapter
    served: dict[str, float]
    verdict: GateVerdict


async def retrain(
    backend: TinkerBackend,
    spec: TrainSpec,
    *,
    checkpoints: CheckpointPolicy,
    eval_rows: Sequence[EvalRow] | None,
    budget: SpendGuard,
    select: Callable[[SavedCheckpoint], float],
    artifact_scorer: Callable[[Adapter], dict[str, float]],
    gate: Callable[[dict[str, float]], GateVerdict],
    work_dir: Path,
    sink: RunSink,
) -> RetrainOutcome:
    """Train, pick the strongest checkpoint, materialize it, score the artifact, and gate it.

    The pipeline shape lives here; the domain arrives as callables. ``fit`` runs the whole training
    schedule and scores every checkpoint's eval rows on the turnstile; ``select`` scores each of the
    resulting checkpoints and the argmax is materialized; ``artifact_scorer`` reads the servable
    adapter locally once ``fit``'s stream has fully drained; ``gate`` turns those scores into a
    verdict. Nothing here registers, promotes, or journals a decision — those side effects belong to
    the caller, after this returns, against the caller's own registry.

    Args:
        backend: The Tinker backend that trains and materializes; checkpoint selection is
            Tinker-shaped, so no other backend qualifies.
        spec: The fine-tuning request forwarded to ``fit`` and ``materialize``.
        checkpoints: Which fractions of the run to snapshot, passed straight to ``fit``.
        eval_rows: Pre-tokenized rows scored against every checkpoint's weights, or None.
        budget: The spend envelope threaded verbatim into ``fit``; the training schedule's projected
            cost reserves against it before the first billable call and reconciles to actuals as the
            run drains. ``retrain`` mints no envelope of its own, so this caller-owned cap is the one
            ledger every billable call in the run debits.
        select: Scores one checkpoint; ``retrain`` materializes the argmax over
            ``report.checkpoints``. A raising ``select`` propagates and fails the run.
        artifact_scorer: Reads scores off the materialized adapter, synchronously and locally.
        gate: Turns the artifact's scores into a promote/reject verdict.
        work_dir: This run's directory, where ``materialize`` writes the adapter.
        sink: The run journal ``fit`` appends each step and checkpoint to.

    Returns:
        The report, the selected checkpoint, its materialized adapter, the artifact scores, and the
        gate's verdict.

    Raises:
        UnservableBase: The base has no mlx-lm LoRA counterpart to materialize into, so the run
            would bill a full hosted fit only to refuse at ``materialize``; it aborts before ``fit``.
    """
    require_servable(spec.base, kind="Tinker")
    report = await backend.fit(
        spec, sink=sink, budget=budget, checkpoints=checkpoints, eval_rows=eval_rows, run_tag=work_dir.name
    )
    best = max(report.checkpoints, key=select)
    adapter = await backend.materialize(best, spec, work_dir=work_dir, cost=report.train_cost_usd)
    served = artifact_scorer(adapter)
    return RetrainOutcome(report, best, adapter, served, gate(served))
