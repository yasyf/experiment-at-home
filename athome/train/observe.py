from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from athome.train.spec import CheckpointPolicy

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from athome.llm.spend import SpendGuard
    from athome.progress import RunSink
    from athome.train.spec import EvalRow, SavedCheckpoint, ScoredSequence, TrainReport, TrainSpec
    from athome.train.tinker import TinkerBackend


@dataclass(frozen=True, slots=True)
class ObserveOutcome:
    """One evaluate-only pass: the fit report, the checkpoint chosen from it, and its scores.

    Fire-probability and AUC math are out of scope; a consumer computes them over ``scored``.

    Attributes:
        report: Everything the ``fit`` produced — per-step records and every saved checkpoint.
        best: The checkpoint ``select`` ranked highest across ``report.checkpoints``; its
            ``sampler_path`` is the ``tinker://`` sampler address that was scored.
        scored: ``best``'s eval rows scored against its weights, order-preserving.
    """

    report: TrainReport
    best: SavedCheckpoint
    scored: tuple[ScoredSequence, ...]


async def observe(
    backend: TinkerBackend,
    spec: TrainSpec,
    rows: Sequence[EvalRow],
    *,
    budget: SpendGuard,
    select: Callable[[SavedCheckpoint], float],
    sink: RunSink,
    checkpoints: CheckpointPolicy = CheckpointPolicy(),
    eval_rows: Sequence[EvalRow] | None = None,
) -> ObserveOutcome:
    """Fit ``spec``, pick the best checkpoint by ``select``, and score it under one spend envelope.

    The sanctioned evaluate-only entrypoint: ``fit``, then ``argmax(select)`` over the saved
    checkpoints, then ``score`` — with ``budget`` threaded through both billable calls, so ``fit``'s
    recorded actuals draw down what ``score`` may still reserve. Hosted-only bases are welcome;
    nothing here requires a servable base. The caller owns ``sink`` and its retention: ``observe``
    stages nothing and deletes nothing.

    Args:
        backend: The Tinker backend to fit and score against.
        spec: The fine-tuning request; its base prices the scoring calls.
        rows: Pre-tokenized weighted rows scored against the selected checkpoint, in return order.
        budget: The one spend envelope spanning both the fit and the score.
        select: Ranks a saved checkpoint; the argmax over ``report.checkpoints`` is scored.
        sink: The run journal the fit writes to; the caller owns it and its retention.
        checkpoints: Which fractions of the run to snapshot; the final step is always snapshotted.
        eval_rows: Rows scored against every checkpoint during the fit, or None.

    Returns:
        The fit report, the selected checkpoint, and that checkpoint's scored rows.

    Raises:
        SpendExceeded: The envelope cannot cover the fit's projection, or the fit's actuals leave
            too little headroom for the score's.
    """
    report = await backend.fit(
        spec, budget=budget, sink=sink, checkpoints=checkpoints, eval_rows=eval_rows, run_tag=f"obs-{uuid4().hex[:8]}"
    )
    best = max(report.checkpoints, key=select)
    return ObserveOutcome(report, best, await backend.score(best.sampler_path, rows, base=spec.base, budget=budget))
