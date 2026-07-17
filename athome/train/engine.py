from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable, Sequence

    import tinker
    import torch

    from athome.train.spec import TinkerPrice

    type LossFn = Callable[[Sequence[tinker.Datum], Sequence[torch.Tensor]], tuple[torch.Tensor, dict[str, float]]]
    type Op = TrainOp | ScoreOp | SnapshotOp
    type OpResult = TrainDone | ScoreDone | SnapshotDone
    type InFlight = TrainFlight | ScoreFlight | SnapshotFlight

SUBMIT_AHEAD = 1
MAX_PENDING = 1 + SUBMIT_AHEAD


def token_count(datums: Iterable[tinker.Datum]) -> int:
    """The total input-token count across ``datums`` — what the metered classes bill against."""
    return sum(datum.model_input.length for datum in datums)


@dataclass(frozen=True, slots=True)
class CrossEntropy:
    """Tinker's named ``cross_entropy`` loss: one forward-backward pass per step."""

    passes: ClassVar[int] = 1


@dataclass(frozen=True, slots=True)
class Custom:
    """A client-side loss (DPO): a forward pass and a surrogate backward, so two passes per step.

    Both passes are priced at the train rate. If Tinker bills the preliminary forward at the
    cheaper prefill rate, the projection overstates — deliberately conservative for a spend cap,
    which must over-reserve rather than under.
    """

    fn: LossFn
    passes: ClassVar[int] = 2


type Loss = CrossEntropy | Custom


@dataclass(frozen=True, slots=True)
class TrainOp:
    """One optimization step: a forward-backward and its optim step as a single value.

    Binding the pair into one op means the inefficient shape — awaiting the forward-backward
    before submitting the optim — has no representation for a consumer to write.

    Attributes:
        datums: The step's training batch.
        loss: The loss to train against, which also carries its billing multiplier.
        lr: The learning rate for this step's fresh optimizer params, so a schedule falls out free.
    """

    datums: tuple[tinker.Datum, ...]
    loss: Loss
    lr: float


@dataclass(frozen=True, slots=True)
class ScoreOp:
    """A batched forward pass scoring ``datums``, billed as prefill."""

    datums: tuple[tinker.Datum, ...]


@dataclass(frozen=True, slots=True)
class SnapshotOp:
    """Save the current weights for sampling, optionally scoring eval datums against them.

    The eval datums live on the snapshot, so which weights they were scored against is carried
    by the op rather than inferred from adjacency.

    Attributes:
        name: The sampler checkpoint name.
        ttl_seconds: How long the saved weights live, or None to keep them forever.
        eval: The datums to score against these exact weights, or empty for a bare save.
    """

    name: str
    ttl_seconds: int | None
    eval: tuple[tinker.Datum, ...] = ()


@dataclass(frozen=True, slots=True)
class TrainDone:
    """The result of a :class:`TrainOp`: its op and the forward-backward output for the fold."""

    op: TrainOp
    output: tinker.ForwardBackwardOutput


@dataclass(frozen=True, slots=True)
class ScoreDone:
    """The result of a :class:`ScoreOp`: its op and one logprob output per scored datum."""

    op: ScoreOp
    outputs: Sequence[dict[str, tinker.TensorData]]


@dataclass(frozen=True, slots=True)
class SnapshotDone:
    """The result of a :class:`SnapshotOp`: the saved path and eval outputs, or None when bare."""

    op: SnapshotOp
    path: str
    outputs: Sequence[dict[str, tinker.TensorData]] | None


@dataclass(frozen=True, slots=True)
class TrainFlight:
    op: TrainOp
    fb: tinker.APIFuture[tinker.ForwardBackwardOutput]
    optim: tinker.APIFuture[tinker.OptimStepResponse]


@dataclass(frozen=True, slots=True)
class ScoreFlight:
    op: ScoreOp
    forward: tinker.APIFuture[tinker.ForwardBackwardOutput]


@dataclass(frozen=True, slots=True)
class SnapshotFlight:
    op: SnapshotOp
    save: tinker.APIFuture[tinker.SaveWeightsForSamplerResponse]
    forward: tinker.APIFuture[tinker.ForwardBackwardOutput] | None


def adam_params(lr: float) -> tinker.AdamParams:
    import tinker

    return tinker.AdamParams(learning_rate=lr)


def depth(pending: deque[InFlight]) -> int:
    return sum(isinstance(flight, TrainFlight) for flight in pending)


async def submit(client: tinker.TrainingClient, op: Op) -> InFlight:
    match op:
        case TrainOp(datums=datums, loss=CrossEntropy(), lr=lr):
            fb = await client.forward_backward_async(list(datums), "cross_entropy")
            return TrainFlight(op, fb, await client.optim_step_async(adam_params(lr)))
        case TrainOp(datums=datums, loss=Custom(fn=fn), lr=lr):
            fb = await client.forward_backward_custom_async(list(datums), fn, loss_type_input="logprobs")
            return TrainFlight(op, fb, await client.optim_step_async(adam_params(lr)))
        case ScoreOp(datums=datums):
            return ScoreFlight(op, await client.forward_async(list(datums), "cross_entropy"))
        case SnapshotOp(name=name, ttl_seconds=ttl, eval=eval_datums):
            save = await client.save_weights_for_sampler_async(name, ttl_seconds=ttl)
            forward = await client.forward_async(list(eval_datums), "cross_entropy") if eval_datums else None
            return SnapshotFlight(op, save, forward)


async def drain(flight: InFlight) -> OpResult:
    match flight:
        case TrainFlight(op=op, fb=fb, optim=optim):
            output = await fb.result_async()
            await optim.result_async()
            return TrainDone(op, output)
        case ScoreFlight(op=op, forward=forward):
            return ScoreDone(op, (await forward.result_async()).loss_fn_outputs)
        case SnapshotFlight(op=op, save=save, forward=forward):
            path = (await save.result_async()).path
            return SnapshotDone(op, path, (await forward.result_async()).loss_fn_outputs if forward else None)


async def execute(client: tinker.TrainingClient, schedule: Sequence[Op]) -> AsyncIterator[OpResult]:
    """Submit ``schedule``'s ops to ``client`` and yield their results in submission order.

    This is the only code that holds an ``APIFuture``. It keeps the server's queue non-empty by
    submitting up to :data:`MAX_PENDING` compute-bearing ops ahead of the one it is draining —
    every :class:`TrainOp` submits its forward-backward and optim together before either result
    is awaited, so a step costs one clock cycle instead of three. Snapshots and scores ride the
    same queue between steps without stalling it. Results drain strictly in order; a failed op
    raises from its own yield, with every earlier result already delivered — and an op whose
    *submission* fails raises only after every already-submitted op has drained, so no billable
    work goes unobserved.

    Args:
        client: The Tinker training client whose turnstile executes the ops.
        schedule: The ops to run, in order.

    Yields:
        One :data:`OpResult` per op, in submission order.
    """
    pending: deque[InFlight] = deque()
    ops = iter(schedule)
    deferred: Exception | None = None
    while True:
        while deferred is None and depth(pending) < MAX_PENDING and (op := next(ops, None)) is not None:
            try:
                pending.append(await submit(client, op))
            except Exception as error:  # noqa: BLE001 — deferred until pending ops drain, then re-raised
                deferred = error
        if not pending:
            if deferred is not None:
                raise deferred
            return
        yield await drain(pending.popleft())


def op_cost(op: Op, price: TinkerPrice) -> float:
    match op:
        case TrainOp(datums=datums, loss=loss):
            return price.train * token_count(datums) * loss.passes
        case ScoreOp(datums=datums):
            return price.prefill * token_count(datums)
        case SnapshotOp(eval=eval_datums):
            return price.prefill * token_count(eval_datums)


def projection(schedule: Sequence[Op], price: TinkerPrice) -> float:
    """The dollar cost ``schedule`` will bill at ``price`` — a pure fold over the same value ``execute`` runs.

    A :class:`TrainOp` bills its tokens at the train rate times the loss's pass count; a
    :class:`ScoreOp` and a snapshot's eval datums bill at the prefill rate; a bare snapshot is
    free. Folding the exact schedule that runs is what makes projected and billed cost the same.
    """
    return sum(op_cost(op, price) for op in schedule) / 1e6
