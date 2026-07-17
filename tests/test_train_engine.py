from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from athome.train.engine import (
    CrossEntropy,
    Custom,
    ScoreOp,
    SnapshotOp,
    TrainDone,
    TrainOp,
    execute,
    projection,
    token_count,
)
from athome.train.spec import TinkerPrice
from tests.tinker_fakes import Boom, FakeAdamParams, FakeTraining, install_sdk, make_datum

if TYPE_CHECKING:
    from collections.abc import Sequence

    from athome.train.engine import Op, OpResult

LR = 2e-4
PRICE = TinkerPrice(prefill=0.13, sample=0.40, train=0.40)


@pytest.fixture(autouse=True)
def sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    install_sdk(monkeypatch)


def client(*, fail_fb: int | None = None, fail_submit: int | None = None) -> FakeTraining:
    return FakeTraining("Qwen/Qwen3-8B", 16, 1729, (True, True, False), fail_fb=fail_fb, fail_submit=fail_submit)


async def collect(training: FakeTraining, schedule: Sequence[Op]) -> list[OpResult]:
    return [result async for result in execute(training, schedule)]


def train_ops(count: int, *, length: int = 4) -> tuple[TrainOp, ...]:
    return tuple(TrainOp((make_datum(length),), CrossEntropy(), LR) for _ in range(count))


async def test_a_schedule_of_train_ops_costs_one_cycle_per_step() -> None:
    """The pipelining win: submit-ahead plus the fused optim collapses three cycles per step to one."""
    training = client()

    results = await collect(training, train_ops(5))

    assert len(results) == 5
    assert all(isinstance(result, TrainDone) for result in results)
    assert training.clock.cycles == 5


async def test_the_optim_is_submitted_before_its_forward_backward_is_awaited() -> None:
    training = client()

    await collect(training, train_ops(4))

    events = training.clock.events
    assert all(events.index(f"submit:optim:{step}") < events.index(f"await:fb:{step}") for step in range(1, 5))


async def test_the_next_step_is_submitted_before_the_current_one_is_awaited() -> None:
    training = client()

    await collect(training, train_ops(4))

    events = training.clock.events
    assert all(events.index(f"submit:fb:{step + 1}") < events.index(f"await:fb:{step}") for step in range(1, 4))


async def test_a_snapshot_and_its_eval_ride_between_the_step_and_the_next() -> None:
    training = client()
    schedule = (
        TrainOp((make_datum(3),), CrossEntropy(), LR),
        SnapshotOp("run-step00001", 604_800, (make_datum(2),)),
        TrainOp((make_datum(3),), CrossEntropy(), LR),
        SnapshotOp("run-sft-2", None, (make_datum(2),)),
    )

    results = await collect(training, schedule)

    events = training.clock.events
    assert (
        events.index("submit:optim:1")
        < events.index("submit:save:1")
        < events.index("submit:forward:1")
        < events.index("submit:fb:2")
    )
    assert training.saves == [("run-step00001", 604_800), ("run-sft-2", None)]
    assert [type(result).__name__ for result in results] == ["TrainDone", "SnapshotDone", "TrainDone", "SnapshotDone"]


async def test_the_stream_drains_every_op_to_exhaustion() -> None:
    training = client()
    schedule = (
        *train_ops(3),
        ScoreOp((make_datum(2), make_datum(2))),
        SnapshotOp("run-sft-3", None, ()),
    )

    results = await collect(training, schedule)

    assert len(results) == len(schedule)
    assert not training.clock.queue


async def test_a_failing_op_raises_in_order_with_earlier_results_already_delivered() -> None:
    training = client(fail_fb=3)
    schedule = train_ops(5)
    delivered: list[OpResult] = []

    with pytest.raises(Boom):
        async for result in execute(training, schedule):
            delivered.append(result)

    assert len(delivered) == 2
    assert [result.op for result in delivered] == list(schedule[:2])


async def test_a_submission_failure_raises_only_after_pending_ops_drain() -> None:
    """A failed submit must not orphan already-submitted billable work: pending ops drain first."""
    training = client(fail_submit=3)
    schedule = train_ops(5)
    delivered: list[OpResult] = []

    with pytest.raises(Boom):
        async for result in execute(training, schedule):
            delivered.append(result)

    assert [result.op for result in delivered] == list(schedule[:2])


async def test_the_queue_depth_never_exceeds_the_submit_ahead_bound() -> None:
    """The complement of the read-ahead test: fb s+2 may only be submitted after step s drains."""
    training = client()

    await collect(training, train_ops(5))

    events = training.clock.events
    assert all(events.index(f"submit:fb:{step + 2}") > events.index(f"await:fb:{step}") for step in range(1, 4))


async def test_a_custom_loss_routes_to_the_custom_forward_backward() -> None:
    training = client()

    def loss(datums: object, logprobs: object) -> tuple[object, dict[str, float]]:
        return None, {}

    await collect(training, (TrainOp((make_datum(3),), Custom(loss), LR),))

    assert training.forward_backward == []
    assert len(training.custom) == 1
    assert training.custom[0].loss_fn is loss
    assert training.custom[0].loss_type_input == "logprobs"


async def test_each_step_constructs_its_own_optimizer_params() -> None:
    training = client()
    schedule = tuple(TrainOp((make_datum(2),), CrossEntropy(), lr) for lr in (1e-4, 2e-4, 3e-4))

    await collect(training, schedule)

    assert training.optim == [FakeAdamParams(1e-4), FakeAdamParams(2e-4), FakeAdamParams(3e-4)]


def test_projection_folds_train_passes_score_and_eval_prefill() -> None:
    def loss(datums: object, logprobs: object) -> tuple[object, dict[str, float]]:
        return None, {}

    schedule = (
        TrainOp((make_datum(10),), CrossEntropy(), LR),
        TrainOp((make_datum(10),), Custom(loss), LR),
        ScoreOp((make_datum(5), make_datum(5))),
        SnapshotOp("s", 604_800, (make_datum(4),)),
        SnapshotOp("bare", None, ()),
    )

    expected = (10 * 0.40 * 1 + 10 * 0.40 * 2 + 10 * 0.13 + 4 * 0.13) / 1e6
    assert projection(schedule, PRICE) == pytest.approx(expected)


def test_token_count_sums_input_lengths() -> None:
    assert token_count((make_datum(3), make_datum(7))) == 10
