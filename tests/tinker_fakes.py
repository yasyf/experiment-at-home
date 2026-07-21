from __future__ import annotations

import importlib.machinery
import sys
from collections import deque
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pytest

LOGPROB = -0.5
CANNED = {"loss": 0.61, "margin": 0.2, "accuracy": 1.0}
type LossFn = Callable[..., tuple[object, dict[str, float]]]


class Boom(RuntimeError):
    """An op's failure — raised at submission time or surfaced when its result is drained."""


class StateNameCollision(RuntimeError):
    """A ``save_state`` at a name already saved, with ``overwrite=False`` — the real SDK refuses this."""


@dataclass(frozen=True, slots=True)
class FakeTensor:
    data: list[float] | list[int]
    dtype: str
    shape: list[int]


@dataclass(frozen=True, slots=True)
class FakeModelInput:
    ids: list[int]

    @staticmethod
    def from_ints(ids: list[int]) -> FakeModelInput:
        return FakeModelInput(ids)

    @property
    def length(self) -> int:
        return len(self.ids)


@dataclass(frozen=True, slots=True)
class FakeDatum:
    model_input: FakeModelInput
    loss_fn_inputs: dict[str, FakeTensor]


@dataclass(frozen=True, slots=True)
class FakeAdamParams:
    learning_rate: float


@dataclass(frozen=True, slots=True)
class FakeSamplingParams:
    max_tokens: int
    temperature: float
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class FakeOutput:
    loss_fn_outputs: list[dict[str, FakeTensor]]
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FakeSaved:
    path: str


@dataclass(frozen=True, slots=True)
class FakeUrl:
    url: str


@dataclass(frozen=True, slots=True)
class FakeWeightsInfo:
    """A saved checkpoint's LoRA shape, as the ``weights_info`` endpoint reports it back."""

    base_model: str
    is_lora: bool
    lora_rank: int
    train_mlp: bool
    train_attn: bool
    train_unembed: bool


@dataclass(frozen=True, slots=True)
class FakeFuture:
    """A resolved ``APIFuture`` — the rest client hands weights-info back through one of these."""

    value: object

    async def result_async(self, timeout: float | None = None) -> object:
        return self.value


@dataclass(slots=True)
class FakeRestClient:
    """The rest client ``create_rest_client`` returns; answers ``weights_info`` from the saved shapes."""

    weights_info: dict[str, FakeWeightsInfo]

    def get_weights_info_by_tinker_path(self, tinker_path: str) -> FakeFuture:
        return FakeFuture(self.weights_info[tinker_path])


@dataclass(frozen=True, slots=True)
class CustomCall:
    datums: list[FakeDatum]
    loss_fn: LossFn
    loss_type_input: str


def logprobs_for(datums: Sequence[FakeDatum]) -> list[dict[str, FakeTensor]]:
    return [
        {
            "logprobs": FakeTensor(
                data=[LOGPROB] * len(datum.loss_fn_inputs["target_tokens"].data),
                dtype="float32",
                shape=[len(datum.loss_fn_inputs["target_tokens"].data)],
            )
        }
        for datum in datums
    ]


def make_datum(length: int, *, weight: float = 1.0) -> FakeDatum:
    """A fake ``Datum`` of ``length`` target tokens, each weighted ``weight``."""
    return FakeDatum(
        model_input=FakeModelInput([1] * length),
        loss_fn_inputs={
            "target_tokens": FakeTensor([1] * length, "int64", [length]),
            "weights": FakeTensor([weight] * length, "float32", [length]),
        },
    )


@dataclass(slots=True)
class ClockSlot:
    kind: str
    seq: int
    value: object
    executed: bool = False


@dataclass(slots=True)
class Clock:
    """Tinker's serial turnstile as deterministic cycle arithmetic, driven lazily by result waits.

    Ops queue at submission; a result wait executes the queue up to its op. Executing the earliest
    op costs one cycle, but a forward-backward whose paired optim is already queued executes both in
    that one cycle — the pipelining win. A fresh training step submitted into a queue that a prior
    execution has drained pays one idle cycle for the stall. So a naive await-each loop costs three
    cycles per step, submitting the pair without reading ahead costs two, and the pipelined stream
    costs one.
    """

    events: list[str] = field(default_factory=list)
    cycles: int = 0
    queue: deque[ClockSlot] = field(default_factory=deque)
    has_executed: bool = False
    fb_count: int = 0

    def submit(self, kind: str, seq: int, value: object) -> ClockFuture:
        if kind == "fb" and self.has_executed and not self.queue:
            self.cycles += 1
            self.events.append(f"idle:{seq}")
        slot = ClockSlot(kind, seq, value)
        self.queue.append(slot)
        self.events.append(f"submit:{kind}:{seq}")
        return ClockFuture(self, slot)

    def drive(self, target: ClockSlot) -> None:
        while not target.executed:
            head = self.queue.popleft()
            head.executed = True
            if head.kind == "fb" and self.queue and self.queue[0].kind == "optim":
                self.queue.popleft().executed = True
            self.cycles += 1
            self.has_executed = True


@dataclass(slots=True)
class ClockFuture:
    clock: Clock
    slot: ClockSlot

    async def result_async(self, timeout: float | None = None) -> object:
        self.clock.events.append(f"await:{self.slot.kind}:{self.slot.seq}")
        self.clock.drive(self.slot)
        if isinstance(self.slot.value, BaseException):
            raise self.slot.value
        return self.slot.value


@dataclass(slots=True)
class FakeTraining:
    """Records every call and answers each forward with a flat ``LOGPROB`` per target token.

    Every submission threads through its :class:`Clock`, so the event log and cycle count reflect
    exactly how the executor drove the turnstile. ``fail_fb`` makes one step's forward-backward
    surface a :class:`Boom` when its result is drained.
    """

    base_model: str
    rank: int
    seed: int
    toggles: tuple[bool, bool, bool]
    fail_fb: int | None = None
    fail_submit: int | None = None
    clock: Clock = field(default_factory=Clock)
    forward_backward: list[tuple[list[FakeDatum], str]] = field(default_factory=list)
    custom: list[CustomCall] = field(default_factory=list)
    forward: list[list[FakeDatum]] = field(default_factory=list)
    optim: list[FakeAdamParams] = field(default_factory=list)
    saves: list[tuple[str, int | None]] = field(default_factory=list)
    state_saves: list[tuple[str, int | None]] = field(default_factory=list)
    weights_info: dict[str, FakeWeightsInfo] = field(default_factory=dict)
    seeded_from: str | None = None

    def _fb_value(self, datums: Sequence[FakeDatum], metrics: dict[str, float]) -> object:
        if self.clock.fb_count == self.fail_fb:
            return Boom(f"fb {self.clock.fb_count}")
        return FakeOutput(logprobs_for(datums), metrics=metrics)

    async def forward_backward_async(self, datums: list[FakeDatum], loss_fn: str) -> ClockFuture:
        self.forward_backward.append((datums, loss_fn))
        self.clock.fb_count += 1
        if self.clock.fb_count == self.fail_submit:
            raise Boom(f"submit {self.clock.fb_count}")
        return self.clock.submit("fb", self.clock.fb_count, self._fb_value(datums, {}))

    async def forward_backward_custom_async(
        self, datums: list[FakeDatum], loss_fn: LossFn, *, loss_type_input: str = "logprobs"
    ) -> ClockFuture:
        self.custom.append(CustomCall(datums, loss_fn, loss_type_input))
        self.clock.fb_count += 1
        return self.clock.submit("fb", self.clock.fb_count, self._fb_value(datums, dict(CANNED)))

    async def optim_step_async(self, adam_params: FakeAdamParams) -> ClockFuture:
        self.optim.append(adam_params)
        return self.clock.submit("optim", self.clock.fb_count, None)

    async def forward_async(self, datums: list[FakeDatum], loss_fn: str) -> ClockFuture:
        self.forward.append(datums)
        return self.clock.submit("forward", self.clock.fb_count, FakeOutput(logprobs_for(datums)))

    async def save_weights_for_sampler_async(self, name: str, ttl_seconds: int | None = None) -> ClockFuture:
        self.saves.append((name, ttl_seconds))
        return self.clock.submit("save", self.clock.fb_count, FakeSaved(f"tinker://run/{name}"))

    async def save_state_async(self, name: str, ttl_seconds: int | None = None, overwrite: bool = False) -> ClockFuture:
        """Save the training state (weights+optimizer) to a distinct ``state/`` namespace and register its shape.

        ``weights_info`` is the service's own registry, shared across every client it mints, so it
        is the cross-run record of which state names already exist. A second ``save_state`` at a
        name already saved refuses under ``overwrite=False``, exactly as the real SDK does — the
        collision a fresh re-run of a completed spec would otherwise hit at its immortal final save.
        """
        path = f"tinker://run/state/{name}"
        if path in self.weights_info and not overwrite:
            raise StateNameCollision(name)
        self.state_saves.append((name, ttl_seconds))
        self.weights_info[path] = FakeWeightsInfo(
            base_model=self.base_model,
            is_lora=True,
            lora_rank=self.rank,
            train_mlp=self.toggles[0],
            train_attn=self.toggles[1],
            train_unembed=self.toggles[2],
        )
        return self.clock.submit("state", self.clock.fb_count, FakeSaved(path))


@dataclass(slots=True)
class FakeService:
    api_key: str
    fail_fb: int | None = None
    clients: list[FakeTraining] = field(default_factory=list)
    weights_info: dict[str, FakeWeightsInfo] = field(default_factory=dict)

    async def create_lora_training_client_async(
        self, base_model: str, rank: int, seed: int, train_mlp: bool, train_attn: bool, train_unembed: bool
    ) -> FakeTraining:
        """Tinker's trainer takes a rank and the three toggles — no alpha, dropout, or module list."""
        self.clients.append(
            FakeTraining(
                base_model=base_model,
                rank=rank,
                seed=seed,
                toggles=(train_mlp, train_attn, train_unembed),
                fail_fb=self.fail_fb,
                weights_info=self.weights_info,
            )
        )
        return self.clients[-1]

    async def create_training_client_from_state_with_optimizer_async(
        self, path: str, user_metadata: dict[str, str] | None = None, weights_access_token: str | None = None
    ) -> FakeTraining:
        """Recreate a client from a saved state: its shape is the checkpoint's own, tagged with the seed path."""
        info = self.weights_info[path]
        self.clients.append(
            FakeTraining(
                base_model=info.base_model,
                rank=info.lora_rank,
                seed=0,
                toggles=(info.train_mlp, info.train_attn, info.train_unembed),
                fail_fb=self.fail_fb,
                weights_info=self.weights_info,
                seeded_from=path,
            )
        )
        return self.clients[-1]

    def create_rest_client(self) -> FakeRestClient:
        return FakeRestClient(self.weights_info)


class FakeTokenizer:
    """A char-level chat tokenizer: the templated prompt is always a prefix of the full text."""

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool, enable_thinking: bool
    ) -> str:
        rendered = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        return f"{rendered}<assistant>" if add_generation_prompt else rendered

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


def fake_sdk() -> ModuleType:
    """A stand-in ``tinker`` module carrying the value types the training code constructs."""
    module = ModuleType("tinker")
    module.__spec__ = importlib.machinery.ModuleSpec("tinker", None)
    module.Datum = FakeDatum
    module.ModelInput = FakeModelInput
    module.TensorData = FakeTensor
    module.AdamParams = FakeAdamParams
    module.SamplingParams = FakeSamplingParams
    return module


def install_sdk(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = fake_sdk()
    monkeypatch.setitem(sys.modules, "tinker", module)
    return module
