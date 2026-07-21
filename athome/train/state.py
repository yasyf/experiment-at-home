from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from athome.errors import AthomeError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from athome.wire import Wire

type StateFidelity = Literal["weights+optimizer", "weights"]


class BackendMismatch(AthomeError):
    """Raised when a backend is handed a :data:`StateHandle` variant its stack cannot seed from."""


@dataclass(frozen=True, slots=True)
class TinkerState:
    """A Tinker ``save_state`` checkpoint — weights *and* optimizer, a distinct artifact from a sampler save.

    Attributes:
        state_path: The opaque ``tinker://`` address ``save_state_async`` wrote the training state to,
            never a ``save_weights_for_sampler`` path.
    """

    state_path: str
    fidelity: ClassVar[StateFidelity] = "weights+optimizer"


@dataclass(frozen=True, slots=True)
class LocalState:
    """A durable mlx-lm adapter checkpoint directory: weights only, no optimizer momentum.

    Attributes:
        adapter_dir: The mlx-lm adapter directory a later run reloads with ``--resume-adapter-file``.
    """

    adapter_dir: Path
    fidelity: ClassVar[StateFidelity] = "weights"


@dataclass(frozen=True, slots=True)
class ModalState:
    """An HF adapter repo pinned to a commit: weights only, no optimizer momentum.

    Attributes:
        repo: The ``owner/name`` HF adapter repo the trained adapter was pushed to.
        revision: The commit the adapter landed on, so a continuation reloads the exact weights.
    """

    repo: str
    revision: str
    fidelity: ClassVar[StateFidelity] = "weights"


type StateHandle = TinkerState | LocalState | ModalState


@dataclass(frozen=True, slots=True)
class Resume:
    """The unified restore input both same-run recovery and cross-run continuation lower into.

    Attributes:
        handle: The policy state the training client is seeded from.
        from_step: Same-run crash recovery slices the deterministic plan to ``plan[from_step:]`` and
            labels the remaining steps from ``from_step + 1``; cross-run continuation (DITTO) leaves
            it 0 for a fresh full schedule seeded from the prior state.
        cost_usd: The spend already billed under this run's key, seeded into the SpendGuard so
            ``max_usd`` binds per run rather than per attempt.
        reference: The DPO reference anchor the frozen reference client is rebuilt from — the
            run-start anchor for same-run recovery, the caller's prior state for cross-run
            continuation, or None when the reference is the base model.
    """

    handle: StateHandle
    from_step: int
    cost_usd: float = 0.0
    reference: StateHandle | None = None


def handle_to_json(handle: StateHandle) -> dict[str, Wire]:
    """Serialize a :data:`StateHandle` to a tagged JSON row — the one serialization codepath."""
    match handle:
        case TinkerState(state_path=state_path):
            return {"backend": "tinker", "state_path": state_path}
        case LocalState(adapter_dir=adapter_dir):
            return {"backend": "local", "adapter_dir": str(adapter_dir)}
        case ModalState(repo=repo, revision=revision):
            return {"backend": "modal", "repo": repo, "revision": revision}


def handle_from_json(row: Mapping[str, object]) -> StateHandle:
    """Rebuild a :data:`StateHandle` from a tagged JSON row, crashing on an unknown tag."""
    match row["backend"]:
        case "tinker":
            return TinkerState(state_path=str(row["state_path"]))
        case "local":
            return LocalState(adapter_dir=Path(str(row["adapter_dir"])))
        case "modal":
            return ModalState(repo=str(row["repo"]), revision=str(row["revision"]))
        case unknown:
            raise ValueError(f"unknown state handle backend: {unknown!r}")
