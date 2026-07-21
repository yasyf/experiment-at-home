from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from athome.errors import AthomeError

if TYPE_CHECKING:
    from athome.progress import RunSink
    from athome.train.runstate import RunStateStore
    from athome.train.spec import BackendName, Checkpoint, Method, TrainSettings, TrainSpec
    from athome.train.state import Resume, StateFidelity

TINKER_ENV: Path = Path("~/.athome/tinker.env").expanduser()


class NoBackendAvailable(AthomeError):
    """Raised when no backend can run the request: none is available, or the override cannot."""


@runtime_checkable
class TrainBackend(Protocol):
    """What every training backend exposes: a cheap probe, its methods, and a train call.

    ``available`` and ``supports`` are static so :func:`select` can probe a backend
    without constructing its settings — a required secret would raise before the
    backend was even chosen.
    """

    name: ClassVar[BackendName]
    state_fidelity: ClassVar[StateFidelity]

    @staticmethod
    def available() -> bool:
        """Whether this backend can run here: its credentials or hardware are present."""
        ...

    @staticmethod
    def supports(method: Method) -> bool:
        """Whether this backend trains ``method``."""
        ...

    @classmethod
    def from_settings(cls) -> TrainBackend:
        """Construct the backend from its configuration section."""
        ...

    async def train(
        self,
        spec: TrainSpec,
        *,
        sink: RunSink,
        work_dir: Path,
        resume: Resume | None = None,
        store: RunStateStore | None = None,
    ) -> Checkpoint:
        """Train ``spec``, journaling progress to ``sink``, and converge on a servable artifact.

        Every file the run writes — data splits, adapters, the fused model — goes under
        ``work_dir``, which :func:`athome.train.run` mints fresh per run. A backend never
        derives its own output directory: two runs of one family would then overwrite each
        other's weights, including weights the registry has already registered.

        ``resume`` seeds the run from a prior training state — a same-run crash recovery or a
        cross-run continuation — and ``store`` is the run-state ledger progress persists to; both
        are None for a fresh, unledgered run. The produced ``Checkpoint`` always carries a
        ``state`` handle at this backend's declared :attr:`state_fidelity`.
        """
        ...


def backends() -> tuple[type[TrainBackend], ...]:
    from athome.train.local import LocalBackend
    from athome.train.modal import ModalTrainBackend
    from athome.train.tinker import TinkerBackend

    return (TinkerBackend, LocalBackend, ModalTrainBackend)


def select(spec: TrainSpec, settings: TrainSettings) -> TrainBackend:
    """Pick the backend that runs ``spec``, preferring tinker, then local, then modal.

    An override — ``spec.backend`` first, else ``settings.backend`` — pins the choice;
    otherwise the first backend that is :meth:`~TrainBackend.available` and
    :meth:`~TrainBackend.supports` the spec's method wins. Only the winner is
    constructed.

    Args:
        spec: The fine-tuning request; its ``method`` and ``backend`` drive the choice.
        settings: The ``[train]`` settings, whose ``backend`` is the fallback override.

    Returns:
        The selected backend, constructed from its settings section.

    Raises:
        NoBackendAvailable: The override cannot run the spec, or nothing qualifies.
    """
    if (override := spec.backend or settings.backend) is not None:
        chosen = next(backend for backend in backends() if backend.name == override)
        if not (chosen.available() and chosen.supports(spec.method)):
            raise NoBackendAvailable(
                f"backend {override!r} cannot run method {spec.method!r} "
                f"(available={chosen.available()}, supports={chosen.supports(spec.method)})"
            )
        return chosen.from_settings()
    if usable := [b for b in backends() if b.available() and b.supports(spec.method)]:
        return usable[0].from_settings()
    raise NoBackendAvailable(f"no available backend trains method {spec.method!r}")
