from __future__ import annotations

import tomllib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from athome.research.errors import ResearchError

if TYPE_CHECKING:
    from pathlib import Path


class ImmutableViolation(ResearchError):
    """A candidate edited a path the scoring boundary declares immutable."""


class BudgetExhausted(ResearchError):
    """A run consumed its work-unit, wall-clock, or dollar budget."""


@dataclass(frozen=True, slots=True)
class Budget:
    """The stop conditions for a greedy keep/discard run.

    Work-unit iterations are the first-class, cross-machine-comparable budget;
    wall-clock and dollars are backstops, and ``hard_kill_s`` bounds one iteration.

    Attributes:
        max_units: Work-unit iterations to run before stopping.
        max_wall_s: Wall-clock backstop, in seconds.
        hard_kill_s: Per-iteration hard-kill timeout, in seconds.
        max_usd: Spend backstop, in dollars.
    """

    max_units: int
    max_wall_s: float | None = None
    hard_kill_s: float | None = None
    max_usd: float | None = None


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """A TOML-loaded research experiment: the typed ``program.md`` + ``prepare.py`` split.

    The metric is read from ``metric_file`` (a structured JSON channel the
    ``metric_command`` writes), never grepped from stdout. The ``immutable_paths``
    globs are the scoring boundary the candidate must not touch.

    Example:
        >>> spec = ExperimentSpec.load(Path("experiment.toml"))
    """

    name: str
    metric_command: tuple[str, ...]
    metric_key: str
    direction: Literal["min", "max"]
    mutable_paths: tuple[str, ...]
    immutable_paths: tuple[str, ...]
    budget: Budget
    metric_file: str = ".athome-metric.json"

    @classmethod
    def load(cls, path: Path) -> ExperimentSpec:
        """Loads the experiment from a TOML file; the ``[budget]`` table becomes a :class:`Budget`."""
        with path.open("rb") as file:
            data = tomllib.load(file)
        return cls(
            budget=Budget(**data.pop("budget")),
            **{key: tuple(value) if isinstance(value, list) else value for key, value in data.items()},
        )


@dataclass(frozen=True, slots=True)
class Comparability:
    """Two runs are comparable iff both fields match (config + dataset fingerprint).

    Attributes:
        config_hash: Fingerprint of the run configuration.
        dataset_digest: Order-invariant fingerprint of the dataset.
    """

    config_hash: str
    dataset_digest: str
