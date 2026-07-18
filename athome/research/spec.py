from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from math import isfinite
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal

from athome.research.errors import AccountingIntegrityError, ResearchError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class ImmutableViolation(ResearchError):
    """A candidate edited a path the scoring boundary declares immutable."""


class UnknownSpecField(ResearchError):
    """A TOML carried a field its target dataclass does not declare; refused, never splatted through."""


class BudgetExhausted(ResearchError):
    """A run consumed its work-unit, wall-clock, or dollar budget."""


class InvalidBudget(ResearchError):
    """A budget or ceiling carried a non-finite, non-positive, or non-integer cap; refused, never clamped."""


class UnconfinedPath(ResearchError):
    """A spec path escapes the work directory (absolute, ``..`` traversal, or empty)."""


class PoisonedJournal(AccountingIntegrityError):
    """A resumed journal was unreadable, malformed, or carried an invalid metric or spend."""


class ConcurrentRun(ResearchError):
    """Another live run already holds the per-experiment single-writer lock."""


class ProposalTimeout(ResearchError):
    """A driver's proposal exceeded its bound; carries the spend recovered from the run log."""

    def __init__(self, message: str, *, cost: float) -> None:
        super().__init__(message)
        self.cost = cost


def unbounded_glob(pattern: str) -> bool:
    return PurePosixPath(pattern).parts[0] in {"*", "**"}


def finite_number(value: object) -> bool:
    match value:
        case bool():
            return False
        case int() | float():
            try:
                coerced = float(value)
            except Exception:
                # This total predicate treats every conversion failure as non-finite.
                return False
            return isfinite(coerced)
        case _:
            return False


def positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def escapes_workdir(path: str) -> bool:
    return not (pure := PurePosixPath(path)).parts or pure.is_absolute() or ".." in pure.parts


def reject_unknown_fields(cls: type, data: Mapping[str, object], *, source: str) -> None:
    """Refuse any mapping key the target dataclass does not declare as a field.

    The splat-construction loaders (:meth:`ExperimentSpec.load` and the policy
    loader) call this before ``cls(**data)``, so a hostile TOML cannot smuggle a
    field past the declared schema and fails with a named refusal instead of an
    incidental ``TypeError``.

    Raises:
        UnknownSpecField: ``data`` carries keys ``cls`` does not declare.
    """
    if unknown := sorted(set(data) - {field.name for field in fields(cls)}):
        raise UnknownSpecField(f"{source} carries fields {cls.__name__} does not declare: {unknown}")


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

    def __post_init__(self) -> None:
        if not positive_int(self.max_units):
            raise InvalidBudget(f"max_units must be a positive int, not {self.max_units!r}")
        for label, value in (
            ("max_wall_s", self.max_wall_s),
            ("hard_kill_s", self.hard_kill_s),
            ("max_usd", self.max_usd),
        ):
            if value is not None and not (finite_number(value) and value > 0):
                raise InvalidBudget(f"{label} must be a finite positive number, not {value!r}")


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """A TOML-loaded research experiment: the typed ``program.md`` + ``prepare.py`` split.

    The metric is read from ``metric_file`` (a structured JSON channel the
    ``metric_command`` writes), never grepped from stdout. The ``immutable_paths``
    globs are the scoring boundary the candidate must not touch. An optional
    ``hypothesis`` (free text) renders into the contract's ``## Hypothesis``
    section and nowhere else — never a shell, path, or key.

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
    hypothesis: str | None = None
    known_good_dir: str | None = None

    def __post_init__(self) -> None:
        if bad := [pattern for pattern in self.mutable_paths if unbounded_glob(pattern)]:
            raise ImmutableViolation(f"mutable_paths must be a tight allowlist; unbounded globs rejected: {bad}")
        if escapes_workdir(self.metric_file):
            raise UnconfinedPath(f"metric_file {self.metric_file!r} must be a relative path inside the work directory")
        if self.known_good_dir is not None and escapes_workdir(self.known_good_dir):
            raise ImmutableViolation("known_good_dir must be a repo-relative directory")

    @classmethod
    def loads(cls, text: str, *, source: str) -> ExperimentSpec:
        """Loads the experiment from TOML text already in hand; the ``[budget]`` table becomes a :class:`Budget`.

        The seam for verified loads: a caller that hash-checks a file parses the
        exact bytes it hashed instead of re-reading the path.

        Args:
            text: The TOML document.
            source: Where the text came from, named in refusals.

        Raises:
            UnknownSpecField: the TOML carries a field the spec or its budget does not declare.
        """
        data = tomllib.loads(text)
        budget = data.pop("budget")
        reject_unknown_fields(Budget, budget, source=source)
        reject_unknown_fields(cls, data, source=source)
        return cls(
            budget=Budget(**budget),
            **{key: tuple(value) if isinstance(value, list) else value for key, value in data.items()},
        )

    @classmethod
    def load(cls, path: Path) -> ExperimentSpec:
        """Loads the experiment from a TOML file; the ``[budget]`` table becomes a :class:`Budget`.

        Raises:
            UnknownSpecField: the TOML carries a field the spec or its budget does not declare.
        """
        return cls.loads(path.read_text(), source=str(path))


@dataclass(frozen=True, slots=True)
class Comparability:
    """Two runs are comparable iff both fields match (config + dataset fingerprint).

    Attributes:
        config_hash: Fingerprint of the run configuration.
        dataset_digest: Order-invariant fingerprint of the dataset.
    """

    config_hash: str
    dataset_digest: str
