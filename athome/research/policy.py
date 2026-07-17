"""Operator-authored proposal policy: the outer loop's unwidenable bounds as pure data.

Every executable field a proposed experiment can carry is copied verbatim from one
of these templates; every number it can carry is bounded by these ceilings. The
proposer cannot widen any of it — a policy is loaded once, frozen, and violations
are refused, never clamped.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from athome.research.errors import ResearchError
from athome.research.spec import Budget, reject_unknown_fields, unbounded_glob

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class PolicyViolation(ResearchError):
    """The policy itself is self-inconsistent; refused at load, before any proposal is read."""


def build[T](cls: type[T], data: Mapping[str, object], *, source: str) -> T:
    reject_unknown_fields(cls, data, source=source)
    return cls(**{key: tuple(value) if isinstance(value, list) else value for key, value in data.items()})


@dataclass(frozen=True, slots=True)
class CampaignBudget:
    """Cumulative stop conditions for a whole campaign; every cap is required.

    Attributes:
        max_experiments: Experiments to run before the campaign stops.
        max_total_usd: Cumulative spend cap the reservation math enforces.
        max_wall_s: Cumulative wall-clock cap across the campaign.
        max_consecutive_failures: Consecutive rejected/failed rounds that abort the campaign.
    """

    max_experiments: int
    max_total_usd: float
    max_wall_s: float
    max_consecutive_failures: int


@dataclass(frozen=True, slots=True)
class ExperimentTemplate:
    """One operator-authored experiment shape the proposer may instantiate.

    The metric command/key, direction, and immutable paths are copied verbatim into
    every admitted spec; ``mutable_allowlist`` is the exact set of globs a proposal
    may pick its ``mutable_paths`` from (membership by string equality, never glob
    subsumption).

    Attributes:
        name: The template's identifier, referenced by proposals.
        metric_command: The scoring command, copied verbatim.
        metric_key: The metric field in the structured JSON channel, copied verbatim.
        direction: Whether the metric is minimized or maximized, copied verbatim.
        immutable_paths: The scoring boundary, copied verbatim.
        mutable_allowlist: The exact globs a proposal may list as mutable.
        metric_file: The structured JSON channel filename, copied verbatim.
    """

    name: str
    metric_command: tuple[str, ...]
    metric_key: str
    direction: Literal["min", "max"]
    immutable_paths: tuple[str, ...]
    mutable_allowlist: tuple[str, ...]
    metric_file: str = ".athome-metric.json"

    def __post_init__(self) -> None:
        if self.direction not in ("min", "max"):
            raise PolicyViolation(f"template {self.name!r} direction must be 'min' or 'max', not {self.direction!r}")
        if bad := [pattern for pattern in self.mutable_allowlist if unbounded_glob(pattern)]:
            raise PolicyViolation(f"template {self.name!r} allowlists unbounded globs: {bad}")
        if not self.mutable_allowlist:
            raise PolicyViolation(f"template {self.name!r} has an empty mutable_allowlist; it can admit no proposal")


@dataclass(frozen=True, slots=True)
class ProposalPolicy:
    """The frozen operator policy a campaign's every proposal is validated against.

    In ``auto`` mode proposals run unattended, so the per-experiment ``max_usd`` and
    ``max_wall_s`` ceilings are mandatory — the campaign's reservation math uses
    declared caps, never estimates. In ``gated`` mode an operator approves each
    proposal before it runs.

    Example:
        >>> policy = ProposalPolicy.load(Path("policy.toml"))
    """

    mode: Literal["auto", "gated"]
    campaign: CampaignBudget
    ceilings: Budget
    templates: tuple[ExperimentTemplate, ...]

    def __post_init__(self) -> None:
        if self.mode not in ("auto", "gated"):
            raise PolicyViolation(f"mode must be 'auto' or 'gated', not {self.mode!r}")
        if not self.templates:
            raise PolicyViolation("a policy with no templates can admit no proposal")
        names = [template.name for template in self.templates]
        if dupes := sorted({name for name in names if names.count(name) > 1}):
            raise PolicyViolation(f"duplicate template names: {dupes}")
        if self.mode == "auto" and (self.ceilings.max_usd is None or self.ceilings.max_wall_s is None):
            raise PolicyViolation("auto mode requires max_usd and max_wall_s ceilings: reservations use declared caps")

    @classmethod
    def load(cls, path: Path) -> ProposalPolicy:
        """Loads the policy from a TOML file with unknown fields refused at every level.

        The ``[campaign]`` table becomes a :class:`CampaignBudget`, ``[ceilings]``
        a :class:`~athome.research.spec.Budget`, and each ``[[templates]]`` entry an
        :class:`ExperimentTemplate`.

        Raises:
            UnknownSpecField: any table carries a field its dataclass does not declare.
            PolicyViolation: the loaded policy is self-inconsistent.
        """
        with path.open("rb") as file:
            data = tomllib.load(file)
        campaign = data.pop("campaign")
        ceilings = data.pop("ceilings")
        templates = data.pop("templates")
        reject_unknown_fields(cls, data, source=str(path))
        return cls(
            campaign=build(CampaignBudget, campaign, source=str(path)),
            ceilings=build(Budget, ceilings, source=str(path)),
            templates=tuple(build(ExperimentTemplate, template, source=str(path)) for template in templates),
            **data,
        )
