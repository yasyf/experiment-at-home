from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import anyio

from athome.cache import atomic_write_text
from athome.research.errors import ResearchError
from athome.research.gate import matches, monotone_gate
from athome.research.loop import (
    InvalidBaseline,
    baseline_digest,
    extract_tree,
    finite_number,
    measure,
    read_baseline,
    run_git,
    score_commit,
)
from athome.research.spec import ExperimentSpec

STABILITY_RTOL = 0.1


class PreflightFailure(ResearchError):
    """A mandatory research validation probe failed before the unit loop began."""


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """The successful pre-flight checks and frozen experiment anchor.

    The baseline is the fixed score of the original untouched tree against which
    uplift is reported. It does not move when the independently tracked incumbent
    advances to the best journaled commit after each successful unit.

    Attributes:
        checks: Human-readable outcomes for every mandatory probe.
        baseline: The frozen, finite metric anchor for the experiment.
    """

    checks: tuple[str, ...]
    baseline: float


def checked_metric(value: object, *, probe: str) -> float:
    if not finite_number(value):
        raise PreflightFailure(f"{probe} produced non-finite metric {value!r}")
    return float(value)


async def preflight(
    spec: ExperimentSpec,
    *,
    repo: Path,
    incumbent: str,
    scratch_dir: Path,
    baseline_path: Path,
    resume: bool,
) -> PreflightReport:
    """Runs every mandatory validation probe before an experiment executes a unit.

    A fresh run may score and persist a new frozen baseline when its commit or
    scoring-channel digest changes. A resumed run never scores a replacement
    baseline: it loads the original anchor and requires its digest to match the
    current scoring channel. That anchor remains fixed while ``incumbent`` is the
    separate best-so-far commit, which can advance after every kept unit.

    Args:
        spec: The experiment and its scoring boundary.
        repo: The git repository containing the experiment.
        incumbent: The current best commit to validate and score.
        scratch_dir: An empty directory reserved for isolated probe checkouts.
        baseline_path: The experiment's persisted frozen-baseline record.
        resume: Whether this invocation is continuing a journaled experiment.

    Returns:
        The ordered check outcomes and the experiment's frozen baseline.

    Raises:
        PreflightFailure: A path glob, score, stability, reachability, or resumed
            baseline validation failed.
    """
    paths = tuple(
        path for path in (await run_git(repo, "ls-tree", "-r", "--name-only", "-z", incumbent)).split("\0") if path
    )
    patterns = spec.mutable_paths + spec.immutable_paths
    if unbound := [pattern for pattern in patterns if not any(matches(path, (pattern,)) for path in paths)]:
        raise PreflightFailure(f"path globs matched no files at {incumbent}: {unbound}")
    checks = ["globs bind: passed"]

    digest = baseline_digest(spec)
    try:
        cached = await read_baseline(anyio.Path(baseline_path))
    except InvalidBaseline as error:
        raise PreflightFailure(
            f"frozen baseline at {baseline_path} is corrupt; delete it to re-anchor a fresh run"
        ) from error
    if resume:
        if cached is None:
            raise PreflightFailure(f"resumed experiment has no frozen baseline at {baseline_path}")
        if cached.spec_digest != digest:
            raise PreflightFailure(
                f"resumed baseline digest {cached.spec_digest} does not match current scoring channel {digest} — "
                "the metric command, key, or file changed since this experiment's anchor was frozen; start a new "
                "experiment"
            )
        baseline = checked_metric(cached.metric, probe="resumed baseline")
    elif cached is not None and cached.commit == incumbent and cached.spec_digest == digest:
        baseline = checked_metric(cached.metric, probe="cached baseline")
    else:
        measured = await score_commit(
            spec,
            repo=repo,
            score_dir=scratch_dir / "baseline",
            commit=incumbent,
        )
        baseline = checked_metric(measured.metric, probe="baseline")
        await atomic_write_text(
            anyio.Path(baseline_path),
            json.dumps({"commit": incumbent, "metric": baseline, "spec_digest": digest}),
        )
    checks.append(f"baseline: {baseline}")

    if resume:
        checks.extend(("stability: skipped (resume)", "reachability: skipped (resume)"))
        return PreflightReport(checks=tuple(checks), baseline=baseline)

    repeated = await score_commit(
        spec,
        repo=repo,
        score_dir=scratch_dir / "stability",
        commit=incumbent,
    )
    stable = checked_metric(repeated.metric, probe="stability")
    within_tolerance = stable == baseline if baseline == 0 else abs(stable - baseline) / abs(baseline) <= STABILITY_RTOL
    if not within_tolerance:
        raise PreflightFailure(
            f"stability metric {stable} differs from baseline {baseline} by more than relative tolerance "
            f"{STABILITY_RTOL}"
        )
    checks.append(f"stability: {stable}")

    if spec.known_good_dir is None:
        checks.append("reachability: skipped (no known_good_dir)")
        return PreflightReport(checks=tuple(checks), baseline=baseline)

    reachability_dir = scratch_dir / "reachability"
    await extract_tree(repo, incumbent, reachability_dir)
    try:
        await anyio.to_thread.run_sync(
            partial(shutil.copytree, repo / spec.known_good_dir, reachability_dir, dirs_exist_ok=True)
        )
    except OSError as error:
        raise PreflightFailure(
            f"known_good_dir {spec.known_good_dir!r} could not be copied for reachability: {error}"
        ) from error
    reachable = await measure(spec, reachability_dir)
    reachable_metric = checked_metric(reachable.metric, probe="reachability")
    if not monotone_gate(reachable_metric, baseline, direction=spec.direction):
        raise PreflightFailure(
            f"known-good metric {reachable_metric} does not strictly beat frozen baseline {baseline} "
            f"for direction {spec.direction}"
        )
    checks.append(f"reachability: {reachable_metric}")
    return PreflightReport(checks=tuple(checks), baseline=baseline)
