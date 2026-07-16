from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from importlib import import_module
from pathlib import Path

import pytest

from athome.research.loop import Measurement, baseline_digest
from athome.research.preflight import PreflightFailure, PreflightReport, preflight
from athome.research.spec import Budget, ExperimentSpec

SCORE_PY = textwrap.dedent(
    """
    import json, pathlib
    namespace = {}
    exec(pathlib.Path("train.py").read_text(), namespace)
    pathlib.Path(".athome-metric.json").write_text(json.dumps({"loss": namespace["LOSS"]}))
    """
).strip()

UNSTABLE_SCORE_PY = textwrap.dedent(
    """
    import json, pathlib
    loss = 1.0 if pathlib.Path.cwd().name == "baseline" else 1.5
    pathlib.Path(".athome-metric.json").write_text(json.dumps({"loss": loss}))
    """
).strip()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def research_repo(root: Path, *, score: str = SCORE_PY, loss: float = 1.0) -> Path:
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "probe@localhost")
    git(root, "config", "user.name", "probe")
    (root / "train.py").write_text(f"LOSS = {loss}\n")
    (root / "score.py").write_text(score + "\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    return root


def make_spec(
    *,
    mutable_paths: tuple[str, ...] = ("train.py",),
    known_good_dir: str | None = None,
) -> ExperimentSpec:
    return ExperimentSpec(
        name="probe",
        metric_command=(sys.executable, "score.py"),
        metric_key="loss",
        direction="min",
        mutable_paths=mutable_paths,
        immutable_paths=("score.py",),
        budget=Budget(max_units=1),
        known_good_dir=known_good_dir,
    )


async def run_preflight(
    repo: Path,
    spec: ExperimentSpec,
    *,
    resume: bool = False,
) -> PreflightReport:
    return await preflight(
        spec,
        repo=repo,
        incumbent=git(repo, "rev-parse", "HEAD"),
        scratch_dir=repo / "scratch",
        baseline_path=repo / "baseline.json",
        resume=resume,
    )


async def test_unmatched_path_glob_names_the_glob(tmp_path: Path) -> None:
    repo = research_repo(tmp_path)

    with pytest.raises(PreflightFailure, match=r"missing/\*\.py"):
        await run_preflight(repo, make_spec(mutable_paths=("missing/*.py",)))


@pytest.mark.parametrize("preexisting", (False, True), ids=("absent", "unchanged"))
async def test_failed_baseline_score_never_persists_a_null_metric(tmp_path: Path, preexisting: bool) -> None:
    repo = research_repo(tmp_path, score="raise SystemExit(1)")
    baseline_path = repo / "baseline.json"
    previous = b'{"commit":"prior","metric":0.25,"spec_digest":"prior-digest"}'
    if preexisting:
        baseline_path.write_bytes(previous)

    with pytest.raises(PreflightFailure, match="baseline produced non-finite metric None"):
        await run_preflight(repo, make_spec())

    if preexisting:
        assert baseline_path.read_bytes() == previous
    else:
        assert not baseline_path.exists()


async def test_corrupt_baseline_cache_names_path(tmp_path: Path) -> None:
    repo = research_repo(tmp_path)
    baseline_path = repo / "baseline.json"
    baseline_path.write_text("{")

    with pytest.raises(PreflightFailure) as caught:
        await run_preflight(repo, make_spec())

    assert str(baseline_path) in str(caught.value)


async def test_nondeterministic_stability_score_fails(tmp_path: Path) -> None:
    repo = research_repo(tmp_path, score=UNSTABLE_SCORE_PY)

    with pytest.raises(PreflightFailure, match=r"stability metric 1\.5 differs from baseline 1\.0"):
        await run_preflight(repo, make_spec())


@pytest.mark.parametrize(
    ("known_loss", "passes"),
    (pytest.param(1.0, False, id="tie"), pytest.param(0.5, True, id="strict-improvement")),
)
async def test_known_good_reachability_uses_extracted_overlay_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    known_loss: float,
    passes: bool,
) -> None:
    repo = research_repo(tmp_path)
    known_good = repo / "known-good"
    known_good.mkdir()
    (known_good / "train.py").write_text(f"LOSS = {known_loss}\n")
    probes = import_module("athome.research.preflight")
    original_score = probes.score_commit
    original_extract = probes.extract_tree
    original_measure = probes.measure
    score_dirs: list[str] = []
    extracted_dirs: list[str] = []
    measured_dirs: list[str] = []

    async def score_spy(*args: object, **kwargs: object) -> Measurement:
        score_dirs.append(Path(kwargs["score_dir"]).name)
        return await original_score(*args, **kwargs)

    async def extract_spy(source: Path, treeish: str, dest: Path) -> None:
        extracted_dirs.append(dest.name)
        await original_extract(source, treeish, dest)

    async def measure_spy(spec: ExperimentSpec, workdir: Path) -> Measurement:
        measured_dirs.append(workdir.name)
        return await original_measure(spec, workdir)

    monkeypatch.setattr(probes, "score_commit", score_spy)
    monkeypatch.setattr(probes, "extract_tree", extract_spy)
    monkeypatch.setattr(probes, "measure", measure_spy)

    if passes:
        report = await run_preflight(repo, make_spec(known_good_dir="known-good"))
        assert report.checks[-1] == "reachability: 0.5"
    else:
        with pytest.raises(PreflightFailure, match="does not strictly beat frozen baseline 1.0"):
            await run_preflight(repo, make_spec(known_good_dir="known-good"))

    assert score_dirs == ["baseline", "stability"]
    assert extracted_dirs == ["reachability"]
    assert measured_dirs == ["reachability"]


async def test_missing_known_good_dir_fails_reachability_after_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = research_repo(tmp_path)
    probes = import_module("athome.research.preflight")
    original_score = probes.score_commit
    original_extract = probes.extract_tree
    score_dirs: list[str] = []
    extracted_dirs: list[str] = []

    async def score_spy(*args: object, **kwargs: object) -> Measurement:
        score_dirs.append(Path(kwargs["score_dir"]).name)
        return await original_score(*args, **kwargs)

    async def extract_spy(source: Path, treeish: str, dest: Path) -> None:
        extracted_dirs.append(dest.name)
        await original_extract(source, treeish, dest)

    monkeypatch.setattr(probes, "score_commit", score_spy)
    monkeypatch.setattr(probes, "extract_tree", extract_spy)

    with pytest.raises(PreflightFailure, match="not-there"):
        await run_preflight(repo, make_spec(known_good_dir="not-there"))

    assert score_dirs == ["baseline", "stability"]
    assert extracted_dirs == ["reachability"]


async def test_resume_digest_mismatch_names_both_digests(tmp_path: Path) -> None:
    repo = research_repo(tmp_path)
    spec = make_spec()
    baseline_path = repo / "baseline.json"
    baseline_path.write_text(
        json.dumps({"commit": git(repo, "rev-parse", "HEAD"), "metric": 1.0, "spec_digest": "prior-digest"})
    )

    with pytest.raises(PreflightFailure) as caught:
        await run_preflight(repo, spec, resume=True)

    assert "prior-digest" in str(caught.value)
    assert baseline_digest(spec) in str(caught.value)


async def test_resume_skips_stability_and_reachability_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = research_repo(tmp_path)
    spec = make_spec(known_good_dir="not-present")
    (repo / "baseline.json").write_text(
        json.dumps(
            {
                "commit": git(repo, "rev-parse", "HEAD"),
                "metric": 1.0,
                "spec_digest": baseline_digest(spec),
            }
        )
    )
    probes = import_module("athome.research.preflight")

    async def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("resumed preflight must not score or materialize optional probes")

    monkeypatch.setattr(probes, "score_commit", forbidden)
    monkeypatch.setattr(probes, "extract_tree", forbidden)
    monkeypatch.setattr(probes, "measure", forbidden)

    report = await run_preflight(repo, spec, resume=True)

    assert report.checks == (
        "globs bind: passed",
        "baseline: 1.0",
        "stability: skipped (resume)",
        "reachability: skipped (resume)",
    )
