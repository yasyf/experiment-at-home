from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from click.testing import CliRunner

from athome.research import loop
from athome.research.cli import cli as research_cli

if TYPE_CHECKING:
    import pytest

    from athome.research.driver import Driver
    from athome.research.spec import ExperimentSpec


SPEC_TOML = textwrap.dedent(
    """
    name = "toy"
    metric_command = ["python", "score.py"]
    metric_key = "loss"
    direction = "min"
    mutable_paths = ["train.py"]
    immutable_paths = ["score.py"]

    [budget]
    max_units = 1
    """
).strip()


def test_run_threads_the_mirror_cc_notes_flag_to_the_loop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spec_path = tmp_path / "experiment.toml"
    spec_path.write_text(SPEC_TOML + "\n")
    captured: dict[str, object] = {}

    async def fake_run(
        spec: ExperimentSpec,
        *,
        driver: Driver,
        repo: Path,
        mirror_cc_notes: bool = False,
    ) -> loop.LoopResult:
        captured.update(spec=spec, driver=driver, repo=repo, mirror_cc_notes=mirror_cc_notes)
        return loop.LoopResult(kept=0, best=None)

    monkeypatch.setattr(loop, "run", fake_run)

    result = CliRunner().invoke(
        research_cli,
        ["run", str(spec_path), "--repo", str(tmp_path), "--mirror-cc-notes", "--json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"best": None, "kept": 0}
    assert captured["repo"] == tmp_path.resolve()
    assert captured["mirror_cc_notes"] is True
