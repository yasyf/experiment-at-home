from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from athome.config import AthomeSettings, SectionSettings, load


class ProbeSettings(SectionSettings):
    section = ("probe",)
    name: str = "default"
    count: int = 0
    path: Path = Path("~/probe")


def test_defaults_when_config_absent() -> None:
    load.cache_clear()
    settings = load(ProbeSettings)
    assert settings.name == "default"
    assert settings.count == 0


def test_env_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATHOME_PROBE_NAME", "fromenv")
    monkeypatch.setenv("ATHOME_PROBE_COUNT", "5")
    load.cache_clear()
    settings = load(ProbeSettings)
    assert settings.name == "fromenv"
    assert settings.count == 5


def test_toml_section_pluck(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text('[probe]\nname = "fromtoml"\ncount = 7\n\n[other]\nname = "ignored"\n')
    load.cache_clear()
    settings = load(ProbeSettings)
    assert settings.name == "fromtoml"
    assert settings.count == 7


def test_env_beats_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.toml").write_text('[probe]\nname = "fromtoml"\n')
    monkeypatch.setenv("ATHOME_PROBE_NAME", "fromenv")
    load.cache_clear()
    assert load(ProbeSettings).name == "fromenv"


def test_tilde_expansion() -> None:
    load.cache_clear()
    assert load(ProbeSettings).path == Path("~/probe").expanduser()


def test_malformed_toml_raises(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text("name = = not valid ][\n")
    load.cache_clear()
    with pytest.raises(tomllib.TOMLDecodeError):
        load(ProbeSettings)


def test_root_roots_resolve_from_env(tmp_path: Path) -> None:
    load.cache_clear()
    settings = load(AthomeSettings)
    assert settings.cache_root == tmp_path / "cache"
    assert settings.logs_root == tmp_path / "logs"
    assert settings.batches_root == tmp_path / "batches"
    assert settings.env_prefix_cmd is None


def test_load_is_cached_per_model() -> None:
    load.cache_clear()
    assert load(AthomeSettings) is load(AthomeSettings)


def test_settings_are_frozen() -> None:
    load.cache_clear()
    settings = load(AthomeSettings)
    with pytest.raises(ValidationError):
        settings.cache_root = Path("/elsewhere")
