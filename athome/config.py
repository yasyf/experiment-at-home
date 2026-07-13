from __future__ import annotations

import os
from functools import lru_cache, reduce
from pathlib import Path
from typing import Any, ClassVar

from pydantic import field_validator
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

CONFIG_FILE = Path.home() / ".athome" / "config.toml"


def env_prefix_for(settings_cls: type[SectionSettings]) -> str:
    return "ATHOME_" + "".join(f"{part.upper().replace('-', '_').replace('.', '_')}_" for part in settings_cls.section)


def base_environ() -> dict[str, str]:
    return dict(os.environ)


class SectionTomlSource(TomlConfigSettingsSource):
    def __init__(self, settings_cls: type[SectionSettings]) -> None:
        self.section = settings_cls.section
        super().__init__(settings_cls, toml_file=CONFIG_FILE)

    def _read_files(self, files: Path | None, deep_merge: bool = False) -> dict[str, Any]:
        return reduce(
            lambda data, part: data.get(part, {}),
            self.section,
            super()._read_files(files, deep_merge=deep_merge),
        )


class SectionSettings(BaseSettings):
    """Base for every athome settings model; binds one [section] of ~/.athome/config.toml.

    Precedence: init kwargs > env (ATHOME_<SECTION>_<FIELD>) > TOML section > defaults.
    """

    section: ClassVar[tuple[str, ...]] = ()
    model_config = SettingsConfigDict(frozen=True, extra="ignore")

    @field_validator("*", mode="after")
    @classmethod
    def expand_user_paths(cls, value: object) -> object:
        return value.expanduser() if isinstance(value, Path) else value

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            EnvSettingsSource(settings_cls, env_prefix=env_prefix_for(settings_cls)),
            SectionTomlSource(settings_cls),
        )


class AthomeSettings(SectionSettings):
    """Root settings: the shared filesystem roots and the machine env prefix."""

    cache_root: Path = Path("~/.athome/cache")
    logs_root: Path = Path("~/.athome/logs")
    batches_root: Path = Path("~/.athome/batches")
    env_prefix_cmd: str | None = None


@lru_cache
def load[S: SectionSettings](model: type[S]) -> S:
    """Construct a settings model once per process (cleared in tests via ``load.cache_clear()``)."""
    return model()
