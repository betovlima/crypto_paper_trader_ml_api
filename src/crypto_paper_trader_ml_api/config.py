from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class BootstrapSettings(BaseSettings):
    """Only the database connection is allowed outside the database.

    Runtime, market, model, feature, risk and backtest parameters are stored in
    ``ml_runtime_parameters`` and are loaded from there for every new job.
    """

    model_config = SettingsConfigDict(extra="ignore")

    database_url: str = "sqlite:///./data/ml_backtests.db"

    def ensure_runtime_directories(self) -> None:
        if self.database_url.startswith("sqlite:///./"):
            relative = self.database_url.removeprefix("sqlite:///./")
            Path(relative).parent.mkdir(parents=True, exist_ok=True)
        elif self.database_url.startswith("sqlite:////"):
            absolute = self.database_url.removeprefix("sqlite:////")
            Path(f"/{absolute}").parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_bootstrap_settings() -> BootstrapSettings:
    settings = BootstrapSettings()
    settings.ensure_runtime_directories()
    return settings
