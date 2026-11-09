from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MLBacktestCreate(BaseModel):
    experiment_type: Literal["TARGET_STOP", "BTC_DIRECTION_RETURN"] | None = None
    symbol: str | None = None
    history_source: Literal["MEXC", "BINANCE_SPOT_ARCHIVE"] | None = None
    period_years: int | None = None
    decision_timeframe: str | None = None
    trend_timeframe: str | None = None
    initial_capital: float | None = None
    fee_rate: float | None = None
    slippage_rate: float | None = None
    entry_threshold: float | None = None
    exit_threshold: float | None = None
    reinvestment: bool | None = None

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.upper().replace("/", "").replace("-", "").strip()
        if not normalized.isalnum():
            raise ValueError("symbol must contain only letters and numbers")
        return normalized


class ResolvedMLBacktestCreate(BaseModel):
    experiment_type: Literal["TARGET_STOP", "BTC_DIRECTION_RETURN"] = "TARGET_STOP"
    symbol: str
    history_source: Literal["MEXC", "BINANCE_SPOT_ARCHIVE"] = "MEXC"
    period_years: int
    decision_timeframe: str
    trend_timeframe: str
    initial_capital: float
    fee_rate: float
    slippage_rate: float
    entry_threshold: float
    exit_threshold: float
    reinvestment: bool


class MLBacktestJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    history_source: Literal["MEXC", "BINANCE_SPOT_ARCHIVE"] = "MEXC"
    experiment_type: Literal["TARGET_STOP", "BTC_DIRECTION_RETURN"] = "TARGET_STOP"

    id: str
    status: str
    stage: str
    progress: int
    symbol: str
    decision_timeframe: str
    trend_timeframe: str
    period_years: int
    initial_capital: float
    replay_parent_job_id: str | None = None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class MLBacktestJobDetail(MLBacktestJobResponse):
    audit_available: bool = False
    request: dict
    settings_snapshot: dict | None
    result: dict | None


class BTCCapitalizationRequest(BaseModel):
    # 0 means a single fused 2h+6h+24h BTC/CASH financial simulation.
    horizon_hours: Literal[0, 2, 6, 24] = 0
    initial_capital: float = Field(default=10000.0, gt=0, le=1000000000)
    fee_rate: float = Field(default=0.001, ge=0, le=0.05)
    slippage_rate: float = Field(default=0.0005, ge=0, le=0.05)
    entry_edge: float = Field(default=0, ge=0, le=0.2)
    min_probability: float = Field(default=0.5, ge=0, le=1)
    exit_return: float = Field(default=0, ge=-0.2, le=0.2)

 
class MLResearchStrategyCreate(BaseModel):
    name: str = Field(min_length=3, max_length=120)
    source_job_id: str = Field(min_length=8, max_length=64)
    # A single research strategy now evaluates all three horizons. Explicit
    # single-horizon requests remain valid for older saved experiments.
    horizon_hours: Literal[0, 2, 6, 24] = 0


class MLStrategyTuneRequest(BaseModel):
    mode: Literal["QUICK", "BAYESIAN", "EVENTS", "DIRECTIONAL_CHANGE", "TECHNICAL", "TREND_REGIMES", "INDICATOR_LAB"]
    trials: int = Field(default=8, ge=4, le=40)
    # One click starts optimization and automatically schedules the final OOS run.
    auto_execute: bool = False



class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    parameter_source: str


class RuntimeParameterItem(BaseModel):
    key: str
    category: str
    label: str
    description: str
    value_type: str
    value: Any
    constraints: dict[str, Any]
    public: bool
    requires_restart: bool
    updated_at: datetime


class RuntimeParameterUpdate(BaseModel):
    key: str
    value: Any


class RuntimeParametersUpdateRequest(BaseModel):
    parameters: list[RuntimeParameterUpdate] = Field(min_length=1, max_length=200)


class RuntimeParametersResponse(BaseModel):
    source: Literal["DATABASE"] = "DATABASE"
    parameters: list[RuntimeParameterItem]


class PublicMLConfigurationResponse(BaseModel):
    source: Literal["DATABASE"] = "DATABASE"
    defaults: dict[str, Any]
    limits: dict[str, Any]
    options: dict[str, Any]
    ui: dict[str, Any]
