from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_paper_trader_ml_api.engine import calculate_results, simulate_portfolio
from crypto_paper_trader_ml_api.schemas import ResolvedMLBacktestCreate

from .helpers import default_parameters


def prediction_frame(size: int = 400) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=size, freq="30min", tz="UTC")
    price = 100 + np.linspace(0, 20, size) + np.sin(np.arange(size) / 8)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price + 0.2,
            "atr_14": 1.2,
            "ema_21": price - 0.5,
            "ema_50": price - 1.0,
            "trend_ema_gap": 0.01,
            "entry_probability": np.where(np.arange(size) % 45 == 0, 0.8, 0.2),
            "exit_probability": np.where(np.arange(size) % 45 == 25, 0.8, 0.2),
            "entry_label": np.where(np.arange(size) % 2 == 0, 1, 0),
            "exit_label": np.where(np.arange(size) % 3 == 0, 1, 0),
        }
    )


def test_simulation_returns_financial_summary() -> None:
    parameters = default_parameters()
    request = ResolvedMLBacktestCreate(
        symbol="BTCUSDT",
        period_years=4,
        decision_timeframe="30min",
        trend_timeframe="4hour",
        initial_capital=1000,
        fee_rate=0.001,
        slippage_rate=0.0005,
        entry_threshold=0.65,
        exit_threshold=0.60,
        reinvestment=True,
    )
    frame = prediction_frame()
    simulation = simulate_portfolio(frame, request, parameters)
    result = calculate_results(
        frame,
        request,
        simulation,
        {"fold_count": 3, "feature_importance": [{"feature": "rsi_14", "importance": 0.2}]},
        parameters,
    )
    assert result["summary"]["capital_final"] > 0
    assert result["execution"]["walk_forward_folds"] == 3
    assert result["execution"]["parameter_source"] == "DATABASE_SNAPSHOT"
    assert len(result["equity_curve"]) > 0
