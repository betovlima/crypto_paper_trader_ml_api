from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_buy_and_hold_deducts_the_same_trade_costs_as_strategy():
    request = ResolvedMLBacktestCreate(
        symbol="BTCUSDT", period_years=1, decision_timeframe="30min",
        trend_timeframe="4hour", initial_capital=1000,
        fee_rate=0.001, slippage_rate=0.0005,
        entry_threshold=0.99, exit_threshold=0.99,
        reinvestment=True,
    )
    frame = prediction_frame(40)
    simulation = simulate_portfolio(frame, request, default_parameters())
    first_close = float(frame.iloc[0]["close"])
    last_close = float(frame.iloc[-1]["close"])
    bought_quantity = 1000 * (1 - 0.001) / (first_close * (1 + 0.0005))
    expected_final = bought_quantity * last_close * (1 - 0.0005) * (1 - 0.001)
    assert simulation["equity_curve"][-1]["buy_hold"] == pytest.approx(expected_final)
    assert simulation["trades"] == []
    assert simulation["signal_diagnostics"]["entry_probability_above_threshold"] == 0
    assert simulation["signal_diagnostics"]["entry_probability_max"] == pytest.approx(0.8)
    frame = frame.assign(
        entry_baseline_probability=0.5, exit_baseline_probability=0.5,
    )
    result = calculate_results(
        frame, request, simulation, {"fold_count": 1, "feature_importance": []},
        default_parameters(),
    )
    assert result["summary"]["has_executed_trades"] is False
    assert result["insights"]["no_trade_reason"] == "ENTRY_PROBABILITY_BELOW_THRESHOLD"
    assert result["insights"]["benchmark_costs_included"] is True
    assert result["models"]["entry"]["baseline_brier_score"] is not None
    assert result["models"]["entry"]["brier_skill_score"] is not None
    assert result["comparisons"][1]["capital_final"] == pytest.approx(expected_final, abs=1e-3)


def test_buy_and_hold_drawdown_includes_entry_and_exit_execution_costs():
    request = ResolvedMLBacktestCreate(
        symbol="BTCUSDT", period_years=1, decision_timeframe="30min",
        trend_timeframe="4hour", initial_capital=1000,
        fee_rate=0.002, slippage_rate=0.001,
        entry_threshold=0.99, exit_threshold=0.99, reinvestment=True,
    )
    frame = prediction_frame(8)
    for column in ("open", "high", "low", "close"):
        frame[column] = 100.0
    params = default_parameters()
    simulation = simulate_portfolio(frame, request, params)
    result = calculate_results(
        frame, request, simulation, {"fold_count": 1, "feature_importance": []},
        params,
    )
    buy_and_hold = result["comparisons"][1]
    assert buy_and_hold["max_drawdown"] < 0
    assert buy_and_hold["return_total"] < 0
    assert result["summary"]["max_drawdown"] == 0
