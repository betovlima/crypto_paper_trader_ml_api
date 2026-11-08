"""Historical BTC/CASH rotation trace regression tests."""

import pandas as pd
import pytest

from crypto_paper_trader_ml_api.engine import calculate_results, simulate_portfolio
from crypto_paper_trader_ml_api.rotation_analysis import build_rotation_timeline
from crypto_paper_trader_ml_api.schemas import ResolvedMLBacktestCreate

from .helpers import default_parameters


def _row(index, price, position="CASH", actions=None):
    return {
        "timestamp": f"2026-01-01T{index:02d}:00:00+00:00",
        "btc_close": price,
        "position": position,
        "equity": 1000.0,
        "buy_hold": 1000.0,
        "actions": actions or [],
    }


def _request():
    return ResolvedMLBacktestCreate(
        symbol="BTCUSDT", period_years=1,
        decision_timeframe="30min", trend_timeframe="4hour",
        initial_capital=1000.0, fee_rate=0.001,
        slippage_rate=0.0005, entry_threshold=0.65,
        exit_threshold=0.6, reinvestment=True,
    )


def test_distance_to_peak_uses_only_closing_prices_observed_so_far():
    trace = build_rotation_timeline(
        [_row(0, 100), _row(1, 108), _row(2, 102), _row(3, 120)], 100
    )
    assert [row["peak_close_so_far"] for row in trace] == [100, 108, 108, 120]
    assert trace[2]["distance_to_peak"] == pytest.approx(102 / 108 - 1)
    assert trace[0]["distance_to_peak"] == 0
    assert trace[3]["distance_to_peak"] == 0


def test_sampling_retains_every_executed_switch():
    rows = [_row(i, 100 + i, "BTC" if 4 <= i < 8 else "CASH") for i in range(18)]
    rows[4]["actions"] = [{"action": "BUY", "execution_price": 104, "reason": "XGB_ENTRY"}]
    rows[8]["actions"] = [{"action": "SELL", "execution_price": 108, "reason": "XGB_EXIT"}]
    trace = build_rotation_timeline(rows, maximum_points=2)
    assert [action["action"] for row in trace for action in row["actions"]] == ["BUY", "SELL"]
    assert trace[0]["timestamp"] == rows[0]["timestamp"]
    assert trace[-1]["timestamp"] == rows[-1]["timestamp"]
    assert any(item["position"] == "BTC" for item in trace)
    assert len(trace) > 2


def test_two_actions_on_one_candle_remain_two_distinct_events():
    actions = [
        {"action": "BUY", "execution_price": 101, "reason": "XGB_ENTRY"},
        {"action": "SELL", "execution_price": 98, "reason": "PROTECTIVE_STOP"},
    ]
    trace = build_rotation_timeline([_row(0, 100), _row(1, 99, "CASH", actions)], 1)
    assert [event["action"] for event in trace[-1]["actions"]] == ["BUY", "SELL"]
    assert trace[-1]["position"] == "CASH"


def test_empty_legacy_trace_is_allowed():
    assert build_rotation_timeline([], 100) == []


def test_invalid_sampling_budget_or_close_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        build_rotation_timeline([_row(0, 100)], 0)
    with pytest.raises(ValueError, match="positive"):
        build_rotation_timeline([_row(0, 0)], 100)


def test_portfolio_simulator_returns_exact_positions_and_end_to_end_trace():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="30min", tz="UTC"),
        "open": [100.0] * 5,
        "high": [100.1] * 5,
        "low": [99.9] * 5,
        "close": [100.0] * 5,
        "atr_14": [1.0] * 5,
        "ema_21": [101.0] * 5,
        "ema_50": [99.0] * 5,
        "trend_ema_gap": [0.01] * 5,
        "entry_probability": [0.9, 0.1, 0.1, 0.1, 0.1],
        "exit_probability": [0.1, 0.9, 0.1, 0.1, 0.1],
        "entry_label": [1, 0, 0, 0, 1],
        "exit_label": [0, 0, 1, 0, 0],
    })
    params = default_parameters(**{
        "simulation.initial_stop_atr": 100.0,
        "simulation.initial_target_atr": 100.0,
        "simulation.trailing_stop_atr": 100.0,
        "simulation.minimum_trend_gap": 0.0,
    })
    request = _request()
    sim = simulate_portfolio(frame, request, params)
    rows = sim["rotation_rows"]

    assert len(rows) == len(frame)
    assert rows[0]["position"] == "CASH"
    assert rows[1]["position"] == "BTC"
    assert rows[1]["actions"][0]["action"] == "BUY"
    assert rows[2]["position"] == "CASH"
    assert rows[2]["actions"][0]["action"] == "SELL"

    result = calculate_results(
        frame, request, sim, {"fold_count": 1, "feature_importance": []}, params
    )
    assert result["rotation_timeline"]
    assert result["rotation_timeline"][-1]["position"] == "CASH"
    assert "distance_to_peak" in result["rotation_timeline"][0]
    assert set(result["equity_curve"][0]) == {"timestamp", "equity", "buy_hold"}

def test_terminal_liquidation_is_reflected_in_final_position_and_actions():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=4, freq="30min", tz="UTC"),
        "open": [100.0] * 4, "high": [100.1] * 4,
        "low": [99.9] * 4, "close": [100.0] * 4,
        "atr_14": [1.0] * 4, "ema_21": [101.0] * 4,
        "ema_50": [99.0] * 4, "trend_ema_gap": [0.01] * 4,
        "entry_probability": [0.9, 0.1, 0.1, 0.1],
        "exit_probability": [0.1, 0.1, 0.1, 0.1],
    })
    params = default_parameters(**{
        "simulation.initial_stop_atr": 100.0,
        "simulation.initial_target_atr": 100.0,
        "simulation.trailing_stop_atr": 100.0,
        "simulation.minimum_trend_gap": 0.0,
    })
    simulated = simulate_portfolio(frame, _request(), params)
    last = simulated["rotation_rows"][-1]
    assert last["position"] == "CASH"
    assert last["actions"][-1]["action"] == "SELL"
    assert last["actions"][-1]["reason"] == "END_OF_TEST"
    assert last["equity"] == simulated["equity_curve"][-1]["equity"]
