"""Regression checks for causal V1/V2 BTC/CASH research replay."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd
import pytest

from crypto_paper_trader_ml_api.capitalization import (
    _joint_signals, _signals, _summary_series, compare_capitalization,
)
from .test_archive_replay_ui import _make_archive


def _three_forecasts(predicted: float) -> bytes:
    source = _make_archive(predicted=predicted)
    with ZipFile(BytesIO(source)) as original:
        parts = {name: original.read(name) for name in original.namelist()}
    original_row = pd.read_csv(BytesIO(parts["forecast_predictions_oos.csv"])).iloc[0].to_dict()
    forecasts = []
    for i in range(3):
        row = dict(original_row)
        row["timestamp"] = (pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(minutes=30 * i)).isoformat()
        row["forecast_future_at_utc"] = (pd.Timestamp(row["timestamp"]) + pd.Timedelta(hours=2, minutes=30)).isoformat()
        forecasts.append(row)
    parts["forecast_predictions_oos.csv"] = pd.DataFrame(forecasts).to_csv(index=False).encode()
    target = BytesIO()
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as output:
        for name, blob in parts.items():
            output.writestr(name, blob)
    return target.getvalue()


def _all_horizon_forecasts(predicted: float) -> bytes:
    """Create three simultaneous OOS forecast targets on the same candles."""
    from json import dumps, loads
    source = _three_forecasts(predicted)
    with ZipFile(BytesIO(source)) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    base = pd.read_csv(BytesIO(parts["forecast_predictions_oos.csv"]))
    joined = []
    for horizon in (2, 6, 24):
        frame = base.copy()
        frame["forecast_horizon_hours"] = horizon
        frame["forecast_future_at_utc"] = (
            pd.to_datetime(frame["timestamp"], utc=True)
            + pd.Timedelta(hours=horizon, minutes=30)
        )
        joined.append(frame)
    parts["forecast_predictions_oos.csv"] = pd.concat(
        joined, ignore_index=True
    ).to_csv(index=False).encode()
    folds = loads(parts["folds.json"])
    parts["folds.json"] = dumps([
        {**folds[0], "horizon_hours": horizon}
        for horizon in (2, 6, 24)
    ]).encode()
    metrics = loads(parts["forecast_metrics.json"])
    parts["forecast_metrics.json"] = dumps([
        {**metrics[0], "horizon_hours": horizon,
         "metrics": {**metrics[0]["metrics"], "rows": 3}}
        for horizon in (2, 6, 24)
    ]).encode()
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as zipped:
        for name, blob in parts.items():
            zipped.writestr(name, blob)
    return output.getvalue()


def test_optimized_policy_capitalization_uses_one_portfolio_and_benchmark_only():
    from crypto_paper_trader_ml_api.btc_policy import DEFAULT_POLICY
    report = compare_capitalization(
        _all_horizon_forecasts(.0001),
        _all_horizon_forecasts(.015),
        **{**_options(), "horizon": 0},
        optimized_policy={**DEFAULT_POLICY, "policy.entry_probability": .52,
                          "policy.min_hold_bars": 0, "policy.cooldown_bars": 1},
    )
    assert report["status"] == "VERIFIED"
    assert report["portfolio_rule"] is not None
    assert report["optimized_policy"]["policy.enabled"]
    assert [row["code"] for row in report["results"]] == [
        "BTC_CASH_ROTATION", "BUY_AND_HOLD",
    ]
    rotation, baseline = report["results"]
    assert rotation["initial_capital"] == baseline["initial_capital"] == 10000
    assert rotation["switch_count"] >= 2
    assert rotation["actions"][0]["action"] == "BUY"
    assert all(row["capital_after"] >= 0 for row in rotation["actions"])
    assert rotation["execution_rule"].startswith("Prior candle close")


def test_optimized_policy_rejects_isolated_horizon():
    from crypto_paper_trader_ml_api.btc_policy import DEFAULT_POLICY
    with pytest.raises(ValueError, match="conjuntamente"):
        compare_capitalization(
            _all_horizon_forecasts(.0001),
            _all_horizon_forecasts(.015),
            **_options(), optimized_policy=DEFAULT_POLICY,
        )



def test_joint_signal_requires_two_positive_horizons_and_same_timestamp():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    signal_at = start.isoformat()
    signals = {
        2: {1: {"predicted_return": 0.01, "p_up": 0.8, "signal_at": signal_at},
            2: {"predicted_return": 0.01, "p_up": 0.8, "signal_at": signal_at}},
        6: {1: {"predicted_return": 0.01, "p_up": 0.8, "signal_at": signal_at},
            2: {"predicted_return": -0.01, "p_up": 0.4, "signal_at": signal_at}},
        24: {1: {"predicted_return": -0.02, "p_up": 0.4, "signal_at": signal_at},
             2: {"predicted_return": -0.02, "p_up": 0.4, "signal_at": signal_at}},
    }
    combined = _joint_signals(signals)
    assert combined[1]["positive_votes"] == 2
    assert combined[2]["positive_votes"] == 1
    assert combined[1]["horizons_hours"] == [2, 6, 24]
    assert combined[1]["p_up"] == pytest.approx(2 / 3)


def test_joint_capitalization_is_one_portfolio_not_mean_of_three():
    report = compare_capitalization(
        _all_horizon_forecasts(0.0001), _all_horizon_forecasts(0.01),
        **{**_options(), "horizon": 0},
    )
    assert report["status"] == "VERIFIED"
    assert report["horizon_hours"] == 0
    assert report["horizons_hours"] == [2, 6, 24]
    assert report["signal_count"] == 3
    assert "2 of 3" in report["policy"]
    original, candidate, buy_hold = report["results"]
    assert original["capital_final"] == 10000
    assert original["switch_count"] == 0
    assert candidate["switch_count"] == 2
    assert buy_hold["switch_count"] == 2



def _options() -> dict:
    return {
        "horizon": 2, "initial_capital": 10000,
        "fee_rate": 0.001, "slippage_rate": 0.0005,
        "entry_edge": 0, "min_probability": 0.5, "exit_return": 0,
    }


def test_v2_can_buy_while_original_remains_cash_on_identical_oos():
    original = _three_forecasts(0.0001)
    replay = _three_forecasts(0.01)
    report = compare_capitalization(original, replay, **_options())
    assert report["status"] == "VERIFIED"
    assert report["signal_count"] == 3
    v1, v2, hold = report["results"]
    assert v1["switch_count"] == 0
    assert v1["capital_final"] == 10000
    assert v2["switch_count"] == 2
    assert v2["actions"][0]["action"] == "BUY"
    assert v2["actions"][0]["timestamp"] == "2024-01-01T00:30:00+00:00"
    assert v2["actions"][0]["signal_at"] == "2024-01-01T00:30:00+00:00"
    assert hold["switch_count"] == 2
    assert v2["capital_final"] > 0


def test_replay_rejects_changed_market_candles():
    with pytest.raises(ValueError, match="Candles históricos diferentes"):
        compare_capitalization(
            _three_forecasts(0.0001),
            _make_archive(predicted=0.01, market_suffix="changed"),
            **_options(),
        )


def test_gap_never_receives_synthetic_or_stale_execution():
    source = _make_archive(predicted=0.01)
    with ZipFile(BytesIO(source)) as z:
        import csv
        from io import TextIOWrapper
        with z.open("market_history_decision.csv") as stream:
            rows = list(csv.DictReader(TextIOWrapper(stream, encoding="utf-8-sig")))
    # Explicit missing bar at 00:30 means 00:00 forecast must not execute at 01:00.
    rows.pop(1)
    from crypto_paper_trader_ml_api.capitalization import _utc
    candles = [
        {"timestamp": _utc(row["timestamp"]), "open": float(row["open"]), "close": float(row["close"])}
        for row in rows
    ]
    with pytest.raises(ValueError, match="Não há sinais executáveis"):
        _signals(source, 2, candles, 1800)


def test_no_trade_equity_and_costs_are_finite():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = [
        {"timestamp": start + timedelta(minutes=30 * i), "open": 100 + i, "close": 100 + i}
        for i in range(5)
    ]
    result = _summary_series(
        candles, {}, start=1, end=4, capital=10000, fee=0.001,
        slippage=0.0005, edge=0, min_probability=0.5,
        exit_return=0, baseline=False, label="V1", bar_seconds=1800,
    )
    assert result["capital_final"] == pytest.approx(10000)
    assert result["switch_count"] == 0
    assert result["exposure"] == 0
