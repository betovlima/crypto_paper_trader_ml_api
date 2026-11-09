"""V15 causal BTC research: additional indicators, delayed fills, audit markers."""
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from crypto_paper_trader_ml_api.app import app
from crypto_paper_trader_ml_api import service
from crypto_paper_trader_ml_api.btc_trend_regimes import (
    POLICIES, completed_bars, build_decision_map, simulate_policy,
    evaluate_archived_trend_regimes,
)
from crypto_paper_trader_ml_api.database import MLBacktestAudit, SessionLocal
from .test_btc_event_research import _event_archive
from .test_strategy_research import _insert_source, _cleanup


def _candles(days=135):
    ts = pd.date_range("2024-01-01", periods=days * 48, freq="30min", tz="UTC")
    cycle = np.arange(len(ts), dtype=float)
    close = 100 + cycle * .018 + 5 * np.sin(cycle / 65)
    return pd.DataFrame({
        "timestamp": ts, "open": close, "high": close + .25,
        "low": close - .25, "close": close, "volume": np.full(len(ts), 1.0),
    })


def test_only_closed_higher_timeframe_candles_are_eligible():
    frame = _candles(8)
    full = completed_bars(frame, source_seconds=1800,
                          bar_seconds=4 * 3600, frequency="4h")
    assert full.iloc[0]["end"] == pd.Timestamp("2024-01-01T04:00:00Z")
    assert len(full) == 8 * 6
    incomplete = frame.loc[frame["timestamp"] != pd.Timestamp("2024-01-01T02:30:00Z")].reset_index(drop=True)
    changed = completed_bars(incomplete, source_seconds=1800,
                             bar_seconds=4 * 3600, frequency="4h")
    assert pd.Timestamp("2024-01-01T04:00:00Z") not in set(changed["end"])
    assert len(changed) == len(full) - 1


def test_future_candles_do_not_change_past_4h_or_daily_decisions():
    frame = _candles(135)
    revised = frame.copy()
    cut = pd.Timestamp("2024-04-30T00:00:00Z")
    mask = revised["timestamp"] >= cut
    revised.loc[mask, ["open", "high", "low", "close"]] *= 2
    original = build_decision_map(frame, 1800)
    modified = build_decision_map(revised, 1800)
    same_times = [at for at in original if at < cut and at in modified]
    assert same_times
    for at in same_times:
        assert set(original[at]) == set(modified[at])
        for period in original[at]:
            a, da, wa = original[at][period]
            b, db, wb = modified[at][period]
            assert a["close"] == b["close"]
            if da is not None:
                assert da["close"] == db["close"]
            if wa is not None:
                assert wa["close"] == wb["close"]


def test_policy_reinvests_without_leverage_and_records_actual_next_opens():
    frame = _candles()
    decisions = build_decision_map(frame, 1800)
    assert len(POLICIES) == 12
    for policy, _ in POLICIES:
        result = simulate_policy(frame, 0, len(frame), policy,
                                 source_seconds=1800, slippage=.0005,
                                 decision_map=decisions, include_actions=True)
        assert result["initial_capital"] == 10000
        assert result["capital_final"] > 0
        assert result["max_drawdown"] <= 0
        assert 0 <= result["exposure"] <= 1
        assert result["operation_count"] == len(result["actions"])
        assert result["completed_cycles"] <= result["sell_count"]
        for action in result["actions"]:
            assert action["target_fraction"] in (0.0, .5, 1.0)
            if action["signal_at"]:
                assert pd.Timestamp(action["timestamp"]) - pd.Timestamp(action["signal_at"]) == pd.Timedelta(minutes=30)
                assert action["signal_price"] is not None
                assert action["execution_price"] == frame.at[action["execution_candle_index"], "open"]


def test_source_oos_is_not_used_to_approve_or_generate_trades():
    baseline = evaluate_archived_trend_regimes(_event_archive())
    modified = evaluate_archived_trend_regimes(_event_archive(post_oos_mutation=True))
    assert baseline["trials"] == modified["trials"]
    assert baseline["summary"]["chosen_only_on_development"] == modified["summary"]["chosen_only_on_development"]
    assert baseline["summary"]["approved_policies"] == modified["summary"]["approved_policies"]
    assert len(baseline["trials"]) == len(POLICIES)
    assert not baseline["summary"]["oos_executed"]
    assert baseline["summary"]["cash_baseline"]["validation"]["capital_final"] == 10000
    assert not baseline["summary"]["approved_policies"]  # short toy archive has no 100-day warmup


def test_rejected_trials_never_expose_a_promoted_winner():
    result = evaluate_archived_trend_regimes(_event_archive())
    summary = result["summary"]
    assert summary["result_status"] == "NO_STRATEGY_APPROVED"
    assert summary["approved_validation"] is None
    assert summary["approved_policies"] == []
    assert summary["chart"]["diagnostic_only"] is True
    assert "inflections" in summary["chart"]
    assert "signals" in summary["chart"]
    assert summary["rejected_count"] == len(POLICIES)


def test_one_click_mode_persists_and_exports_all_rejected_trials(monkeypatch):
    source = _insert_source()
    tasks = []

    class Executor:
        def submit(self, fn, *args):
            tasks.append((fn, args))

    monkeypatch.setattr(service, "_get_executor", lambda settings: Executor())
    with SessionLocal.begin() as session:
        session.get(MLBacktestAudit, source).archive_zip = _event_archive()
    try:
        with TestClient(app) as client:
            response = client.post("/v1/ml-strategies", json={
                "name": "BTC indicator lab V15", "source_job_id": source,
            })
            assert response.status_code == 201, response.text
            sid = response.json()["id"]
            started = client.post(f"/v1/ml-strategies/{sid}/optimize", json={
                "mode": "INDICATOR_LAB", "trials": len(POLICIES),
                "auto_execute": False,
            })
            assert started.status_code == 202, started.text
            assert len(tasks) == 1
            tasks[0][0](*tasks[0][1])
            saved = client.get(f"/v1/ml-strategies/{sid}").json()
            assert saved["status"] == "RESEARCH_COMPLETE", saved
            assert saved["mode"] == "INDICATOR_LAB"
            assert saved["champion"] is None
            assert saved["execution_job_id"] is None
            assert saved["summary"]["approved_policies"] == []
            exported = client.get(f"/v1/ml-strategies/{sid}/report.zip")
            assert exported.status_code == 200, exported.text
            with ZipFile(BytesIO(exported.content)) as z:
                assert {"trend_regimes.json", "trend_regimes_portfolios.csv",
                        "trend_regimes_trades.csv", "indicator_lab_markers.csv",
                        "source_historical_audit.zip"} <= set(z.namelist())
                assert b"NO_STRATEGY_APPROVED" in z.read("trend_regimes.json")
    finally:
        _cleanup(source)


def test_v14_mode_still_advertises_six_original_policies():
    from crypto_paper_trader_ml_api.btc_trend_regimes import V14_POLICIES
    assert len(V14_POLICIES) == 6
    assert POLICIES[:6] == V14_POLICIES
    old = evaluate_archived_trend_regimes(_event_archive(), policy_set=V14_POLICIES)
    assert len(old["trials"]) == 6
    assert old["summary"]["candidate_count"] == 6


def test_v15_indicators_have_no_future_data_leakage():
    frame = _candles(135)
    cut = pd.Timestamp("2024-04-30T00:00:00Z")
    changed = frame.copy()
    changed.loc[changed["timestamp"] >= cut, ["open", "close", "high", "low"]] *= 1.7
    before = completed_bars(frame, source_seconds=1800, bar_seconds=4*3600, frequency="4h")
    after = completed_bars(changed, source_seconds=1800, bar_seconds=4*3600, frequency="4h")
    fields = ["rsi14", "macd", "macd_hist", "adx14", "di_plus", "di_minus",
              "boll_upper", "boll_lower", "boll_width",
              "kelt_upper", "kelt_lower", "roc20", "stoch_k", "stoch_d", "ema200"]
    pd.testing.assert_frame_equal(
        before.loc[before["end"] <= cut, fields].reset_index(drop=True),
        after.loc[after["end"] <= cut, fields].reset_index(drop=True),
    )


def test_extrema_confirmation_is_strictly_after_historical_pivot():
    from crypto_paper_trader_ml_api.btc_directional_change import find_confirmed_extrema
    frame = _candles(135)
    detected = find_confirmed_extrema(frame, len(frame)-35*48, len(frame), .02, 1800)
    assert detected
    for marker in detected:
        assert marker["confirmed_index"] > marker["pivot_index"]
        assert marker["lag_candles"] >= 1
        assert marker["threshold_fraction"] == .02


def test_delayed_fills_require_a_real_intermediate_candle():
    frame = _candles()
    decisions = build_decision_map(frame, 1800)
    result = simulate_policy(frame, 0, len(frame), "DAILY_SMA50_BAND",
                             source_seconds=1800, slippage=.0005,
                             decision_map=decisions, include_actions=True)
    signals = [a for a in result["actions"] if a["signal_at"]]
    assert signals, "Synthetic rising market must generate at least one causal entry"
    for action in signals:
        signal_t = pd.Timestamp(action["signal_at"])
        fill_t = pd.Timestamp(action["timestamp"])
        assert fill_t == signal_t + pd.Timedelta(minutes=30)
        intermediate = frame.loc[frame["timestamp"] == signal_t]
        assert len(intermediate) == 1
        assert action["execution_price"] == frame.at[action["execution_candle_index"], "open"]
        assert fill_t > signal_t


def test_market_gap_cancels_a_pending_order():
    frame = _candles(135)
    # Drop the first 30m candle at 00:00 after a complete daily close;
    # a signal pending from 00:00 may NOT fill at 00:30.
    candles = frame.loc[frame["timestamp"] != pd.Timestamp("2024-03-10T00:00:00Z")]
    candles = candles.reset_index(drop=True)
    decisions = build_decision_map(candles, 1800)
    result = simulate_policy(candles, 0, len(candles), "DAILY_SMA50_BAND",
                             source_seconds=1800, slippage=.0005,
                             decision_map=decisions, include_actions=True)
    for trade in result["actions"]:
        if trade["signal_at"]:
            assert pd.Timestamp(trade["timestamp"]) - pd.Timestamp(trade["signal_at"]) == pd.Timedelta(minutes=30)
