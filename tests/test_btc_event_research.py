"""BTC-specific CUSUM/Triple Barrier event research: causal fills and pre-OOS isolation."""
from __future__ import annotations

import json
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from crypto_paper_trader_ml_api import service, strategy_research
from crypto_paper_trader_ml_api.app import app
from crypto_paper_trader_ml_api.btc_event_research import (
    _build_data, _eligible, _labels, _portfolio,
    evaluate_archived_event_research, POLICIES,
)
from crypto_paper_trader_ml_api.database import MLBacktestAudit, SessionLocal
from .test_archive_replay_ui import _make_archive
from .test_features import synthetic_candles
from .test_strategy_research import _cleanup, _give_source_three_horizons, _insert_source


def _event_candles(size=2700):
    frame = synthetic_candles(size)
    # Deterministic BTC-like jumps mixed with calm and volatile spells,
    # with high and low enclosing open/close. No random data dependencies.
    steps = np.sin(np.arange(size) / 9) * .002 + np.cos(np.arange(size) / 22) * .001
    steps[np.arange(size) % 57 == 0] += .014
    steps[np.arange(size) % 91 == 0] -= .018
    close = 40000 * np.exp(np.cumsum(steps))
    op = np.concatenate([[close[0]], close[:-1]])
    frame["open"] = op
    frame["close"] = close
    frame["high"] = np.maximum(op, close) * 1.005
    frame["low"] = np.minimum(op, close) * .995
    return frame


def _event_archive(*, post_oos_mutation=False, size=2700):
    original = _make_archive()
    candle = _event_candles(size)
    first_oos = candle["timestamp"].iloc[min(2100, size - 100)]
    if post_oos_mutation:
        mask = candle["timestamp"] >= first_oos
        candle.loc[mask, "high"] *= 5
        candle.loc[mask, "open"] *= 2
        candle.loc[mask, "low"] *= .4
        candle.loc[mask, "close"] *= 3
    with ZipFile(BytesIO(original)) as z:
        fold = [{
            "fold": 1, "horizon_hours": h,
            "training_start": candle["timestamp"].iloc[0].isoformat(),
            "test_start": first_oos.isoformat(),
        } for h in (2, 6, 24)]
        changes = {
            "market_history_decision.csv": candle.to_csv(index=False),
            "market_history_trend.csv": candle.iloc[::8].to_csv(index=False),
            "folds.json": json.dumps(fold),
        }
        output = BytesIO()
        with ZipFile(output, "w") as out:
            for name in z.namelist():
                out.writestr(name, changes.get(name, z.read(name)))
    return output.getvalue()


def test_events_are_known_at_close_and_can_only_buy_next_open():
    df = _build_data(_event_candles(720), bar_seconds=1800)
    assert len(df) == 720
    assert int((df["event"] == "UP").sum() + (df["event"] == "DOWN").sum()) > 5
    for idx in range(1, len(df)):
        ok, event_at = _eligible(df, idx, "CUSUM_UP_BREAKOUT", 1800)
        if ok:
            assert event_at == idx - 1
            assert df.at[event_at, "event"] == "UP"
            assert df.at[idx, "timestamp"] > df.at[event_at, "timestamp"]


def test_both_barriers_in_same_30min_candle_never_count_as_success():
    dates = pd.date_range("2024-01-01", periods=20, freq="30min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": dates, "open": [100.0] * 20,
        "high": [105.0] * 20, "low": [95.0] * 20,
        "close": [100.0] * 20,
        "sigma": [.002] * 20,
        "event": ["UP"] + [""] * 19,
    })
    result = _labels(df, 0, len(df), 1800)
    assert result["UP"]["AMBIGUOUS_BOTH_CONSERVATIVE_LOSS"] == 1
    assert result["UP"].get("UPPER_FIRST", 0) == 0


def test_one_trade_per_bar_and_buy_and_hold_not_subject_to_event_barriers():
    df = _build_data(_event_candles(750), bar_seconds=1800)
    result = _portfolio(df, 380, 700, "CUSUM_UP_BREAKOUT",
                        fee_rate=.001, slippage=.0005, bar_seconds=1800)
    assert len(set(action["timestamp"] for action in result["actions"])) == len(result["actions"])
    assert result["completed_cycles"] == result["buys"] == result["sells"]
    hold = _portfolio(df, 380, 700, "BUY_HOLD",
                      fee_rate=.001, slippage=.0005, bar_seconds=1800)
    cash = _portfolio(df, 380, 700, "CASH",
                      fee_rate=.001, slippage=.0005, bar_seconds=1800)
    assert hold["completed_cycles"] == 1
    assert [action["action"] for action in hold["actions"]] == ["BUY", "SELL"]
    assert cash["capital_final"] == 10000.0
    assert cash["fees_paid"] == 0
    assert cash["completed_cycles"] == 0
    assert hold["fees_paid"] > 0


def test_archived_pre_oos_results_unaffected_by_what_happens_after_first_oos():
    base = evaluate_archived_event_research(_event_archive())
    replaced = evaluate_archived_event_research(_event_archive(post_oos_mutation=True))
    assert len(base["trials"]) == 4
    assert [t["parameters"]["policy"] for t in base["trials"]] == [x[0] for x in POLICIES]
    assert base["trials"] == replaced["trials"]
    b, x = base["summary"], replaced["summary"]
    for key in b:
        if key != "source_archive_sha256":
            assert b[key] == x[key], key
    assert b["source_archive_sha256"] != x["source_archive_sha256"]
    assert not b["oos_executed"]
    assert b["cash_baseline"]["validation"]["capital_final"] == 10000.0
    assert b["buy_hold_baseline"]["validation"]["completed_cycles"] == 1


def test_event_experiment_rejects_insufficient_historical_candles():
    with pytest.raises(ValueError, match="Poucos candles"):
        evaluate_archived_event_research(_event_archive(size=1600))


def test_one_click_api_stores_and_exports_event_research_without_oos(monkeypatch):
    source = _insert_source()
    _give_source_three_horizons(source)
    with SessionLocal.begin() as session:
        session.get(MLBacktestAudit, source).archive_zip = _event_archive()
    tasks = []
    class Executor:
        def submit(self, func, *args):
            tasks.append((func, args))
    monkeypatch.setattr(service, "_get_executor", lambda _settings: Executor())
    try:
        with TestClient(app) as client:
            created = client.post("/v1/ml-strategies", json={
                "name": "BTC Event Discovery", "source_job_id": source,
            })
            assert created.status_code == 201, created.text
            identifier = created.json()["id"]
            job = client.post(f"/v1/ml-strategies/{identifier}/optimize", json={
                "mode": "EVENTS", "trials": 4, "auto_execute": True,
            })
            assert job.status_code == 202, job.text
            assert len(tasks) == 1
            tasks[0][0](*tasks[0][1])
            result = client.get(f"/v1/ml-strategies/{identifier}").json()
            assert result["status"] == "RESEARCH_COMPLETE", result
            assert result["execution_job_id"] is None
            assert result["champion"] is None
            assert result["summary"]["oos_executed"] is False
            assert result["summary"]["experiment"] == "BTC_CUSUM_TRIPLE_BARRIER_FIXED_BASELINES_V11"
            assert len(result["trials"]) == 4
            assert len(tasks) == 1
            exported = client.get(f"/v1/ml-strategies/{identifier}/report.zip")
            assert exported.status_code == 200, exported.text
            with ZipFile(BytesIO(exported.content)) as z:
                assert "source_historical_audit.zip" in z.namelist()
                assert "event_research.json" in z.namelist()
                assert "event_research_portfolios.csv" in z.namelist()
                assert "event_trade_actions.csv" in z.namelist()
                trade_audit = z.read("event_trade_actions.csv").decode()
                assert "fee_rate,fee_tier,fee_usd,slippage_usd" in trade_audit
                assert "CUSUM_UP_BREAKOUT" in trade_audit
                assert "BUY_HOLD" in trade_audit
                summary = json.loads(z.read("event_research.json"))["summary"]
                assert summary["oos_executed"] is False
                assert summary["cash_baseline"]["validation"]["capital_final"] == 10000.0
    finally:
        _cleanup(source)
