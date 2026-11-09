"""Causal Bitcoin extrema detection, temporal isolation and full research export."""
from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from crypto_paper_trader_ml_api import strategy_research, service
from crypto_paper_trader_ml_api.app import app
from crypto_paper_trader_ml_api.btc_directional_change import (
    find_confirmed_extrema, capitalize_confirmations,
    evaluate_archived_directional_change, THRESHOLDS,
)
from crypto_paper_trader_ml_api.database import MLBacktestAudit, SessionLocal
from .test_btc_event_research import _event_archive
from .test_strategy_research import _insert_source, _cleanup


def _toy_prices():
    prices = [100, 98, 95, 96, 97, 102, 110, 109, 108, 107, 103, 99, 100, 101]
    dates = pd.date_range("2024-01-01", periods=len(prices), freq="30min", tz="UTC")
    return pd.DataFrame({
        "timestamp": dates, "open": prices,
        "high": prices, "low": prices, "close": prices,
        "volume": [1] * len(prices),
        "sigma": [.004] * len(prices),
    })


def test_pivot_is_retrospective_and_buy_occurs_only_on_next_open():
    df = _toy_prices()
    events = find_confirmed_extrema(df, 0, len(df), .02, 1800)
    bottom = next(e for e in events if e["kind"] == "BOTTOM_CONFIRMED")
    top = next(e for e in events if e["kind"] == "TOP_CONFIRMED")
    assert bottom["pivot_index"] == 2
    assert bottom["pivot_close"] == 95
    assert bottom["confirmed_index"] == 4
    assert bottom["confirmed_close"] == 97
    assert top["pivot_close"] == 110
    assert top["confirmed_index"] == 9
    assert bottom["pivot_timestamp"] < bottom["confirmed_at"]
    result = capitalize_confirmations(
        df, 0, len(df), events, bar_seconds=1800,
        slippage=.0005, include_actions=True,
    )
    actions = result["actions"]
    assert [a["action"] for a in actions] == ["BUY", "SELL"]
    assert actions[0]["execution_candle_index"] == bottom["confirmed_index"] + 1
    assert actions[0]["timestamp"] > bottom["confirmed_at"]
    assert actions[0]["execution_price"] == df.at[5, "open"]
    assert actions[1]["execution_candle_index"] == top["confirmed_index"] + 1
    assert actions[0]["fee_usd"] > 0
    assert actions[1]["fee_usd"] > 0
    assert result["completed_cycles"] == 1


def test_past_signals_are_unchanged_when_only_future_prices_change():
    df = _toy_prices()
    base = find_confirmed_extrema(df, 0, 11, .02, 1800)
    changed = df.copy()
    changed.loc[11:, "close"] *= 7
    assert base == find_confirmed_extrema(changed, 0, 11, .02, 1800)


def test_gap_does_not_manufacture_pivot_confirmation():
    df = _toy_prices()
    df.at[4, "timestamp"] = df.at[3, "timestamp"] + pd.Timedelta(hours=2)
    events = find_confirmed_extrema(df, 0, 5, .02, 1800)
    assert all(e["confirmed_index"] != 4 for e in events)


def test_all_thresholds_are_predeclared_and_adaptive_is_bounded():
    from crypto_paper_trader_ml_api.btc_directional_change import _threshold
    df = _toy_prices()
    assert len(THRESHOLDS) == 4
    assert [x[1] for x in THRESHOLDS] == [.01, .02, .03, None]
    assert _threshold(df, 5, None, 1800) >= .01
    assert _threshold(df, 5, None, 1800) <= .04


def test_archived_holdout_is_unchanged_when_only_oos_candles_are_modified():
    before = evaluate_archived_directional_change(_event_archive())
    after = evaluate_archived_directional_change(_event_archive(post_oos_mutation=True))
    assert before["trials"] == after["trials"]
    assert before["summary"] == after["summary"]
    assert len(before["trials"]) == 4
    assert all(row["metrics"]["development"] for row in before["trials"])
    assert before["summary"]["oos_executed"] is False
    assert before["summary"]["first_oos_untouched_from"].startswith("2024-02")
    assert before["summary"]["cash_baseline"]["validation"]["capital_final"] == 10000
    assert before["summary"]["buy_hold_baseline"]["validation"]["completed_cycles"] == 1


def test_full_trade_replay_does_not_change_persisted_strategy_metrics():
    blob = _event_archive()
    ordinary = evaluate_archived_directional_change(blob)
    export = evaluate_archived_directional_change(blob, include_details=True)
    assert ordinary["trials"] == export["trials"]
    assert ordinary["summary"] == export["summary"]
    assert any(row["kind"] == "BOTTOM_CONFIRMED" for row in export["extrema"])
    assert any(row["kind"] == "TOP_CONFIRMED" for row in export["extrema"])
    assert any(row["action"] == "BUY" for row in export["trades"])
    assert any(row["action"] == "SELL" for row in export["trades"])
    assert all(row["confirmed_at"] < row["timestamp"]
               for row in export["trades"] if row["signal_type"] != "END_OF_WINDOW")


def test_one_click_directional_study_stores_and_exports_without_oos(monkeypatch):
    source = _insert_source()
    tasks = []

    class Executor:
        def submit(self, func, *args):
            tasks.append((func, args))
            return None

    monkeypatch.setattr(service, "_get_executor", lambda settings: Executor())
    with SessionLocal.begin() as session:
        session.get(MLBacktestAudit, source).archive_zip = _event_archive()
    try:
        with TestClient(app) as client:
            created = client.post("/v1/ml-strategies", json={
                "name": "BTC minima maxima Directional Change",
                "source_job_id": source,
            })
            assert created.status_code == 201, created.text
            sid = created.json()["id"]
            launch = client.post(f"/v1/ml-strategies/{sid}/optimize", json={
                "mode": "DIRECTIONAL_CHANGE",
                "trials": 4,
                "auto_execute": True,
            })
            assert launch.status_code == 202, launch.text
            assert len(tasks) == 1
            tasks[0][0](*tasks[0][1])
            saved = client.get(f"/v1/ml-strategies/{sid}").json()
            assert saved["status"] == "RESEARCH_COMPLETE", saved
            assert saved["mode"] == "DIRECTIONAL_CHANGE"
            assert saved["champion"] is None
            assert saved["execution_job_id"] is None
            assert len(saved["trials"]) == 4
            assert saved["summary"]["oos_executed"] is False
            exported = client.get(f"/v1/ml-strategies/{sid}/report.zip")
            assert exported.status_code == 200, exported.text
            with ZipFile(BytesIO(exported.content)) as z:
                for name in (
                    "directional_change.json", "directional_change_portfolios.csv",
                    "directional_change_extrema.csv", "directional_change_trades.csv",
                    "source_historical_audit.zip",
                ):
                    assert name in z.namelist()
                entries = z.read("directional_change_extrema.csv").decode()
                fills = z.read("directional_change_trades.csv").decode()
                assert "pivot_timestamp,pivot_close" in entries
                assert "confirmed_at,confirmed_close" in entries
                assert "execution_price" in fills
                assert "CONFIRMED_BOTTOM_NEXT_OPEN" in fills
    finally:
        _cleanup(source)
