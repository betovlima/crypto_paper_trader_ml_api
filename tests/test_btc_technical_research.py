"""Safety, causality and API contract tests for the BTC technical research mode."""
from io import BytesIO
from zipfile import ZipFile

import pandas as pd
from fastapi.testclient import TestClient

from crypto_paper_trader_ml_api import service, strategy_research
from crypto_paper_trader_ml_api.app import app
from crypto_paper_trader_ml_api.btc_technical_research import (
    POLICIES, add_indicators, _signal, evaluate_archived_technical_research,
    simulate_policy,
)
from crypto_paper_trader_ml_api.database import MLBacktestAudit, SessionLocal
from .test_btc_event_research import _event_archive
from .test_strategy_research import _insert_source, _cleanup


def _prices():
    dates = pd.date_range("2025-01-01", periods=180, freq="30min", tz="UTC")
    prices = [100 + i * .22 if i < 80 else 117.6 - (i - 80) * .35 for i in range(180)]
    return pd.DataFrame({
        "timestamp": dates, "open": prices, "high": [p + .1 for p in prices],
        "low": [p - .1 for p in prices], "close": prices,
        "volume": [1.0] * len(prices),
    })


def test_causal_indicators_do_not_change_when_future_candles_change():
    before = _prices()
    altered = before.copy()
    altered.loc[125:, "close"] *= 5
    altered.loc[125:, "high"] *= 5
    original = add_indicators(before, 1800)
    changed = add_indicators(altered, 1800)
    cols = ["ema9", "ema21", "stoch_k", "stoch_d", "atr14", "upper20", "lower20"]
    pd.testing.assert_frame_equal(original.loc[:124, cols], changed.loc[:124, cols])
    for policy, _ in POLICIES[1:]:
        assert [_signal(original, i, policy, False, 0) for i in range(35, 125)] == [
            _signal(changed, i, policy, False, 0) for i in range(35, 125)]


def test_all_trades_execute_after_signal_and_always_leave_capitalized_cash():
    df = add_indicators(_prices(), 1800)
    assert len(POLICIES) == 6
    for name, _ in POLICIES:
        result = simulate_policy(df, 0, len(df), name, seconds=1800,
                                 slippage=.0005, include_actions=True)
        assert result["initial_capital"] == 10000
        assert result["capital_final"] > 0
        assert result["completed_cycles"] == sum(a["action"] == "SELL"
                                                  for a in result["actions"])
        for action in result["actions"]:
            if action["signal_at"] is not None:
                assert action["timestamp"] > action["signal_at"]
                assert action["execution_candle_index"] > 0
        assert result["max_drawdown"] <= 0


def test_archived_technical_comparison_keeps_pre_oos_holdout():
    original = evaluate_archived_technical_research(_event_archive())
    changed = evaluate_archived_technical_research(_event_archive(post_oos_mutation=True))
    assert original["trials"] == changed["trials"]
    assert original["summary"] == changed["summary"]
    assert len(original["trials"]) == len(POLICIES)
    assert original["summary"]["oos_executed"] is False
    assert original["summary"]["cash_baseline"]["validation"]["capital_final"] == 10000


def test_one_click_technical_research_and_export(monkeypatch):
    source = _insert_source()
    tasks = []

    class LocalExecutor:
        def submit(self, fn, *args):
            tasks.append((fn, args))

    monkeypatch.setattr(service, "_get_executor", lambda settings: LocalExecutor())
    with SessionLocal.begin() as session:
        session.get(MLBacktestAudit, source).archive_zip = _event_archive()
    try:
        with TestClient(app) as client:
            response = client.post("/v1/ml-strategies", json={
                "name": "BTC technical one click", "source_job_id": source,
            })
            assert response.status_code == 201, response.text
            sid = response.json()["id"]
            started = client.post(f"/v1/ml-strategies/{sid}/optimize", json={
                "mode": "TECHNICAL", "trials": len(POLICIES),
                "auto_execute": False,
            })
            assert started.status_code == 202, started.text
            assert len(tasks) == 1
            tasks[0][0](*tasks[0][1])
            result = client.get(f"/v1/ml-strategies/{sid}").json()
            assert result["status"] == "RESEARCH_COMPLETE", result
            assert result["mode"] == "TECHNICAL"
            assert result["execution_job_id"] is None
            assert result["champion"] is None
            exported = client.get(f"/v1/ml-strategies/{sid}/report.zip")
            assert exported.status_code == 200, exported.text
            with ZipFile(BytesIO(exported.content)) as z:
                names = set(z.namelist())
                assert {"technical_research.json", "technical_research_portfolios.csv",
                        "technical_research_trades.csv", "source_historical_audit.zip"} <= names
    finally:
        _cleanup(source)
