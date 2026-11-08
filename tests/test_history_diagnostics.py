"""Verify real history coverage is disclosed and exported without future data."""

import json
from io import BytesIO
from zipfile import ZipFile

from crypto_paper_trader_ml_api import engine
from crypto_paper_trader_ml_api.schemas import ResolvedMLBacktestCreate
from .helpers import default_parameters
from .test_simulation import prediction_frame


def test_backtest_reports_requested_vs_actual_history_and_exports_raw_candles(monkeypatch):
    raw = prediction_frame(48)
    fetched = []

    class FakeClient:
        def __init__(self, _parameters, _version):
            pass

        def fetch_history(self, *, symbol, timeframe, period_years, progress, cancelled):
            fetched.append((symbol, timeframe, period_years))
            return raw.copy()

        def close(self):
            pass

    def fake_features(decision, parameters, trend, **kwargs):
        assert kwargs["decision_timeframe"] == "30min"
        assert kwargs["trend_timeframe"] == "4hour"
        return decision

    monkeypatch.setattr(engine, "MEXCHistoryClient", FakeClient)
    monkeypatch.setattr(engine, "build_features", fake_features)
    monkeypatch.setattr(engine, "create_labels", lambda frame, parameters: frame)
    monkeypatch.setattr(
        engine, "walk_forward_predict",
        lambda frame, params, progress, cancelled: (
            frame.copy(),
            {"fold_count": 1, "feature_importance": [], "fold_windows": [
                {"fold": 1, "training_start": "2025-01-01T00:00:00+00:00"}
            ], "label_horizon_candles": 12},
        ),
    )
    req = ResolvedMLBacktestCreate(
        symbol="BTCUSDT", period_years=4,
        decision_timeframe="30min", trend_timeframe="4hour",
        initial_capital=1000.0, fee_rate=0.001, slippage_rate=0.0005,
        entry_threshold=0.9, exit_threshold=0.9, reinvestment=True,
    )
    snapshots = []
    result = engine.run_ml_backtest(
        req, default_parameters(), lambda value, stage: None,
        lambda: False, audit_writer=snapshots.append,
    )

    coverage = result["execution"]["history_coverage"]
    assert coverage["requested_years"] == 4
    assert coverage["raw_candles"] == 48
    assert coverage["oos_candles"] == 48
    assert coverage["requested_period_coverage_ratio"] < 0.01
    assert coverage["coverage_status"] == "PARTIAL"
    assert coverage["missing_intervals"] == 0
    assert result["execution"]["fold_windows"][0]["fold"] == 1
    assert len(fetched) == 2
    assert len(snapshots) == 1

    with ZipFile(BytesIO(snapshots[0])) as archive:
        names = set(archive.namelist())
        assert "market_history_decision.csv" in names
        assert "market_history_trend.csv" in names
        assert json.loads(archive.read("manifest.json"))["history_coverage"]["raw_candles"] == 48
        assert json.loads(archive.read("history_coverage.json"))["coverage_status"] == "PARTIAL"
