"""The offline replay must keep historical source and fold configuration fixed."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import replay_btc_forecast  # noqa: E402

from crypto_paper_trader_ml_api.schemas import MLBacktestCreate  # noqa: E402
from crypto_paper_trader_ml_api.service import _resolve_request  # noqa: E402
from .helpers import default_parameters  # noqa: E402
from .test_features import synthetic_candles  # noqa: E402


def test_replay_uses_only_original_csv_market_and_keeps_fold_settings(tmp_path, monkeypatch):
    params = default_parameters()
    original_params = params.snapshot()
    original_params["xgboost.gamma"] = 0.1234
    original_params["forecast.horizons_hours"] = [2]
    original_params["forecast.maximum_folds"] = 2
    req = _resolve_request(
        MLBacktestCreate(
            experiment_type="BTC_DIRECTION_RETURN", symbol="BTCUSDT",
            history_source="BINANCE_SPOT_ARCHIVE",
            decision_timeframe="30min", trend_timeframe="4hour",
        ),
        params,
    )
    raw = synthetic_candles(64)
    trend = raw.iloc[::8].reset_index(drop=True)
    original = tmp_path / "v1.zip"
    with ZipFile(original, "w") as z:
        z.writestr("manifest.json", json.dumps({
            "kind": "BTC_DIRECTION_RETURN_OOS_AUDIT",
            "history_source": "BINANCE_SPOT_ARCHIVE",
            "symbol": "BTCUSDT", "decision_timeframe": "30min",
            "trend_timeframe": "4hour", "market_interruptions": {},
            "market_archive_checksums": {"30min": [{"sha256": "archived"}]},
        }))
        z.writestr("request.json", req.model_dump_json())
        z.writestr("parameters_snapshot.json", json.dumps(original_params))
        z.writestr("market_history_decision.csv", raw.to_csv(index=False))
        z.writestr("market_history_trend.csv", trend.to_csv(index=False))
        z.writestr("folds.json", json.dumps([{"fold": 1, "horizon_hours": 2}]))
        z.writestr("forecast_metrics.json", json.dumps([{
            "horizon_hours": 2,
            "metrics": {
                "rows": 1, "roc_auc": 0.5, "brier_skill_score": 0.0,
                "return_mae": 0.01, "return_mae_skill_vs_zero": 0.0,
                "return_rmse": 0.012,
            },
        }]))
        z.writestr("forecast_predictions_oos.csv",
            "forecast_horizon_hours,timestamp,forecast_future_at_utc,forecast_realized_return,forecast_realized_up\n"
            "2,2024-01-01T00:00:00Z,2024-01-01T02:30:00Z,0.01,1\n")

    captured = []

    def fake_runner(request, parameters, progress, cancelled, audit_writer, historical_client):
        captured.append((
            request, parameters, historical_client,
            historical_client.fetch_history(
                symbol="BTCUSDT", timeframe="30min",
                period_years=request.period_years,
            ),
        ))
        audit_writer(original.read_bytes())
        return {"forecast": {"horizons": [{"horizon_hours": 2}]}}

    monkeypatch.setattr(replay_btc_forecast, "run_direction_backtest", fake_runner)
    output = tmp_path / "v2.zip"
    result = replay_btc_forecast.replay_audit(original, output)
    assert result["controlled_comparison"]["verified_same_market_data"] is True
    assert output.is_file()
    assert len(captured) == 1
    _, runtime, client, candles = captured[0]
    assert runtime.floating("xgboost.gamma") == 0.1234
    assert runtime.list("forecast.horizons_hours") == [2]
    assert runtime.integer("forecast.maximum_folds") == 2
    assert runtime.floating("forecast.regression_target_scale") > 1
    assert client.archive_manifest["30min"][0]["sha256"] == "archived"
    # Both operands are pandas Series, not DataFrames. Compare the exact
    # chronological timestamp values, including their timezone and dtype.
    pd.testing.assert_series_equal(
        pd.to_datetime(candles["timestamp"], utc=True),
        pd.to_datetime(raw["timestamp"], utc=True),
        check_names=False,
    )
