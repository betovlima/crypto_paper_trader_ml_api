"""Strict chronological BTC direction and return forecast research tests."""

from __future__ import annotations

from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from crypto_paper_trader_ml_api.direct_audit import build_direction_audit_archive
from crypto_paper_trader_ml_api.direct_forecast import (
    _metrics, create_direction_labels, walk_forward_direction,
)
from crypto_paper_trader_ml_api.features import build_features
from crypto_paper_trader_ml_api.schemas import MLBacktestCreate
from crypto_paper_trader_ml_api.service import _resolve_request

from .helpers import default_parameters
from .test_features import synthetic_candles


def test_direct_labels_are_future_close_returns_for_2_6_and_24_hours():
    candles = synthetic_candles(110)
    source = candles["close"].to_numpy()
    for horizon_hours, horizon_bars in ((2, 4), (6, 12), (24, 48)):
        labelled = create_direction_labels(candles, "30min", horizon_hours)
        assert len(labelled) == len(candles)
        assert labelled["forecast_horizon_candles"].iloc[0] == horizon_bars
        expected = source[horizon_bars] / source[0] - 1
        assert labelled.loc[0, "forecast_realized_return"] == pytest.approx(expected)
        assert labelled.loc[0, "forecast_realized_up"] == float(expected > 0)
        assert labelled.loc[0, "forecast_future_at_utc"] == (
            candles.loc[horizon_bars, "timestamp"] + pd.Timedelta(minutes=30)
        )
        assert labelled["forecast_realized_return"].tail(horizon_bars).isna().all()
        assert labelled["forecast_realized_up"].tail(horizon_bars).isna().all()
        assert labelled["forecast_future_at_utc"].tail(horizon_bars).isna().all()


def test_no_direct_forecast_labels_bridge_a_documented_trading_halt():
    source = synthetic_candles(200).drop(index=[55, 56]).reset_index(drop=True)
    source["market_segment"] = (source["timestamp"].diff().gt(
        pd.Timedelta(minutes=30)
    )).cumsum()
    labelled = create_direction_labels(source, "30min", 6)
    assert labelled.loc[40, "market_segment"] == 0
    assert labelled.loc[55, "market_segment"] == 1
    assert pd.isna(labelled.loc[50, "forecast_realized_return"])
    assert pd.isna(labelled.loc[50, "forecast_realized_up"])
    assert pd.isna(labelled.loc[50, "forecast_future_at_utc"])
    assert pd.notna(labelled.loc[20, "forecast_realized_return"])
    assert pd.notna(labelled.loc[80, "forecast_realized_return"])


def test_metrics_compare_against_train_only_probability_and_zero_return_baselines():
    examples = pd.DataFrame({
        "forecast_realized_up": [1, 0, 1, 0],
        "forecast_p_up": [0.8, 0.2, 0.7, 0.3],
        "forecast_baseline_p_up": [0.5] * 4,
        "forecast_realized_return": [0.03, -0.02, 0.01, -0.01],
        "forecast_predicted_return": [0.02, -0.01, 0.01, -0.02],
        "forecast_baseline_mean_return": [0.001] * 4,
    })
    metrics = _metrics(examples)
    assert metrics["rows"] == 4
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["direction_accuracy"] == pytest.approx(1.0)
    assert metrics["brier_skill_score"] > 0
    assert metrics["return_mae"] < metrics["zero_return_mae"]
    assert metrics["return_mae_skill_vs_zero"] > 0
    assert metrics["baseline_direction_accuracy"] == pytest.approx(0.5)


def test_24hour_walk_forward_uses_separate_purged_train_early_stop_calibration_and_test():
    data = synthetic_candles(2400)
    params = default_parameters(**{
        "xgboost.n_estimators": 8,
        "xgboost.early_stopping_rounds": 3,
        "xgboost.n_jobs": 1,
        "forecast.maximum_folds": 2,
        "walk_forward.minimum_usable_candles": 300,
        "walk_forward.initial_train_minimum": 300,
        "walk_forward.validation_minimum": 40,
        "walk_forward.minimum_test_size": 40,
    })
    trend = data.iloc[::8].reset_index(drop=True)
    featured = build_features(
        data, params, trend,
        decision_timeframe="30min", trend_timeframe="4hour",
    )
    forecast, metadata = walk_forward_direction(
        featured, params, timeframe="30min", horizon_hours=24,
        cancelled=lambda: False,
    )
    assert not forecast.empty
    assert forecast["forecast_fold"].nunique() == 2
    assert forecast["forecast_p_up"].between(0, 1).all()
    assert np.isfinite(forecast["forecast_predicted_return"]).all()
    assert len(metadata["fold_windows"]) == 2
    for fold in metadata["fold_windows"]:
        assert fold["purge_candles_at_each_boundary"] == 48
        for earlier, later in (
            ("training", "early_stopping"),
            ("early_stopping", "calibration"),
            ("calibration", "test"),
        ):
            assert pd.Timestamp(fold[earlier + "_last_label_at"]) < pd.Timestamp(
                fold[later + "_start"]
            )
        assert fold["training_rows"] > fold["early_stopping_rows"] > 0
        assert fold["calibration_rows"] >= 40
        assert fold["test_rows"] >= 40
    assert (forecast["forecast_future_at_utc"] > forecast["timestamp"]).all()


def test_direction_audit_keeps_every_oos_prediction_and_no_trading_claim():
    raw = synthetic_candles(80)
    forecast = create_direction_labels(raw, "30min", 2).head(30).copy()
    forecast["forecast_p_up"] = 0.55
    forecast["forecast_predicted_return"] = 0.004
    forecast["forecast_baseline_p_up"] = 0.48
    forecast["forecast_baseline_mean_return"] = 0.001
    forecast["forecast_fold"] = 1
    params = default_parameters(**{"forecast.horizons_hours": [2]})
    request = _resolve_request(
        MLBacktestCreate(
            symbol="BTCUSDT", experiment_type="BTC_DIRECTION_RETURN",
            history_source="BINANCE_SPOT_ARCHIVE",
            decision_timeframe="30min", trend_timeframe="4hour",
        ), params,
    )
    result = {
        "forecast": {"horizons": [{"horizon_hours": 2, "metrics": _metrics(forecast)}]},
        "execution": {
            "fold_windows": [{"fold": 1}], "history_coverage": {"raw_candles": 80},
            "market_interruptions": {},
        },
    }
    binary = build_direction_audit_archive(
        predictions=[forecast], decision_history=raw, trend_history=raw,
        request=request, parameters=params, result=result,
        source_archives={"30min": [{"sha256": "abc", "rows": 80}]},
    )
    with ZipFile(BytesIO(binary)) as archive:
        members = set(archive.namelist())
        assert {
            "manifest.json", "request.json", "parameters_snapshot.json",
            "forecast_metrics.json", "folds.json", "history_coverage.json",
            "market_interruptions.json", "forecast_predictions_oos.csv",
            "market_history_decision.csv", "market_history_trend.csv", "README.txt",
        } == members
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["research_only"] is True
        assert manifest["financial_trading_enabled"] is False
        assert manifest["horizons"][0]["predictions"] == len(forecast)
        assert manifest["history_source"] == "BINANCE_SPOT_ARCHIVE"
        predictions_csv = pd.read_csv(BytesIO(archive.read("forecast_predictions_oos.csv")))
        assert len(predictions_csv) == 30
        assert "forecast_realized_return" in predictions_csv


def test_old_target_stop_mode_remains_default_and_invalid_direct_symbols_are_rejected():
    params = default_parameters()
    unchanged = _resolve_request(
        MLBacktestCreate(symbol="BTCUSDT", experiment_type="TARGET_STOP"), params
    )
    assert unchanged.experiment_type == "TARGET_STOP"
    forecast = _resolve_request(MLBacktestCreate(
        symbol="BTCUSDT", experiment_type="BTC_DIRECTION_RETURN",
    ), params)
    assert forecast.experiment_type == "BTC_DIRECTION_RETURN"
    with pytest.raises(ValueError, match="supports BTCUSDT only"):
        _resolve_request(MLBacktestCreate(
            symbol="ETHUSDT", experiment_type="BTC_DIRECTION_RETURN",
        ), params)
    with pytest.raises(ValueError, match="exact multiple"):
        _resolve_request(MLBacktestCreate(
            symbol="BTCUSDT", experiment_type="BTC_DIRECTION_RETURN",
            decision_timeframe="4hour",
        ), params)


def test_full_direction_job_routes_to_research_engine_and_writes_audit(monkeypatch):
    from crypto_paper_trader_ml_api import direct_forecast

    raw = synthetic_candles(180)
    fetched_timeframes = []

    class FakeMarket:
        archive_manifest = {"30min": [{"sha256": "test-verified"}]}
        market_interruptions = {}

        def __init__(self, *_args):
            pass

        def fetch_history(self, *, symbol, timeframe, period_years, progress, cancelled):
            assert symbol == "BTCUSDT"
            assert timeframe in {"30min", "4hour"}
            fetched_timeframes.append(timeframe)
            if timeframe == "4hour":
                return raw.iloc[::8].reset_index(drop=True).copy()
            return raw.copy()

        def close(self):
            pass

    def fake_predict(featured, parameters, *, timeframe, horizon_hours, cancelled, progress=None):
        frame = create_direction_labels(featured, timeframe, horizon_hours)
        frame = frame.dropna(subset=["forecast_realized_return"]).head(30).copy()
        frame["forecast_p_up"] = 0.51
        frame["forecast_predicted_return"] = 0.001
        frame["forecast_baseline_p_up"] = 0.5
        frame["forecast_baseline_mean_return"] = 0.0
        frame["forecast_fold"] = 1
        return frame, {
            "horizon_candles": horizon_hours * 2,
            "fold_windows": [{"fold": 1, "horizon_hours": horizon_hours}],
            "feature_importance": {"direction": [], "return": []},
        }

    monkeypatch.setattr(direct_forecast, "MEXCHistoryClient", FakeMarket)
    monkeypatch.setattr(direct_forecast, "build_features", lambda decision, parameters, trend, **kwargs: decision.assign(market_segment=0))
    monkeypatch.setattr(direct_forecast, "walk_forward_direction", fake_predict)
    params = default_parameters(**{"forecast.horizons_hours": [2, 6, 24]})
    request = _resolve_request(MLBacktestCreate(
        experiment_type="BTC_DIRECTION_RETURN", symbol="BTCUSDT",
        decision_timeframe="30min", trend_timeframe="4hour",
    ), params)
    archives = []
    result = direct_forecast.run_direction_backtest(
        request, params, lambda _value, _stage: None,
        lambda: False, audit_writer=archives.append,
    )
    assert fetched_timeframes == ["30min", "4hour"]
    assert result["experiment_type"] == "BTC_DIRECTION_RETURN"
    assert result["summary"]["financial_trading_enabled"] is False
    assert result["trades"] == []
    assert result["equity_curve"] == []
    assert result["rotation_timeline"] == []
    assert [item["horizon_hours"] for item in result["forecast"]["horizons"]] == [2, 6, 24]
    assert len(archives) == 1
    with ZipFile(BytesIO(archives[0])) as audit:
        manifest = json.loads(audit.read("manifest.json"))
        assert manifest["research_only"] is True
        assert [item["predictions"] for item in manifest["horizons"]] == [30, 30, 30]
        all_rows = pd.read_csv(BytesIO(audit.read("forecast_predictions_oos.csv")))
        assert len(all_rows) == 90
        assert sorted(all_rows["forecast_horizon_hours"].unique().tolist()) == [2, 6, 24]


def test_service_dispatches_direct_jobs_without_touching_target_stop_engine(monkeypatch):
    from crypto_paper_trader_ml_api import service
    from crypto_paper_trader_ml_api.database import MLBacktestAudit, MLBacktestJob, SessionLocal
    from datetime import datetime, timezone
    from uuid import uuid4

    job_id = str(uuid4())
    params = default_parameters()
    request = _resolve_request(MLBacktestCreate(
        experiment_type="BTC_DIRECTION_RETURN", symbol="BTCUSDT",
    ), params)
    with SessionLocal.begin() as session:
        session.add(MLBacktestJob(
            id=job_id, status="QUEUED", stage="Queued", progress=0,
            symbol="BTCUSDT", decision_timeframe=request.decision_timeframe,
            trend_timeframe=request.trend_timeframe, period_years=request.period_years,
            initial_capital=request.initial_capital, request_json=request.model_dump_json(),
            settings_snapshot_json=json.dumps(params.snapshot()), result_json=None,
            error_message=None, created_at=datetime.now(timezone.utc),
        ))

    calls = []
    def fake_direct(req, parameters, progress, cancelled, audit_writer=None):
        calls.append(req.experiment_type)
        audit_writer(b"direction-only-zip")
        return {"experiment_type": "BTC_DIRECTION_RETURN", "forecast": {"horizons": []}}

    def blocked_legacy(*args, **kwargs):
        raise AssertionError("Legacy BTC/CASH engine must not run for direct forecasts")

    monkeypatch.setattr(service, "run_direction_backtest", fake_direct)
    monkeypatch.setattr(service, "run_ml_backtest", blocked_legacy)
    try:
        service._run_job(job_id)
        with SessionLocal() as session:
            job = session.get(MLBacktestJob, job_id)
            audit = session.get(MLBacktestAudit, job_id)
            assert job.status == "COMPLETED"
            assert audit.archive_zip == b"direction-only-zip"
            assert json.loads(job.result_json)["experiment_type"] == "BTC_DIRECTION_RETURN"
            assert calls == ["BTC_DIRECTION_RETURN"]
    finally:
        with SessionLocal.begin() as session:
            audit = session.get(MLBacktestAudit, job_id)
            if audit is not None:
                session.delete(audit)
                session.flush()
            job = session.get(MLBacktestJob, job_id)
            if job is not None:
                session.delete(job)
