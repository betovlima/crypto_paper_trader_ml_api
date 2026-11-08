"""End-to-end DB-based UI replay tests: no external ZIP paths or shell scripts."""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
from uuid import uuid4
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from crypto_paper_trader_ml_api.archive_replay import (
    ArchivedMarketClient, compare_archives, inspect_source_archive,
)
from crypto_paper_trader_ml_api.database import (
    MLBacktestAudit, MLBacktestJob, SessionLocal,
)
from crypto_paper_trader_ml_api.schemas import MLBacktestCreate
from crypto_paper_trader_ml_api.service import _resolve_request
from crypto_paper_trader_ml_api import service
from .helpers import default_parameters
from .test_features import synthetic_candles


def _make_archive(*, predicted: float = 0.005, market_suffix: str = "") -> bytes:
    request = _resolve_request(
        MLBacktestCreate(
            experiment_type="BTC_DIRECTION_RETURN", symbol="BTCUSDT",
            history_source="BINANCE_SPOT_ARCHIVE",
            decision_timeframe="30min", trend_timeframe="4hour",
        ), default_parameters(),
    )
    raw = synthetic_candles(70)
    trend = raw.iloc[::8].reset_index(drop=True)
    if market_suffix:
        raw.loc[0, "close"] += 1
    prediction = pd.DataFrame([{
        "forecast_horizon_hours": 2, "forecast_fold": 1,
        "timestamp": "2024-01-01T00:00:00.000000Z",
        "forecast_future_at_utc": "2024-01-01T02:30:00.000000Z",
        "forecast_realized_return": 0.02,
        "forecast_realized_up": 1.0,
        "forecast_predicted_return": predicted,
        "forecast_p_up": 0.6,
    }])
    manifest = {
        "kind": "BTC_DIRECTION_RETURN_OOS_AUDIT",
        "symbol": "BTCUSDT", "history_source": "BINANCE_SPOT_ARCHIVE",
        "decision_timeframe": "30min", "trend_timeframe": "4hour",
        "market_archive_checksums": {"30min": [{"sha256": "verified"}]},
        "market_interruptions": {},
    }
    metrics = [{
        "horizon_hours": 2,
        "metrics": {
            "rows": 1, "roc_auc": 0.52 + predicted,
            "brier_skill_score": 0.01, "return_mae": 0.01,
            "return_mae_skill_vs_zero": 0.001, "return_rmse": 0.011,
        },
    }]
    folds = [{
        "fold": 1, "horizon_hours": 2,
        "training_start": "2023-01-01T00:00:00Z",
        "test_start": "2024-01-01T00:00:00Z",
    }]
    out = BytesIO()
    with ZipFile(out, "w") as zip_file:
        content = {
            "manifest.json": json.dumps(manifest),
            "request.json": request.model_dump_json(),
            "parameters_snapshot.json": json.dumps(default_parameters().snapshot()),
            "forecast_metrics.json": json.dumps(metrics),
            "folds.json": json.dumps(folds),
            "market_history_decision.csv": raw.to_csv(index=False),
            "market_history_trend.csv": trend.to_csv(index=False),
            "forecast_predictions_oos.csv": prediction.to_csv(index=False),
        }
        for name, value in content.items():
            zip_file.writestr(name, value)
    return out.getvalue()


def test_archived_market_reuses_saved_btc_candles_not_exchange():
    blob = _make_archive()
    request, settings = inspect_source_archive(blob)
    client = ArchivedMarketClient(blob, request)
    assert request.experiment_type == "BTC_DIRECTION_RETURN"
    assert settings["forecast.maximum_folds"] >= 2
    decision = client.fetch_history(
        symbol=request.symbol, timeframe="30min",
        period_years=request.period_years,
    )
    trend = client.fetch_history(
        symbol=request.symbol, timeframe="4hour",
        period_years=request.period_years,
    )
    assert len(decision) == 70
    assert len(trend) == 9
    assert client.archive_manifest["30min"][0]["sha256"] == "verified"
    with pytest.raises(ValueError, match="Timeframe"):
        client.fetch_history(
            symbol=request.symbol, timeframe="1hour",
            period_years=request.period_years,
        )


def test_replay_comparator_reports_predictive_deltas_only_when_audits_match():
    old, new = _make_archive(predicted=0.005), _make_archive(predicted=0.008)
    comparison = compare_archives(old, new)
    assert comparison["status"] == "VERIFIED"
    assert comparison["same_market_history"]
    assert comparison["same_fold_windows"]
    assert comparison["horizons"][0]["fold_return_diversity"][0] == {
        "fold": 1, "v1_distinct_returns": 1, "v2_distinct_returns": 1,
    }
    delta = comparison["horizons"][0]["metrics"]["roc_auc"]["delta"]
    assert delta == pytest.approx(0.003)
    with pytest.raises(ValueError, match="Candles históricos diferentes"):
        compare_archives(old, _make_archive(predicted=0.008, market_suffix="different"))


def test_ui_replay_creates_db_job_from_original_snapshot_and_persists_comparison(monkeypatch):
    original_id = str(uuid4())
    old_archive = _make_archive(predicted=0.005)
    old_request, old_params = inspect_source_archive(old_archive)
    class RecordedExecutor:
        pending = []

        def submit(self, fn, *args):
            self.pending.append((fn, args))
            return None

    executor = RecordedExecutor()
    monkeypatch.setattr(service, "_get_executor", lambda _parameters: executor)
    with SessionLocal.begin() as session:
        session.add(MLBacktestJob(
            id=original_id, status="COMPLETED", stage="Completed", progress=100,
            symbol="BTCUSDT", decision_timeframe="30min",
            trend_timeframe="4hour", period_years=old_request.period_years,
            initial_capital=old_request.initial_capital,
            request_json=old_request.model_dump_json(),
            settings_snapshot_json=json.dumps(old_params),
            result_json=json.dumps({"experiment_type": "BTC_DIRECTION_RETURN"}),
            created_at=datetime.now(timezone.utc),
        ))
        session.add(MLBacktestAudit(job_id=original_id, archive_zip=old_archive))

    new_id = None
    try:
        job = service.create_replay_job(original_id)
        new_id = job["id"]
        assert job["replay_parent_job_id"] == original_id
        assert job["status"] == "QUEUED"
        assert len(executor.pending) == 1
        with SessionLocal() as session:
            replay = session.get(MLBacktestJob, new_id)
            settings = json.loads(replay.settings_snapshot_json)
            assert settings["xgboost.gamma"] == old_params["xgboost.gamma"]
            assert "forecast.regression_target_scale" in settings
            assert replay.replay_parent_job_id == original_id

        def fake_model(request, parameters, progress, cancelled, audit_writer, historical_client):
            assert request.experiment_type == "BTC_DIRECTION_RETURN"
            assert historical_client.replay_source_sha256
            assert len(historical_client.fetch_history(
                symbol="BTCUSDT", timeframe="30min",
                period_years=request.period_years,
            )) == 70
            audit_writer(_make_archive(predicted=0.008))
            return {
                "experiment_type": "BTC_DIRECTION_RETURN",
                "execution": {"history_source": "BINANCE_SPOT_ARCHIVE"},
                "forecast": {"horizons": []}, "trades": [],
            }

        monkeypatch.setattr(service, "run_direction_backtest", fake_model)
        service._run_job(new_id)
        detail = service.get_job(new_id, include_result=True)
        assert detail["status"] == "COMPLETED"
        assert detail["audit_available"] is True
        assert detail["result"]["replay_comparison"]["status"] == "VERIFIED"
        assert detail["result"]["replay_comparison"]["source_job_id"] == original_id
    finally:
        with SessionLocal.begin() as session:
            for job_id in (new_id, original_id):
                if not job_id:
                    continue
                audit = session.get(MLBacktestAudit, job_id)
                if audit:
                    session.delete(audit)
                    session.flush()
                job = session.get(MLBacktestJob, job_id)
                if job:
                    session.delete(job)


def test_replay_rejects_missing_or_non_replayable_original_job(monkeypatch):
    class NotCalled:
        def submit(self, *args):
            raise AssertionError("No background job may be scheduled")

    monkeypatch.setattr(service, "_get_executor", lambda _p: NotCalled())
    with pytest.raises(LookupError, match="não encontrada"):
        service.create_replay_job(str(uuid4()))
    original_id = str(uuid4())
    request, _ = inspect_source_archive(_make_archive())
    with SessionLocal.begin() as session:
        session.add(MLBacktestJob(
            id=original_id, status="COMPLETED", stage="Completed", progress=100,
            symbol="BTCUSDT", decision_timeframe="30min",
            trend_timeframe="4hour", period_years=request.period_years,
            initial_capital=request.initial_capital,
            request_json=request.model_dump_json(),
            settings_snapshot_json=json.dumps(default_parameters().snapshot()),
            created_at=datetime.now(timezone.utc),
        ))
    try:
        with pytest.raises(ValueError, match="não está armazenado"):
            service.create_replay_job(original_id)
    finally:
        with SessionLocal.begin() as session:
            job = session.get(MLBacktestJob, original_id)
            if job:
                session.delete(job)
