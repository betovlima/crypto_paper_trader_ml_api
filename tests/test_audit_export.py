"""Audit ZIP includes unsampled OOS evidence and is downloadable by job ID."""

import csv
from datetime import datetime, timezone
from io import BytesIO, StringIO
import json
from uuid import uuid4
from zipfile import ZipFile

import pandas as pd
from fastapi.testclient import TestClient

from crypto_paper_trader_ml_api.app import app
from crypto_paper_trader_ml_api.audit_export import build_audit_archive
from crypto_paper_trader_ml_api.database import (
    MLBacktestAudit, MLBacktestJob, SessionLocal,
)
from crypto_paper_trader_ml_api.engine import simulate_portfolio
from crypto_paper_trader_ml_api.schemas import ResolvedMLBacktestCreate

from .helpers import default_parameters


def _request():
    return ResolvedMLBacktestCreate(
        symbol="BTCUSDT",
        period_years=1,
        decision_timeframe="30min",
        trend_timeframe="4hour",
        initial_capital=1000.0,
        fee_rate=0.001,
        slippage_rate=0.0005,
        entry_threshold=0.65,
        exit_threshold=0.60,
        reinvestment=True,
    )


def _frame():
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="30min", tz="UTC"),
        "open": [100.0] * 5,
        "high": [100.1] * 5,
        "low": [99.9] * 5,
        "close": [100.0] * 5,
        "atr_14": [1.0] * 5,
        "ema_21": [99.5] * 5,
        "ema_50": [99.0] * 5,
        "trend_ema_gap": [0.01] * 5,
        "entry_probability": [0.9, 0.1, 0.1, 0.1, 0.1],
        "exit_probability": [0.1, 0.1, 0.1, 0.1, 0.1],
        "entry_label": [1, 0, 0, 0, 1],
        "exit_label": [0, 0, 1, 0, 0],
        "walk_forward_fold": [1] * 5,
    })


def _parameters():
    return default_parameters(**{
        "simulation.initial_stop_atr": 100.0,
        "simulation.initial_target_atr": 100.0,
        "simulation.trailing_stop_atr": 100.0,
        "simulation.minimum_trend_gap": 0.0,
    })


def _archive(frame=None, parameters=None):
    frame = frame if frame is not None else _frame()
    parameters = parameters if parameters is not None else _parameters()
    simulation = simulate_portfolio(frame, _request(), parameters)
    content = build_audit_archive(
        predictions=frame,
        simulation=simulation,
        request=_request(),
        parameters=parameters,
        result={
            "summary": {"capital_final": simulation["capital_final"]},
            "trades": [],
            "equity_curve": [],
            "rotation_timeline": [],
        },
        model_metadata={"fold_count": 1, "feature_importance": []},
    )
    return content, simulation


def _rows(zipfile, filename):
    return list(csv.DictReader(
        StringIO(zipfile.read(filename).decode("utf-8-sig"))
    ))


def test_full_audit_preserves_every_prediction_candle_and_execution():
    archive, simulation = _archive()
    with ZipFile(BytesIO(archive)) as z:
        names = set(z.namelist())
        assert {
            "manifest.json", "request.json", "parameters_snapshot.json",
            "model_metadata.json", "report.json", "predictions_oos.csv",
            "positions_full.csv", "executions_full.csv", "trades_full.csv",
            "README.txt", "folds.json", "history_coverage.json",
            "market_interruptions.json",
        } == names
        manifest = json.loads(z.read("manifest.json"))
        assert manifest["scope"] == "OUT_OF_SAMPLE"
        assert manifest["unsampled"] is True
        assert manifest["candles"] == 5
        assert manifest["schema_version"] == 2
        assert manifest["raw_history_included"] is False
        assert json.loads(z.read("market_interruptions.json")) == {}
        prediction_rows = _rows(z, "predictions_oos.csv")
        position_rows = _rows(z, "positions_full.csv")
        action_rows = _rows(z, "executions_full.csv")
        trade_rows = _rows(z, "trades_full.csv")
        assert len(prediction_rows) == len(position_rows) == len(_frame())
        assert len(action_rows) == sum(
            len(row["actions"]) for row in simulation["rotation_rows"]
        )
        assert len(trade_rows) == len(simulation["trades"])
        assert {"entry_probability", "exit_probability", "walk_forward_fold"}.issubset(prediction_rows[0])
        assert position_rows[0]["position_at_candle_close"] == "CASH"
        assert position_rows[1]["position_at_candle_close"] == "BTC"
        assert position_rows[-1]["position_at_candle_close"] == "CASH"
        assert action_rows[-1]["reason"] == "END_OF_TEST"
        assert action_rows[-1]["execution_timing"] == "CANDLE_CLOSE"
        assert json.loads(z.read("parameters_snapshot.json"))["simulation.minimum_trend_gap"] == 0


def test_same_bar_purchase_and_protective_sale_are_both_included():
    frame = _frame()
    frame.loc[1, "low"] = 90.0
    params = default_parameters(**{
        "simulation.initial_stop_atr": 0.5,
        "simulation.initial_target_atr": 100.0,
        "simulation.trailing_stop_atr": 100.0,
        "simulation.minimum_trend_gap": 0.0,
    })
    archive, simulation = _archive(frame, params)
    with ZipFile(BytesIO(archive)) as z:
        actions = _rows(z, "executions_full.csv")
        assert [a["action"] for a in actions[:2]] == ["BUY", "SELL"]
        assert actions[0]["timestamp_utc"] == actions[1]["timestamp_utc"]
        assert actions[1]["reason"] == "PROTECTIVE_STOP"
        assert actions[1]["execution_timing"] == "INTRABAR"
        positions = _rows(z, "positions_full.csv")
        assert positions[1]["position_at_candle_close"] == "CASH"
        assert simulation["rotation_rows"][1]["position"] == "CASH"


def _persist_job(*, status: str, archive: bytes | None = None) -> str:
    job_id = str(uuid4())
    with SessionLocal.begin() as session:
        session.add(MLBacktestJob(
            id=job_id,
            status=status,
            stage=status.title(),
            progress=100 if status == "COMPLETED" else 20,
            symbol="BTCUSDT",
            decision_timeframe="30min",
            trend_timeframe="4hour",
            period_years=1,
            initial_capital=1000.0,
            error_message=None,
            created_at=datetime.now(timezone.utc),
            request_json="{}",
            settings_snapshot_json="{}",
            result_json="{}" if status == "COMPLETED" else None,
        ))
        if archive is not None:
            session.flush()
            session.add(MLBacktestAudit(job_id=job_id, archive_zip=archive))
    return job_id


def _delete_job(job_id):
    with SessionLocal.begin() as session:
        audit = session.get(MLBacktestAudit, job_id)
        if audit:
            session.delete(audit)
            session.flush()
        job = session.get(MLBacktestJob, job_id)
        if job:
            session.delete(job)


def test_download_endpoint_returns_exact_archive_and_indicates_availability():
    content, _ = _archive()
    job_id = _persist_job(status="COMPLETED", archive=content)
    try:
        with TestClient(app) as client:
            detail = client.get(f"/v1/ml-backtests/{job_id}")
            response = client.get(f"/v1/ml-backtests/{job_id}/audit.zip")
        assert detail.status_code == 200
        assert detail.json()["audit_available"] is True
        assert response.status_code == 200
        assert response.content == content
        assert response.headers["content-type"].startswith("application/zip")
        assert f"btc-forecast-audit-{job_id}.zip" in response.headers["content-disposition"]
        with ZipFile(BytesIO(response.content)) as z:
            assert json.loads(z.read("manifest.json"))["candles"] == 5
    finally:
        _delete_job(job_id)


def test_legacy_completed_run_cannot_claim_a_complete_export():
    job_id = _persist_job(status="COMPLETED")
    try:
        with TestClient(app) as client:
            detail = client.get(f"/v1/ml-backtests/{job_id}")
            response = client.get(f"/v1/ml-backtests/{job_id}/audit.zip")
        assert detail.json()["audit_available"] is False
        assert response.status_code == 409
        assert "Run a new backtest" in response.json()["detail"]
    finally:
        _delete_job(job_id)


def test_incomplete_and_unknown_runs_cannot_be_exported():
    job_id = _persist_job(status="RUNNING")
    try:
        with TestClient(app) as client:
            unfinished = client.get(f"/v1/ml-backtests/{job_id}/audit.zip")
            missing = client.get(f"/v1/ml-backtests/{uuid4()}/audit.zip")
        assert unfinished.status_code == 409
        assert missing.status_code == 404
    finally:
        _delete_job(job_id)


def test_worker_persists_audit_at_the_same_time_as_finished_results(monkeypatch):
    """No successful job should be marked complete without its full ZIP."""
    from crypto_paper_trader_ml_api import service

    content, _ = _archive()
    job_id = _persist_job(status="QUEUED")
    with SessionLocal.begin() as session:
        job = session.get(MLBacktestJob, job_id)
        job.request_json = _request().model_dump_json()
        job.settings_snapshot_json = json.dumps(_parameters().snapshot())

    def fake_backtest(request, parameters, progress, cancelled, audit_writer=None):
        assert audit_writer is not None
        assert request.symbol == "BTCUSDT"
        audit_writer(content)
        return {"summary": {"capital_final": 1234.0}}

    monkeypatch.setattr(service, "run_ml_backtest", fake_backtest)
    try:
        service._run_job(job_id)
        with SessionLocal() as session:
            job = session.get(MLBacktestJob, job_id)
            saved = session.get(MLBacktestAudit, job_id)
            assert job.status == "COMPLETED"
            assert saved is not None
            assert saved.archive_zip == content
            assert json.loads(job.result_json)["summary"]["capital_final"] == 1234.0
    finally:
        _delete_job(job_id)


def test_worker_rejects_completed_status_if_audit_is_missing(monkeypatch):
    from crypto_paper_trader_ml_api import service

    job_id = _persist_job(status="QUEUED")
    with SessionLocal.begin() as session:
        job = session.get(MLBacktestJob, job_id)
        job.request_json = _request().model_dump_json()
        job.settings_snapshot_json = json.dumps(_parameters().snapshot())

    def missing_audit(request, parameters, progress, cancelled, audit_writer=None):
        return {"summary": {"capital_final": 1000.0}}

    monkeypatch.setattr(service, "run_ml_backtest", missing_audit)
    try:
        service._run_job(job_id)
        with SessionLocal() as session:
            job = session.get(MLBacktestJob, job_id)
            assert job.status == "FAILED"
            assert job.result_json is None
            assert session.get(MLBacktestAudit, job_id) is None
            assert "No complete audit archive" in job.error_message
    finally:
        _delete_job(job_id)


def test_audit_can_include_complete_training_history_and_fold_coverage():
    frame = _frame()
    parameters = _parameters()
    simulation = simulate_portfolio(frame, _request(), parameters)
    history = pd.concat([frame.iloc[:2], frame], ignore_index=True)
    metadata = {
        "fold_count": 1, "feature_importance": [],
        "label_horizon_candles": 12,
        "fold_windows": [{"fold": 1, "purge_candles_at_each_boundary": 12}],
        "history_coverage": {"raw_candles": 7, "coverage_status": "PARTIAL"},
    }
    content = build_audit_archive(
        predictions=frame, simulation=simulation, request=_request(),
        parameters=parameters, result={"summary": {}},
        model_metadata=metadata, decision_history=history, trend_history=history,
    )
    with ZipFile(BytesIO(content)) as z:
        manifest = json.loads(z.read("manifest.json"))
        assert manifest["raw_history_included"] is True
        assert len(_rows(z, "market_history_decision.csv")) == 7
        assert len(_rows(z, "market_history_trend.csv")) == 7
        assert json.loads(z.read("folds.json"))[0]["purge_candles_at_each_boundary"] == 12
        assert json.loads(z.read("history_coverage.json"))["coverage_status"] == "PARTIAL"
