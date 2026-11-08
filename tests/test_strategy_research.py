"""Persisted Strategy Research, reproducible tuning and final-audit workflow."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from crypto_paper_trader_ml_api.app import app
from crypto_paper_trader_ml_api.database import (
    MLBacktestAudit, MLBacktestJob, MLResearchStrategy, SessionLocal,
)
from crypto_paper_trader_ml_api import service, strategy_research
from crypto_paper_trader_ml_api.strategy_research import _candidate, _simulate, _next_candidate
from scipy.stats import qmc
from .test_archive_replay_ui import _make_archive
from crypto_paper_trader_ml_api.archive_replay import inspect_source_archive


def _insert_source() -> str:
    job_id = str(uuid4())
    blob = _make_archive()
    request, settings = inspect_source_archive(blob)
    with SessionLocal.begin() as session:
        session.add(MLBacktestJob(
            id=job_id, status="COMPLETED", stage="Completed", progress=100,
            symbol="BTCUSDT", decision_timeframe="30min",
            trend_timeframe="4hour", period_years=request.period_years,
            initial_capital=request.initial_capital,
            request_json=request.model_dump_json(),
            settings_snapshot_json=json.dumps(settings),
            created_at=datetime.now(timezone.utc),
        ))
        session.flush()
        session.add(MLBacktestAudit(job_id=job_id, archive_zip=blob))
    return job_id


def _cleanup(source_id: str):
    with SessionLocal.begin() as session:
        strategies = session.query(MLResearchStrategy).filter(
            MLResearchStrategy.source_job_id == source_id
        ).all()
        children = [row.execution_job_id for row in strategies if row.execution_job_id]
        for row in strategies:
            session.delete(row)
        session.flush()
        for child in children:
            session.query(MLBacktestAudit).filter(MLBacktestAudit.job_id == child).delete()
            session.query(MLBacktestJob).filter(MLBacktestJob.id == child).delete()
        session.query(MLBacktestAudit).filter(MLBacktestAudit.job_id == source_id).delete()
        session.query(MLBacktestJob).filter(MLBacktestJob.id == source_id).delete()


def test_candidate_respects_lightgbm_depth_constraints_and_is_reproducible():
    point = np.ones(len(strategy_research.SPACE))
    winner = _candidate(point)
    assert winner["lightgbm.max_depth"] == 6
    assert winner["lightgbm.num_leaves"] <= 2 ** winner["lightgbm.max_depth"]
    assert winner["forecast.regression_num_leaves"] <= 16
    lhs = qmc.LatinHypercube(d=len(strategy_research.SPACE), seed=5)
    rng = np.random.default_rng(5)
    a = _next_candidate(0, "QUICK", [], [], lhs, rng)
    assert len(a) == len(strategy_research.SPACE)
    assert np.all((a >= 0) & (a <= 1))


def test_caro_adapted_gp_returns_bounded_candidate_with_risk_metrics():
    rng = np.random.default_rng(42)
    points = [rng.uniform(size=len(strategy_research.SPACE)) for _ in range(10)]
    rows = []
    for i in range(len(points)):
        rows.append({
            "status": "COMPLETED",
            "metrics": {
                "score": 0.03 * i,
                "capital_final": 10000 + 50 * i,
                "sharpe": 0.2 + .04 * i,
                "max_drawdown": -0.13 + 0.002 * i,
                "worst_fold_return": -0.05 + 0.005 * i,
            },
        })
    point = _next_candidate(
        10, "BAYESIAN", points, [v["metrics"]["score"] for v in rows],
        qmc.LatinHypercube(d=len(strategy_research.SPACE), seed=42),
        np.random.default_rng(42), rows,
    )
    assert len(point) == len(strategy_research.SPACE)
    assert np.isfinite(point).all()
    assert ((point >= 0) & (point <= 1)).all()


def test_validation_strategy_executes_only_at_next_open():
    data = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=4, freq="30min", tz="UTC"),
        "open": [100.0, 110.0, 110.0, 110.0],
        "close": [100.0, 110.0, 110.0, 110.0],
    })
    metrics = _simulate(
        data, np.array([0.9, 0.4, 0.4, 0.4]),
        np.array([0.05, -0.05, -0.05, -0.05]),
        fee=0, slippage=0, timeframe="30min",
    )
    assert metrics["buys"] == 1
    assert metrics["sells"] == 1
    # Buying at the forecast candle close (100) would make 10%; the correct
    # fill occurs at the following candle OPEN (110) and breaks even.
    assert metrics["capital_final"] == pytest.approx(10000)
    assert metrics["worst_fold_return"] == pytest.approx(0)
    assert metrics["sharpe"] == pytest.approx(0)


def test_single_api_supports_strategy_creation_and_bounded_tuning(monkeypatch):
    source = _insert_source()
    class RecordedExecutor:
        pending = []
        def submit(self, fn, *args):
            self.pending.append((fn, args))
    worker = RecordedExecutor()
    monkeypatch.setattr(service, "_get_executor", lambda _params: worker)
    try:
        with TestClient(app) as client:
            created = client.post("/v1/ml-strategies", json={
                "name": "BTC retorno de 2 horas",
                "source_job_id": source, "horizon_hours": 2,
            })
            assert created.status_code == 201, created.text
            identifier = created.json()["id"]
            assert created.json()["status"] == "DRAFT"
            invalid = client.post(f"/v1/ml-strategies/{identifier}/optimize",
                                  json={"mode": "QUICK", "trials": 2})
            assert invalid.status_code == 422
            started = client.post(f"/v1/ml-strategies/{identifier}/optimize",
                                  json={"mode": "QUICK", "trials": 4})
            assert started.status_code == 202
            assert started.json()["status"] == "RUNNING"
            assert len(worker.pending) == 1
            assert client.get(f"/v1/ml-strategies/{identifier}").status_code == 200
            assert any(x["id"] == identifier for x in client.get("/v1/ml-strategies").json())
            report = client.get(f"/v1/ml-strategies/{identifier}/report.zip")
            assert report.status_code == 200
            assert report.content.startswith(b"PK")
            duplicate = client.post(f"/v1/ml-strategies/{identifier}/optimize",
                                    json={"mode": "BAYESIAN", "trials": 24})
            assert duplicate.status_code == 409
    finally:
        _cleanup(source)


def test_strategy_final_execution_replays_original_archive_with_tuned_lightgbm(monkeypatch):
    source = _insert_source()
    worker = type("Worker", (), {"submit": lambda *args: None})()
    monkeypatch.setattr(service, "_get_executor", lambda _params: worker)
    try:
        candidate = strategy_research.create_strategy(
            name="Experimento LightGBM", source_job_id=source, horizon_hours=2,
        )
        strategy_id = candidate["id"]
        settings = {
            "lightgbm.max_depth": 4, "lightgbm.num_leaves": 12,
            "lightgbm.learning_rate": 0.025,
            "forecast.regression_learning_rate": 0.035,
        }
        with SessionLocal.begin() as session:
            row = session.get(MLResearchStrategy, strategy_id)
            row.status = "OPTIMIZED"
            row.champion_json = json.dumps({"parameters": settings, "metrics": {"score": 0.1}})
        created = strategy_research.execute_strategy(strategy_id)
        assert created["status"] == "QUEUED"
        assert created["replay_parent_job_id"] == source
        with SessionLocal() as session:
            job = session.get(MLBacktestJob, created["id"])
            snapshot = json.loads(job.settings_snapshot_json)
            assert snapshot["lightgbm.learning_rate"] == 0.025
            assert snapshot["lightgbm.max_depth"] == 4
        assert strategy_research.get_strategy(strategy_id)["execution_job_id"] == created["id"]
    finally:
        _cleanup(source)
