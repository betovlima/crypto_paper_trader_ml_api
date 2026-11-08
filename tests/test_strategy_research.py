"""Persisted Strategy Research, reproducible tuning and final-audit workflow."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4
from io import BytesIO
from zipfile import ZipFile

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


def _give_source_three_horizons(source_id: str) -> None:
    """Existing archived 30-minute candles, with a complete 2h/6h/24h fold manifest."""
    with SessionLocal() as session:
        original = session.get(MLBacktestAudit, source_id).archive_zip
    with ZipFile(BytesIO(original), "r") as archive:
        base_folds = json.loads(archive.read("folds.json"))
        merged = [
            {**base_folds[0], "horizon_hours": horizon}
            for horizon in (2, 6, 24)
        ]
        new_blob = BytesIO()
        with ZipFile(new_blob, "w") as out:
            for name in archive.namelist():
                out.writestr(
                    name,
                    json.dumps(merged).encode() if name == "folds.json"
                    else archive.read(name),
                )
    with SessionLocal.begin() as session:
        session.get(MLBacktestAudit, source_id).archive_zip = new_blob.getvalue()


def test_new_strategy_uses_all_horizons_and_rejects_incomplete_archive():
    source = _insert_source()
    try:
        with pytest.raises(ValueError, match="folds OOS"):
            strategy_research.create_strategy(
                name="Todos os horizontes", source_job_id=source
            )
        _give_source_three_horizons(source)
        row = strategy_research.create_strategy(
            name="Todos os horizontes", source_job_id=source
        )
        assert row["horizon_hours"] == 0
        assert row["horizons_hours"] == [2, 6, 24]
    finally:
        _cleanup(source)


def test_api_defaults_to_joint_research_without_horizon_field():
    source = _insert_source()
    _give_source_three_horizons(source)
    try:
        with TestClient(app) as client:
            response = client.post("/v1/ml-strategies", json={
                "name": "BTC LightGBM 2h+6h+24h",
                "source_job_id": source,
            })
            assert response.status_code == 201, response.text
            assert response.json()["horizon_hours"] == 0
            assert response.json()["horizons_hours"] == [2, 6, 24]
            selected = client.get("/v1/ml-strategies/" + response.json()["id"])
            assert selected.status_code == 200
            assert selected.json()["horizons_hours"] == [2, 6, 24]
    finally:
        _cleanup(source)


def test_one_optimization_campaign_evaluates_all_horizons_without_oos_leakage(monkeypatch):
    source = _insert_source()
    _give_source_three_horizons(source)
    seen = []
    try:
        strategy = strategy_research.create_strategy(
            name="Otimização conjunta", source_job_id=source,
        )
        strategy_id = strategy["id"]
        with SessionLocal.begin() as session:
            row = session.get(MLResearchStrategy, strategy_id)
            row.status = "RUNNING"
            row.mode = "QUICK"
            row.trials_target = 4
        test_start = pd.Timestamp("2024-02-01T00:00:00Z")
        dataset = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="30min", tz="UTC"),
        })

        def fake_prepare(_archive, horizon, _parameters, cutoff):
            assert cutoff == test_start
            seen.append(("prepare", horizon))
            return dataset, dataset, dataset, dataset, ["signal"], horizon, cutoff

        def fake_score(prepared, candidate, parameters):
            horizon = prepared[5]
            seen.append(("score", horizon))
            return {
                "score": 0.02 + horizon / 1000,
                "capital_final": 10000 + horizon * 2,
                "return_total": horizon * 2 / 10000,
                "max_drawdown": -0.1,
                "worst_fold_return": -0.03,
                "sharpe": 0.5,
                "brier": 0.2,
                "mae": 0.005,
            }

        monkeypatch.setattr(strategy_research, "_prepare", fake_prepare)
        monkeypatch.setattr(strategy_research, "_score", fake_score)
        # The synthetic archive first OOS starts at 2024-01-01, unlike this
        # artificial test period. Use the archived first boundary as ground truth.
        with SessionLocal() as session:
            stored = session.get(MLBacktestAudit, source)
            with ZipFile(BytesIO(stored.archive_zip)) as z:
                first = pd.Timestamp(json.loads(z.read("folds.json"))[0]["test_start"])
        test_start = first
        strategy_research._run_optimization(strategy_id)
        result = strategy_research.get_strategy(strategy_id)
        assert result["status"] == "OPTIMIZED", result["error_message"]
        assert len(result["trials"]) == 4
        assert [h for typ, h in seen if typ == "prepare"] == [2, 6, 24]
        assert len([h for typ, h in seen if typ == "score"]) == 12
        best = result["champion"]["metrics"]
        assert sorted(best["per_horizon"].keys()) == ["2", "24", "6"]
        assert best["score_method"] == "0.70_mean_0.30_worst"
        assert result["summary"]["horizons_hours"] == [2, 6, 24]
        assert result["summary"]["oos_untouched_from"] == first.isoformat()
    finally:
        _cleanup(source)


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


def test_optimization_freezes_all_model_settings_before_final_execution(monkeypatch):
    source = _insert_source()
    try:
        saved = strategy_research.create_strategy(
            name="Snapshot congelado", source_job_id=source, horizon_hours=2,
        )
        with SessionLocal.begin() as session:
            row = session.get(MLResearchStrategy, saved["id"])
            row.status, row.mode, row.trials_target = "RUNNING", "QUICK", 4
        mock_data = pd.DataFrame({"timestamp": pd.date_range(
            "2024-01-01", periods=5, freq="30min", tz="UTC"
        )})
        monkeypatch.setattr(strategy_research, "_prepare", lambda *args: (
            mock_data, mock_data, mock_data, mock_data,
            ["signal"], None, pd.Timestamp("2024-02-01", tz="UTC"),
        ))
        monkeypatch.setattr(strategy_research, "_score", lambda ds, candidate, params: {
            "score": 0.1, "capital_final": 11000.0,
            "max_drawdown": -0.08, "sharpe": 0.5,
            "worst_fold_return": 0.01, "brier": 0.2, "mae": 0.001,
            "buys": 2, "sells": 2, "return_total": 0.1,
        })
        strategy_research._run_optimization(saved["id"])
        result = strategy_research.get_strategy(saved["id"])
        assert result["status"] == "OPTIMIZED", result["error_message"]
        assert len(result["trials"]) == 4
        snapshot = result["champion"]["model_snapshot"]
        assert "lightgbm.n_estimators" in snapshot
        assert "forecast.regression_target_scale" in snapshot
        assert snapshot["lightgbm.learning_rate"] == result["champion"]["parameters"]["lightgbm.learning_rate"]
        assert result["summary"]["oos_untouched_from"].startswith("2024-01-01")  # immutable archive cutoff, not mocked data
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
