"""One API request must optimize and schedule OOS, with no browser coordination."""
from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from crypto_paper_trader_ml_api.app import app
from crypto_paper_trader_ml_api import service, strategy_research
from .test_strategy_research import _insert_source, _give_source_three_horizons, _cleanup


class CapturedExecutor:
    def __init__(self):
        self.jobs = []

    def submit(self, function, *args):
        self.jobs.append((function, args))


def _fast_tuning(monkeypatch, *, tradable):
    window = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=8, freq="30min", tz="UTC"),
    })
    monkeypatch.setattr(strategy_research, "_prepare", lambda archive, horizon, params, cutoff: (
        window, window, window, window, ["signal"], horizon, cutoff,
    ))
    monkeypatch.setattr(strategy_research, "_score", lambda dataset, candidate, params: {
        "score": 0.05, "brier": 0.2, "mae": 0.001, "__policy_predictions": {},
    })
    monkeypatch.setattr(strategy_research, "_combine_horizon_metrics",
        lambda per_horizon, **kwargs: {
            "score": 0.12 if tradable else -0.01,
            "capital_final": 10100 if tradable else 10000,
            "completed_cycles": 2 if tradable else 0,
            "entry_diagnostics": {
                "total_aligned_signals": 8,
                "passes_all_entry_filters": 2 if tradable else 0,
                "passes_expected_return": 2 if tradable else 0,
                "passes_probability": 8,
                "diagnosis": "HAS_ENTRY_OPPORTUNITIES" if tradable else "RETURN_FILTER_BLOCKED",
            },
        })


def test_one_click_creates_final_execution_and_retains_original_source(monkeypatch):
    source = _insert_source()
    _give_source_three_horizons(source)
    executor = CapturedExecutor()
    monkeypatch.setattr(service, "_get_executor", lambda params: executor)
    _fast_tuning(monkeypatch, tradable=True)
    try:
        with TestClient(app) as client:
            created = client.post("/v1/ml-strategies", json={
                "name": "Research V6 auto OOS", "source_job_id": source,
            })
            assert created.status_code == 201, created.text
            strategy_id = created.json()["id"]
            started = client.post(f"/v1/ml-strategies/{strategy_id}/optimize", json={
                "mode": "QUICK", "trials": 4, "auto_execute": True,
            })
            assert started.status_code == 202, started.text
            assert started.json()["status"] == "RUNNING"
            assert len(executor.jobs) == 1
            fn, args = executor.jobs[0]
            fn(*args)
            result = client.get(f"/v1/ml-strategies/{strategy_id}").json()
            assert result["status"] == "EXECUTING", result.get("error_message")
            assert result["summary"]["auto_execute"] is True
            assert result["execution_job_id"] is not None
            assert result["source_job_id"] == source
            assert len(result["trials"]) == 4
            assert len(executor.jobs) == 2
            assert executor.jobs[1][0] == service._run_job
    finally:
        _cleanup(source)


def test_one_click_never_executes_when_all_candidates_remain_cash(monkeypatch):
    source = _insert_source()
    _give_source_three_horizons(source)
    executor = CapturedExecutor()
    monkeypatch.setattr(service, "_get_executor", lambda params: executor)
    _fast_tuning(monkeypatch, tradable=False)
    try:
        with TestClient(app) as client:
            created = client.post("/v1/ml-strategies", json={
                "name": "Research V6 no trades", "source_job_id": source,
            })
            assert created.status_code == 201, created.text
            strategy_id = created.json()["id"]
            started = client.post(f"/v1/ml-strategies/{strategy_id}/optimize", json={
                "mode": "BAYESIAN", "trials": 4, "auto_execute": True,
            })
            assert started.status_code == 202, started.text
            executor.jobs[0][0](*executor.jobs[0][1])
            result = client.get(f"/v1/ml-strategies/{strategy_id}").json()
            assert result["status"] == "NO_TRADES", result.get("error_message")
            assert result["execution_job_id"] is None
            assert result["champion"] is None
            assert len(executor.jobs) == 1
    finally:
        _cleanup(source)
