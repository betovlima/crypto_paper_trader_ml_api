"""End-to-end report of a persisted optimized BTC/CASH policy on archived OOS data."""
from __future__ import annotations

from io import BytesIO
import json
from zipfile import ZipFile

from crypto_paper_trader_ml_api import service, strategy_research
from crypto_paper_trader_ml_api.btc_policy import DEFAULT_POLICY
from crypto_paper_trader_ml_api.database import MLBacktestAudit, MLBacktestJob, MLResearchStrategy, SessionLocal

from .test_capitalization import _all_horizon_forecasts
from .test_strategy_research import _cleanup, _insert_source


def test_optimized_strategy_export_contains_complete_oos_capital_and_trades(monkeypatch):
    source_id = _insert_source()
    class NoOpExecutor:
        def submit(self, fn, *args):
            return None
    monkeypatch.setattr(service, "_get_executor", lambda _parameters: NoOpExecutor())
    try:
        with SessionLocal.begin() as session:
            source = session.get(MLBacktestAudit, source_id)
            source.archive_zip = _all_horizon_forecasts(.0001)
        strategy = strategy_research.create_strategy(
            name="Research BTC/CASH", source_job_id=source_id,
        )
        policy = {**DEFAULT_POLICY,
                  "policy.entry_probability": .52,
                  "policy.min_hold_bars": 0,
                  "policy.cooldown_bars": 1}
        with SessionLocal.begin() as session:
            row = session.get(MLResearchStrategy, strategy["id"])
            row.status = "OPTIMIZED"
            row.champion_json = json.dumps({
                "parameters": policy,
                "metrics": {"score": 0.1, "capital_final": 10300.0},
            })
        job = strategy_research.execute_strategy(strategy["id"])
        with SessionLocal.begin() as session:
            child = session.get(MLBacktestJob, job["id"])
            child.status = "COMPLETED"
            child.result_json = json.dumps({"execution": {"model_family": "LightGBM"}})
            session.add(MLBacktestAudit(
                job_id=job["id"], archive_zip=_all_horizon_forecasts(.015),
            ))
        report_blob = strategy_research.export_strategy_report(strategy["id"])
        with ZipFile(BytesIO(report_blob)) as report:
            filenames = set(report.namelist())
            assert {
                "strategy.json", "optimization_trials.csv",
                "source_historical_audit.zip", "final_oos_audit.zip",
                "final_execution_parameters.json", "final_oos_capitalization.json",
            }.issubset(filenames)
            final = json.loads(report.read("final_oos_capitalization.json"))
            frozen = json.loads(report.read("final_execution_parameters.json"))
        assert final["status"] == "VERIFIED"
        assert final["horizons_hours"] == [2, 6, 24]
        assert final["results"][0]["code"] == "BTC_CASH_ROTATION"
        assert final["results"][1]["code"] == "BUY_AND_HOLD"
        assert final["results"][0]["actions"][0]["action"] == "BUY"
        assert final["results"][0]["actions_truncated"] is False
        assert frozen["policy.enabled"]
        assert frozen["policy.entry_probability"] == .52
    finally:
        _cleanup(source_id)
