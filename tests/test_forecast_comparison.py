"""Ensure the V1/V2 comparison refuses altered data or test partitions."""
from __future__ import annotations

import csv
from io import StringIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_forecast_audits import compare_audits  # noqa: E402


def _csv_prediction(return_value: str) -> str:
    stream = StringIO()
    fieldnames = [
        "forecast_horizon_hours", "timestamp", "forecast_future_at_utc",
        "forecast_realized_return", "forecast_realized_up",
        "forecast_p_up",
    ]
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({
        "forecast_horizon_hours": "2",
        "timestamp": "2026-01-01T00:00:00.000000Z",
        "forecast_future_at_utc": "2026-01-01T02:30:00.000000Z",
        "forecast_realized_return": return_value,
        "forecast_realized_up": "1.0",
        "forecast_p_up": "0.60",
    })
    return stream.getvalue()


def _audit(
    location: Path, *, market: str = "original", ret: str = "0.01",
    last_fold: str = "2026-01-01T00:00:00Z", auc: float = 0.51,
) -> None:
    manifest = {
        "kind": "BTC_DIRECTION_RETURN_OOS_AUDIT",
        "history_source": "BINANCE_SPOT_ARCHIVE",
        "symbol": "BTCUSDT", "decision_timeframe": "30min",
        "trend_timeframe": "4hour", "market_interruptions": {},
    }
    request = {
        "experiment_type": "BTC_DIRECTION_RETURN",
        "history_source": "BINANCE_SPOT_ARCHIVE",
        "symbol": "BTCUSDT", "period_years": 4,
        "decision_timeframe": "30min", "trend_timeframe": "4hour",
    }
    folds = [{"fold": 1, "horizon_hours": 2, "test_start": last_fold}]
    scores = [{"horizon_hours": 2, "metrics": {
        "rows": 1, "roc_auc": auc, "brier_skill_score": 0.01,
        "return_mae": 0.005, "return_mae_skill_vs_zero": 0.1,
        "return_rmse": 0.006,
    }}]
    with ZipFile(location, "w") as z:
        for name, value in (
            ("manifest.json", manifest), ("request.json", request),
            ("folds.json", folds), ("forecast_metrics.json", scores),
        ):
            z.writestr(name, json.dumps(value))
        z.writestr("market_history_decision.csv", market)
        z.writestr("market_history_trend.csv", "trend stays the same")
        z.writestr("forecast_predictions_oos.csv", _csv_prediction(ret))


def test_controlled_comparison_returns_metric_deltas_only_for_matching_data(tmp_path):
    base, candidate = tmp_path / "base.zip", tmp_path / "candidate.zip"
    _audit(base)
    _audit(candidate, auc=0.56)
    comparison = compare_audits(base, candidate)
    assert comparison["verified_same_market_data"] is True
    assert comparison["verified_same_oos_targets"] is True
    assert comparison["verified_same_fold_boundaries"] is True
    assert comparison["horizons"][0]["metrics"]["roc_auc"]["delta"] == pytest.approx(0.05)


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"market": "different candles"}, "market_history_decision.csv"),
        ({"ret": "0.02"}, "realized OOS targets"),
        ({"last_fold": "2026-01-03T00:00:00Z"}, "fold boundaries"),
    ],
)
def test_comparator_rejects_invalid_claims_of_improvement(tmp_path, kwargs, expected):
    baseline, changed = tmp_path / "base.zip", tmp_path / "changed.zip"
    _audit(baseline)
    _audit(changed, **kwargs)
    with pytest.raises(ValueError, match=expected):
        compare_audits(baseline, changed)
