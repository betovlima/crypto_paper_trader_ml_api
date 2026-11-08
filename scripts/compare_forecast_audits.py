"""Compare two completed BTC_DIRECTION_RETURN ZIP audits under identical data.

Usage:
  python scripts/compare_forecast_audits.py baseline.zip candidate.zip

Fail fast if raw source, candles, realized OOS targets, or fold boundaries
differ. Never claim a controlled improvement across different market periods.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from io import TextIOWrapper
import json
from pathlib import Path
from zipfile import ZipFile


COMPARE_METRICS = (
    "roc_auc", "brier_skill_score", "return_mae",
    "return_mae_skill_vs_zero", "return_rmse",
)


def _json_member(z: ZipFile, path: str):
    return json.loads(z.read(path))


def _stream_sha256(z: ZipFile, path: str) -> str:
    digest = sha256()
    with z.open(path) as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _oos_truth_sha256(z: ZipFile) -> str:
    """Independent of model predictions or feature-column changes."""
    columns = (
        "forecast_horizon_hours", "timestamp", "forecast_future_at_utc",
        "forecast_realized_return", "forecast_realized_up",
    )
    digest = sha256()
    with z.open("forecast_predictions_oos.csv") as stream:
        reader = csv.DictReader(TextIOWrapper(stream, encoding="utf-8-sig"))
        if not set(columns).issubset(reader.fieldnames or []):
            raise ValueError("Audit is missing required OOS ground-truth columns")
        for row in reader:
            payload = "\x1f".join(row[key] for key in columns) + "\n"
            digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def _clean_folds(windows: list[dict]) -> list[dict]:
    fields = (
        "fold", "horizon_hours", "purge_candles_at_each_boundary",
        "training_rows", "training_start", "training_end", "training_last_label_at",
        "early_stopping_rows", "early_stopping_start", "early_stopping_end",
        "early_stopping_last_label_at",
        "calibration_rows", "calibration_start", "calibration_end",
        "calibration_last_label_at",
        "test_rows", "test_start", "test_end", "test_last_label_at",
    )
    return [
        {key: item.get(key) for key in fields}
        for item in windows
    ]


def compare_audits(baseline: str | Path, candidate: str | Path) -> dict:
    with ZipFile(baseline) as left, ZipFile(candidate) as right:
        lm, rm = _json_member(left, "manifest.json"), _json_member(right, "manifest.json")
        if lm.get("kind") != "BTC_DIRECTION_RETURN_OOS_AUDIT" or rm.get(
            "kind"
        ) != "BTC_DIRECTION_RETURN_OOS_AUDIT":
            raise ValueError("Both files must be direct BTC forecasting audits")
        for field in (
            "history_source", "symbol", "decision_timeframe",
            "trend_timeframe", "market_interruptions",
        ):
            if lm.get(field) != rm.get(field):
                raise ValueError(f"Not a controlled experiment: manifest {field} differs")
        lr, rr = _json_member(left, "request.json"), _json_member(right, "request.json")
        for field in (
            "experiment_type", "history_source", "symbol", "period_years",
            "decision_timeframe", "trend_timeframe",
        ):
            if lr.get(field) != rr.get(field):
                raise ValueError(f"Not a controlled experiment: request {field} differs")
        for market in ("market_history_decision.csv", "market_history_trend.csv"):
            if _stream_sha256(left, market) != _stream_sha256(right, market):
                raise ValueError(f"Not a controlled experiment: {market} differs")
        if _clean_folds(_json_member(left, "folds.json")) != _clean_folds(
            _json_member(right, "folds.json")
        ):
            raise ValueError("Not a controlled experiment: fold boundaries differ")
        if _oos_truth_sha256(left) != _oos_truth_sha256(right):
            raise ValueError("Not a controlled experiment: realized OOS targets differ")
        l_metrics = {
            int(item["horizon_hours"]): item["metrics"]
            for item in _json_member(left, "forecast_metrics.json")
        }
        r_metrics = {
            int(item["horizon_hours"]): item["metrics"]
            for item in _json_member(right, "forecast_metrics.json")
        }
        if l_metrics.keys() != r_metrics.keys():
            raise ValueError("Not a controlled experiment: evaluated horizons differ")
        results = []
        for horizon in sorted(l_metrics):
            old, new = l_metrics[horizon], r_metrics[horizon]
            values = {}
            for name in COMPARE_METRICS:
                before, after = old.get(name), new.get(name)
                values[name] = {
                    "baseline": before,
                    "candidate": after,
                    "delta": after - before if before is not None and after is not None else None,
                }
            results.append({
                "horizon_hours": horizon,
                "oos_predictions": old.get("rows"),
                "metrics": values,
            })
        return {
            "verified_same_market_data": True,
            "verified_same_oos_targets": True,
            "verified_same_fold_boundaries": True,
            "history_source": lm["history_source"],
            "horizons": results,
            "notice": "Metrics are observational OOS comparisons; CARO is not yet run.",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Original V1 audit ZIP")
    parser.add_argument("candidate", type=Path, help="New V2 audit ZIP")
    args = parser.parse_args()
    print(json.dumps(compare_audits(args.baseline, args.candidate), indent=2))


if __name__ == "__main__":
    main()
