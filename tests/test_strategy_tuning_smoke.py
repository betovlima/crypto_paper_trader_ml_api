"""Real LightGBM tuning smoke test on a frozen pre-OOS candle history."""
from __future__ import annotations

from io import BytesIO
import json
from zipfile import ZipFile
import numpy as np

from crypto_paper_trader_ml_api.archive_replay import inspect_source_archive
from crypto_paper_trader_ml_api.strategy_research import SPACE, _candidate, _prepare, _score
from crypto_paper_trader_ml_api.settings_service import (
    RuntimeParameters, get_runtime_parameters, get_runtime_parameters_from_snapshot,
)

from .test_archive_replay_ui import _make_archive
from .test_features import synthetic_candles


def test_tuning_trains_on_history_strictly_before_first_oos():
    old = _make_archive()
    candles = synthetic_candles(3200)
    trend = candles.iloc[::8].reset_index(drop=True)
    start_oos = candles["timestamp"].iloc[2500]
    output = BytesIO()
    with ZipFile(BytesIO(old)) as source, ZipFile(output, "w") as target:
        replacements = {
            "market_history_decision.csv": candles.to_csv(index=False),
            "market_history_trend.csv": trend.to_csv(index=False),
            "folds.json": json.dumps([{
                "fold": 1, "horizon_hours": 2,
                "training_start": candles["timestamp"].iloc[0].isoformat(),
                "test_start": start_oos.isoformat(),
            }]),
        }
        for filename in source.namelist():
            target.writestr(
                filename, replacements.get(filename, source.read(filename))
            )
    archive = output.getvalue()
    _, snapshot = inspect_source_archive(archive)
    merged = get_runtime_parameters_from_snapshot(snapshot).snapshot()
    merged.update({
        key: value for key, value in get_runtime_parameters().snapshot().items()
        if key.startswith("lightgbm.") or key.startswith("forecast.regression_")
    })
    parameters = RuntimeParameters(merged)
    dataset = _prepare(archive, 2, parameters)
    train, early, calibration, scoring, features, _, first_oos = dataset
    assert len(features) >= 20
    assert min(len(part) for part in [train, early, calibration, scoring]) >= 80
    assert first_oos == start_oos
    for part in [train, early, calibration, scoring]:
        assert part["forecast_future_at_utc"].max() < start_oos
    assert train["forecast_future_at_utc"].max() < early["timestamp"].min()
    assert early["forecast_future_at_utc"].max() < calibration["timestamp"].min()
    assert calibration["forecast_future_at_utc"].max() < scoring["timestamp"].min()
    candidate = _candidate(np.full(len(SPACE), .5))
    metrics = _score(dataset, candidate, parameters)
    assert np.isfinite(metrics["score"])
    assert np.isfinite(metrics["capital_final"])
    assert np.isfinite(metrics["brier"])
    assert metrics["capital_final"] > 0
