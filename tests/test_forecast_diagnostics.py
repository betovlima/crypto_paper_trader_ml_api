"""Regression scale, real trained LightGBM splits and time-dependence diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_paper_trader_ml_api.direct_forecast import _regressor
from crypto_paper_trader_ml_api.engine import ProbabilityCalibrator
from crypto_paper_trader_ml_api.forecast_diagnostics import (
    _calibration_bins, calibration_diagnostics, temporal_block_confidence_intervals,
    tree_diagnostics,
)

from .helpers import default_parameters


def test_independent_scaled_regressor_learns_nonconstant_returns_without_changing_units():
    x = np.sin(np.arange(640) / 21.0)
    # Both chronological partitions contain both market regimes; a tree
    # cannot extrapolate beyond the training feature range.
    # The target is a small *decimal* return. Training scales it to bps.
    returns = 0.007 * x + 0.003 * x**2
    frame = pd.DataFrame({
        "signal": x,
        "forecast_realized_return": returns,
    })
    train, validation = frame.iloc[:460], frame.iloc[460:]
    params = default_parameters(**{
        "forecast.regression_target_scale": 10000.0,
        "forecast.regression_n_estimators": 60,
        "forecast.regression_max_depth": 3,
        "forecast.regression_learning_rate": 0.10,
        "forecast.regression_min_child_weight": 1.0,
        "forecast.regression_gamma": 0.0,
        "forecast.regression_reg_alpha": 0.0,
        "forecast.regression_reg_lambda": 1.0,
        "forecast.regression_early_stopping_rounds": 10,
        "lightgbm.n_jobs": 1,
        "lightgbm.min_child_samples": 5,
    })
    model = _regressor(train, validation, ["signal"], params)
    assert model is not None
    prediction_decimal = model.predict(validation[["signal"]]) / params.floating(
        "forecast.regression_target_scale"
    )
    diagnostics = tree_diagnostics(model, prediction_decimal)
    assert np.isfinite(prediction_decimal).all()
    assert max(abs(prediction_decimal)) < 0.5
    assert np.ptp(prediction_decimal) > 1e-5
    assert diagnostics["trained"] is True
    assert diagnostics["used_trees"] >= 1
    assert diagnostics["split_nodes_used"] > 0
    assert diagnostics["unique_predictions"] > 1
    assert diagnostics["best_iteration"] is not None


def test_regression_can_be_constant_with_all_single_leaf_trees_and_diagnosed():
    rng = np.random.default_rng(7)
    features = rng.normal(size=180)
    frame = pd.DataFrame({
        "signal": features,
        "forecast_realized_return": 0.01 * features,
    })
    params = default_parameters(**{
        "forecast.regression_n_estimators": 20,
        "forecast.regression_early_stopping_rounds": 4,
        "forecast.regression_gamma": 1000.0,
        "forecast.regression_reg_alpha": 0.0,
        "forecast.regression_target_scale": 1.0,
        "lightgbm.n_jobs": 1,
    })
    model = _regressor(frame.iloc[:130], frame.iloc[130:], ["signal"], params)
    assert model is not None
    prediction = model.predict(frame[["signal"]])
    details = tree_diagnostics(model, prediction)
    assert details["trained"] is True
    assert details["split_nodes_used"] == 0
    assert details["single_leaf_trees_used"] == details["used_trees"]
    assert details["unique_predictions"] == 1


def test_calibration_diagnostics_preserve_raw_and_platt_test_probabilities():
    raw_cal = np.array([0.2, 0.8, 0.7, 0.1, 0.8, 0.3])
    cal_labels = np.array([0, 1, 1, 0, 1, 0])
    calibrator = ProbabilityCalibrator(solver="lbfgs")
    calibrator.fit(raw_cal, cal_labels)
    raw_test = np.array([0.1, 0.9, 0.6, 0.4])
    observed = np.array([0, 1, 1, 0])
    fitted = calibrator.transform(raw_test)
    result = calibration_diagnostics(
        raw_calibration=raw_cal, calibration_labels=cal_labels,
        raw_test=raw_test, calibrated_test=fitted, test_labels=observed,
        baseline_probability=0.5, calibrator=calibrator,
    )
    assert result["method"] == "identity"  # tiny calibration samples must not be overfit
    assert result["test_raw_unique"] == 4
    assert result["test_calibrated_unique"] == 4
    assert result["calibration_coefficient"] is None
    assert sum(item["rows"] for item in result["test_raw_bins"]) == 4
    assert sum(item["rows"] for item in result["test_calibrated_bins"]) == 4
    assert result["test_raw_brier"] >= 0
    assert result["test_calibrated_brier"] >= 0


def test_block_bootstrap_is_reproducible_and_never_resamples_across_fold_ids():
    rng = np.random.default_rng(2026)
    n = 240
    observed = rng.normal(0, 0.01, n)
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="30min", tz="UTC"),
        "forecast_fold": [1] * (n // 2) + [2] * (n // 2),
        "forecast_realized_return": observed,
        "forecast_realized_up": (observed > 0).astype(int),
        "forecast_p_up": np.where(observed > 0, 0.7, 0.3),
        "forecast_baseline_p_up": [0.5] * n,
        "forecast_predicted_return": observed * 0.4,
    })
    a = temporal_block_confidence_intervals(
        frame, horizon_candles=12, samples=50, random_state=2026,
    )
    b = temporal_block_confidence_intervals(
        frame, horizon_candles=12, samples=50, random_state=2026,
    )
    assert a == b
    assert a["block_length_candles"] >= 12
    assert a["samples"] == 50
    for item in a["intervals"].values():
        assert item["valid_replicates"] > 0
        assert item["lower"] <= item["upper"]


def test_calibration_bins_keep_probability_one_in_last_bin():
    bins = _calibration_bins(
        np.array([0.0, 0.1, 0.55, 1.0]), np.array([0, 0, 1, 1]),
    )
    assert sum(row["rows"] for row in bins) == 4
    assert bins[0]["rows"] == 1
    assert bins[-1]["rows"] == 1
    assert bins[-1]["observed_up_rate"] == pytest.approx(1.0)
