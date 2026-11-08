"""Out-of-sample-within-development return-forecast diagnostics by horizon.

This module receives only the PRE-OOS economic validation partition; labels
are diagnostic observations, never fed into the forecaster or trade signals.
All return units are fractions (0.01 is 1%). Overlapping multi-hour targets
can be autocorrelated; do not treat sample counts as independent trades.
"""
from __future__ import annotations

import math
import numpy as np


def describe_return_predictions(
    forecasts: np.ndarray, realized: np.ndarray, *,
    training_baseline_return: float, horizon_hours: int,
) -> dict:
    predictions = np.asarray(forecasts, dtype=float)
    actual = np.asarray(realized, dtype=float)
    if predictions.ndim != 1 or actual.ndim != 1 or len(predictions) != len(actual):
        raise ValueError("Previsões e retornos precisam ter a mesma dimensão")
    if len(predictions) == 0 or not (
        np.isfinite(predictions).all()
        and np.isfinite(actual).all()
        and math.isfinite(training_baseline_return)
    ):
        raise ValueError("Dados inválidos para diagnóstico de retornos")
    n = len(predictions)
    error = predictions - actual
    model_mae = float(np.mean(np.abs(error)))
    model_rmse = float(np.sqrt(np.mean(error * error)))
    naive_error = actual - training_baseline_return
    baseline_mae = float(np.mean(np.abs(naive_error)))
    baseline_rmse = float(np.sqrt(np.mean(naive_error * naive_error)))
    mean_forecast = float(np.mean(predictions))
    mean_actual = float(np.mean(actual))
    spread_forecast = float(np.std(predictions))
    spread_actual = float(np.std(actual))
    correlation = (
        float(np.corrcoef(predictions, actual)[0, 1])
        if n >= 3 and spread_forecast > 1e-12 and spread_actual > 1e-12
        else None
    )
    sort_by_prediction = np.argsort(predictions, kind="stable")
    bins = []
    # Quantiles are descriptive and computed entirely on PRE-OOS validation.
    for idx, bucket in enumerate(np.array_split(sort_by_prediction, min(5, n)), 1):
        if not len(bucket):
            continue
        bins.append({
            "rank": idx, "count": int(len(bucket)),
            "predicted_mean": float(np.mean(predictions[bucket])),
            "realized_mean": float(np.mean(actual[bucket])),
            "realized_positive_fraction": float(np.mean(actual[bucket] > 0)),
        })
    return {
        "horizon_hours": int(horizon_hours),
        "rows": n,
        "model_mae": model_mae,
        "model_rmse": model_rmse,
        "training_mean_baseline_prediction": float(training_baseline_return),
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "mae_skill_vs_training_mean": (
            float(1 - model_mae / baseline_mae) if baseline_mae > 1e-12 else None
        ),
        "rmse_skill_vs_training_mean": (
            float(1 - model_rmse / baseline_rmse) if baseline_rmse > 1e-12 else None
        ),
        "prediction_correlation": correlation,
        "mean_predicted_return": mean_forecast,
        "mean_realized_return": mean_actual,
        "mean_prediction_bias": float(np.mean(error)),
        "predicted_return_std": spread_forecast,
        "realized_return_std": spread_actual,
        "prediction_amplitude_ratio": (
            float(spread_forecast / spread_actual) if spread_actual > 1e-12 else None
        ),
        "direction_accuracy": float(np.mean((predictions > 0) == (actual > 0))),
        "constant_direction_accuracy": float(np.mean(
            (training_baseline_return > 0) == (actual > 0)
        )),
        "positive_forecast_fraction": float(np.mean(predictions > 0)),
        "positive_realized_fraction": float(np.mean(actual > 0)),
        "prediction_quantiles": bins,
        "note": (
            "Somente validação pré-OOS; retornos futuros observados são "
            "diagnósticos, nunca entradas do modelo. Horizontes sobrepostos "
            "não representam observações independentes."
        ),
    }
