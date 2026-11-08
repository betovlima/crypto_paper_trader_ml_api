"""Inspect the trained BTC forecasters without selecting on out-of-sample labels.

All confidence intervals are descriptive moving-block bootstrap intervals,
not evidence of causal effects or investment returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score


def tree_diagnostics(model, predictions: np.ndarray) -> dict:
    """Report LightGBM tree structure using only the boosting rounds in use."""
    predictions = np.asarray(predictions, dtype=float)
    result = {
        "trained": model is not None,
        "unique_predictions": int(np.unique(predictions).size),
        "prediction_standard_deviation": float(np.std(predictions)),
        "prediction_min": float(np.min(predictions)),
        "prediction_max": float(np.max(predictions)),
        "best_iteration": None,
        "best_score": None,
        "total_trees": 0,
        "used_trees": 0,
        "split_nodes_used": 0,
        "leaf_nodes_used": 0,
        "single_leaf_trees_used": 0,
    }
    if model is None:
        return result
    booster = model.booster_
    trees = booster.dump_model()["tree_info"]
    best = getattr(model, "best_iteration_", 0) or booster.current_iteration()
    used = min(len(trees), max(0, int(best)))
    best_scores = getattr(model, "best_score_", {})
    try:
        best_score = next(iter(next(iter(best_scores.values())).values()))
        best_score = float(best_score)
    except (StopIteration, TypeError, AttributeError):
        best_score = None
    result.update({
        "best_iteration": int(best) if best else None,
        "best_score": best_score,
        "total_trees": len(trees),
        "used_trees": used,
    })

    def count_nodes(node: dict) -> tuple[int, int]:
        if "leaf_index" in node:
            return 0, 1
        children = ("left_child", "right_child")
        counts = [count_nodes(node[child]) for child in children if child in node]
        return 1 + sum(splits for splits, _ in counts), sum(leaves for _, leaves in counts)

    for tree in trees[:used]:
        splits, leaves = count_nodes(tree["tree_structure"])
        result["split_nodes_used"] += splits
        result["leaf_nodes_used"] += leaves
        result["single_leaf_trees_used"] += int(splits == 0)
    return result


def _calibration_bins(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10,
) -> list[dict]:
    predicted = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    observed = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    groups = np.minimum(np.searchsorted(edges, predicted, side="right") - 1, bins - 1)
    groups = np.maximum(groups, 0)
    return [
        {
            "lower": float(edges[i]), "upper": float(edges[i + 1]),
            "rows": int(mask.sum()),
            "mean_predicted": float(predicted[mask].mean()) if mask.any() else None,
            "observed_up_rate": float(observed[mask].mean()) if mask.any() else None,
        }
        for i in range(bins)
        for mask in [groups == i]
    ]


def calibration_diagnostics(
    *,
    raw_calibration: np.ndarray, calibration_labels: np.ndarray,
    raw_test: np.ndarray, calibrated_test: np.ndarray, test_labels: np.ndarray,
    baseline_probability: float, calibrator,
) -> dict:
    """Raw vs calibrated outcomes on strictly separate chronological partitions.

    Test metrics are diagnostic only and must not select/re-fit the calibrator.
    """
    cal_labels = np.asarray(calibration_labels, dtype=int)
    test_labels = np.asarray(test_labels, dtype=int)
    raw_cal = np.clip(np.asarray(raw_calibration, dtype=float), 0.0, 1.0)
    raw = np.clip(np.asarray(raw_test, dtype=float), 0.0, 1.0)
    calibrated = np.clip(np.asarray(calibrated_test, dtype=float), 0.0, 1.0)
    model = getattr(calibrator, "model", None)
    return {
        "method": "logistic_platt" if model is not None else "identity",
        "calibration_rows": len(cal_labels),
        "calibration_observed_up_rate": float(np.mean(cal_labels)),
        "calibration_raw_brier": float(brier_score_loss(cal_labels, raw_cal)),
        "calibration_raw_unique": int(np.unique(raw_cal).size),
        "calibration_coefficient": float(model.coef_[0][0]) if model is not None else None,
        "calibration_intercept": float(model.intercept_[0]) if model is not None else None,
        "test_raw_brier": float(brier_score_loss(test_labels, raw)),
        "test_calibrated_brier": float(brier_score_loss(test_labels, calibrated)),
        "test_training_baseline_brier": float(
            brier_score_loss(test_labels, np.full(len(test_labels), baseline_probability))
        ),
        "test_raw_unique": int(np.unique(raw).size),
        "test_calibrated_unique": int(np.unique(calibrated).size),
        "test_raw_mean_probability": float(np.mean(raw)),
        "test_calibrated_mean_probability": float(np.mean(calibrated)),
        "test_observed_up_rate": float(np.mean(test_labels)),
        "test_raw_bins": _calibration_bins(raw, test_labels),
        "test_calibrated_bins": _calibration_bins(calibrated, test_labels),
    }


def _scores(rows: pd.DataFrame) -> dict:
    y = rows["forecast_realized_up"].to_numpy(int)
    p = rows["forecast_p_up"].to_numpy(float)
    baseline_p = rows["forecast_baseline_p_up"].to_numpy(float)
    returns = rows["forecast_realized_return"].to_numpy(float)
    predicted = rows["forecast_predicted_return"].to_numpy(float)
    result: dict = {}
    result["roc_auc"] = (
        float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None
    )
    ref = float(brier_score_loss(y, baseline_p))
    result["brier_skill_score"] = (
        1 - float(brier_score_loss(y, p)) / ref if ref > 0 else None
    )
    zero_mae = float(mean_absolute_error(returns, np.zeros_like(returns)))
    result["return_mae_skill_vs_zero"] = (
        1 - float(mean_absolute_error(returns, predicted)) / zero_mae
        if zero_mae > 0 else None
    )
    return result


def temporal_block_confidence_intervals(
    rows: pd.DataFrame, *, horizon_candles: int,
    samples: int = 200, random_state: int = 42,
) -> dict:
    """95% percentile CIs from moving blocks sampled separately within each OOS fold.

    Block length at least one day or twice the overlapping target horizon.
    A fixed generator makes the audit repeatable. CIs quantify historical
    uncertainty; they cannot fix selection bias or regime drift.
    """
    if samples < 2 or horizon_candles < 1:
        raise ValueError("Invalid temporal bootstrap parameters")
    groups = [
        group.sort_values("timestamp").reset_index(drop=True)
        for _, group in rows.groupby("forecast_fold", sort=True)
    ]
    if not groups or any(group.empty for group in groups):
        raise ValueError("Temporal bootstrap needs nonempty out-of-sample folds")
    block_length = max(48, 2 * horizon_candles)
    rng = np.random.default_rng(random_state + int(horizon_candles))
    intervals: dict[str, dict] = {}
    replicates = {key: [] for key in (
        "roc_auc", "brier_skill_score", "return_mae_skill_vs_zero"
    )}
    for _ in range(samples):
        blocks = []
        for group in groups:
            size = len(group)
            length = min(block_length, size)
            index: list[int] = []
            while len(index) < size:
                first = int(rng.integers(0, size - length + 1))
                index.extend(range(first, first + length))
            blocks.append(group.iloc[index[:size]])
        sampled = pd.concat(blocks, ignore_index=True)
        for key, value in _scores(sampled).items():
            if value is not None and np.isfinite(value):
                replicates[key].append(value)
    for key, values in replicates.items():
        intervals[key] = {
            "lower": float(np.percentile(values, 2.5)) if values else None,
            "upper": float(np.percentile(values, 97.5)) if values else None,
            "valid_replicates": len(values),
        }
    return {
        "method": "95% moving-block bootstrap, stratified by chronological OOS fold",
        "samples": samples,
        "seed": random_state + int(horizon_candles),
        "block_length_candles": block_length,
        "intervals": intervals,
    }
