"""Independent, decision-time-causal BTC direction and future-return research.

This engine produces auditable forecasts ONLY. It never enters BTC or CASH,
never reuses the target/stop label, and never promotes a model to live trading.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, mean_absolute_error,
    mean_squared_error, roc_auc_score,
)
from xgboost import XGBRegressor

from . import __version__
from .binance_archive import BinancePublicArchiveClient
from .direct_audit import build_direction_audit_archive
from .engine import ProbabilityCalibrator, _fit_model
from .features import build_features, feature_names
from .forecast_diagnostics import (
    calibration_diagnostics, temporal_block_confidence_intervals, tree_diagnostics,
)
from .mexc import MEXCHistoryClient, TIMEFRAME_SECONDS
from .schemas import ResolvedMLBacktestCreate
from .settings_service import RuntimeParameters


EXPERIMENT = "BTC_DIRECTION_RETURN"
HORIZON_UNIT = "HOURS"
MAX_CHART_POINTS = 500


def create_direction_labels(
    featured: pd.DataFrame, timeframe: str, horizon_hours: int,
) -> pd.DataFrame:
    """Attach close-to-close future returns without crossing market closures."""
    candle_seconds = TIMEFRAME_SECONDS[timeframe]
    if horizon_hours <= 0 or (horizon_hours * 3600) % candle_seconds:
        raise ValueError("Forecast horizon must be a multiple of the decision candle")
    h = horizon_hours * 3600 // candle_seconds
    data = featured.copy().sort_values("timestamp").reset_index(drop=True)
    segments = data["market_segment"] if "market_segment" in data else pd.Series(0, index=data.index)
    by_segment = data.groupby(segments, sort=False)
    close = data["close"].astype(float)
    next_close = by_segment["close"].shift(-h).astype(float)
    next_open = by_segment["timestamp"].shift(-h)
    valid = close.gt(0) & next_close.gt(0) & next_close.notna() & next_open.notna()
    realized = (next_close / close - 1).where(valid)
    data["forecast_future_close"] = next_close.where(valid)
    data["forecast_future_at_utc"] = (
        pd.to_datetime(next_open, utc=True) + pd.Timedelta(seconds=candle_seconds)
    ).where(valid)
    data["forecast_realized_return"] = realized
    data["forecast_realized_up"] = realized.gt(0).astype(float).where(valid)
    data["forecast_horizon_hours"] = horizon_hours
    data["forecast_horizon_candles"] = int(h)
    return data


def _regressor(
    training: pd.DataFrame, validation: pd.DataFrame,
    columns: list[str], parameters: RuntimeParameters,
) -> XGBRegressor | None:
    """Fit in scaled target units; probabilities and reported returns stay unscaled."""
    scale = parameters.floating("forecast.regression_target_scale")
    outcomes = training["forecast_realized_return"].to_numpy(float) * scale
    if np.ptp(outcomes) < 1e-8:
        return None
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method=parameters.string("xgboost.tree_method"),
        n_estimators=parameters.integer("forecast.regression_n_estimators"),
        max_depth=parameters.integer("forecast.regression_max_depth"),
        learning_rate=parameters.floating("forecast.regression_learning_rate"),
        min_child_weight=parameters.floating("forecast.regression_min_child_weight"),
        subsample=parameters.floating("xgboost.subsample"),
        colsample_bytree=parameters.floating("xgboost.colsample_bytree"),
        gamma=parameters.floating("forecast.regression_gamma"),
        reg_alpha=parameters.floating("forecast.regression_reg_alpha"),
        reg_lambda=parameters.floating("forecast.regression_reg_lambda"),
        random_state=parameters.integer("xgboost.random_state"),
        n_jobs=parameters.integer("xgboost.n_jobs"),
        early_stopping_rounds=parameters.integer(
            "forecast.regression_early_stopping_rounds"
        ),
    )
    model.fit(
        training[columns], outcomes,
        eval_set=[(
            validation[columns],
            validation["forecast_realized_return"].to_numpy(float) * scale,
        )],
        verbose=False,
    )
    return model


def _metrics(rows: pd.DataFrame) -> dict:
    """Measure directional and numerical forecast skill vs chronological baselines."""
    y = rows["forecast_realized_up"].to_numpy(int)
    probabilities = np.clip(rows["forecast_p_up"].to_numpy(float), 1e-8, 1 - 1e-8)
    baseline_p = np.clip(rows["forecast_baseline_p_up"].to_numpy(float), 1e-8, 1 - 1e-8)
    returns = rows["forecast_realized_return"].to_numpy(float)
    estimates = rows["forecast_predicted_return"].to_numpy(float)
    baseline_mean = rows["forecast_baseline_mean_return"].to_numpy(float)

    brier = float(brier_score_loss(y, probabilities))
    reference_brier = float(brier_score_loss(y, baseline_p))
    skill = (1 - brier / reference_brier) if reference_brier > 0 else None
    mae = float(mean_absolute_error(returns, estimates))
    zero_mae = float(mean_absolute_error(returns, np.zeros_like(returns)))
    mean_mae = float(mean_absolute_error(returns, baseline_mean))
    rmse = float(np.sqrt(mean_squared_error(returns, estimates)))
    zero_rmse = float(np.sqrt(mean_squared_error(returns, np.zeros_like(returns))))
    has_both = len(np.unique(y)) == 2
    return {
        "rows": len(rows),
        "realized_up_rate": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, probabilities)) if has_both else None,
        "pr_auc": float(average_precision_score(y, probabilities)) if has_both else None,
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "brier_score": brier,
        "baseline_brier_score": reference_brier,
        "brier_skill_score": skill,
        "direction_accuracy": float(np.mean((probabilities >= 0.5) == y)),
        "baseline_direction_accuracy": float(np.mean((baseline_p >= 0.5) == y)),
        "return_mae": mae,
        "return_rmse": rmse,
        "zero_return_mae": zero_mae,
        "zero_return_rmse": zero_rmse,
        "training_mean_return_mae": mean_mae,
        "return_mae_skill_vs_zero": 1 - mae / zero_mae if zero_mae > 0 else None,
        "return_rmse_skill_vs_zero": 1 - rmse / zero_rmse if zero_rmse > 0 else None,
        "average_realized_return": float(np.mean(returns)),
        "average_predicted_return": float(np.mean(estimates)),
    }


def _fold_boundaries(block: pd.DataFrame, name: str) -> dict:
    return {
        name + "_rows": len(block),
        name + "_start": pd.Timestamp(block.iloc[0]["timestamp"]).isoformat(),
        name + "_end": pd.Timestamp(block.iloc[-1]["timestamp"]).isoformat(),
        name + "_last_label_at": pd.Timestamp(
            block.iloc[-1]["forecast_future_at_utc"]
        ).isoformat(),
    }


def walk_forward_direction(
    featured: pd.DataFrame, parameters: RuntimeParameters,
    *, timeframe: str, horizon_hours: int, cancelled: Callable[[], bool],
    progress: Callable[[int, str], None] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Train -> early-stop -> calibrate -> test with a purged horizon between each."""
    horizon_frame = create_direction_labels(featured, timeframe, horizon_hours)
    columns = feature_names(parameters)
    usable = horizon_frame.dropna(subset=[
        *columns, "forecast_realized_up", "forecast_realized_return",
        "forecast_future_at_utc",
    ]).reset_index(drop=True)
    minimum = parameters.integer("walk_forward.minimum_usable_candles")
    if len(usable) < minimum:
        raise ValueError(
            f"{horizon_hours}h: only {len(usable)} usable candles; minimum is {minimum}"
        )
    initial = max(
        parameters.integer("walk_forward.initial_train_minimum"),
        int(len(usable) * parameters.floating("walk_forward.initial_train_fraction")),
    )
    remaining = len(usable) - initial
    maximum_folds = parameters.integer("forecast.maximum_folds")
    fold_count = min(
        maximum_folds,
        max(2, remaining // parameters.integer("walk_forward.fold_size_divisor")),
    )
    if remaining <= 0 or fold_count < 2:
        raise ValueError("No independent test period remains after the initial training window")
    fold_size = max(1, remaining // fold_count)
    h = horizon_frame["forecast_horizon_candles"].iloc[0]
    validation_minimum = parameters.integer("walk_forward.validation_minimum")
    validation_fraction = parameters.floating("walk_forward.validation_fraction")
    minimum_test = parameters.integer("walk_forward.minimum_test_size")

    forecasts: list[pd.DataFrame] = []
    windows: list[dict] = []
    classifier_importances = np.zeros(len(columns), dtype=float)
    regression_importances = np.zeros(len(columns), dtype=float)
    trained_classifiers = 0
    trained_regressors = 0

    for fold in range(fold_count):
        if cancelled():
            raise InterruptedError("Direct BTC Forecast cancelled")
        test_begin = initial + fold * fold_size
        test_end = len(usable) if fold == fold_count - 1 else min(len(usable), test_begin + fold_size)
        if test_end - test_begin < minimum_test:
            continue

        calibration_size = max(validation_minimum, int(test_begin * validation_fraction))
        validation_size = max(validation_minimum, int(test_begin * validation_fraction / 2))
        calibration_end = test_begin - h
        calibration_begin = calibration_end - calibration_size
        validation_end = calibration_begin - h
        validation_begin = validation_end - validation_size
        training_end = validation_begin - h
        if training_end <= parameters.integer("walk_forward.initial_train_minimum") // 2:
            continue

        training = usable.iloc[:training_end].copy()
        validation = usable.iloc[validation_begin:validation_end].copy()
        calibration = usable.iloc[calibration_begin:calibration_end].copy()
        testing = usable.iloc[test_begin:test_end].copy()
        if min(len(training), len(validation), len(calibration), len(testing)) == 0:
            continue
        # A data point must have a fully realized target BEFORE the NEXT
        # partition starts. This protects irregular timestamps and hiatuses.
        boundaries = [(training, validation), (validation, calibration), (calibration, testing)]
        for left, right in boundaries:
            max_known = pd.Timestamp(right.iloc[0]["timestamp"])
            left.drop(
                index=left.index[pd.to_datetime(left["forecast_future_at_utc"], utc=True) >= max_known],
                inplace=True,
            )
        if min(len(training), len(validation), len(calibration)) == 0:
            continue
        if len(validation) < validation_minimum or len(calibration) < validation_minimum:
            continue

        classifier = _fit_model(
            training[columns], training["forecast_realized_up"].to_numpy(int),
            validation[columns], validation["forecast_realized_up"].to_numpy(int),
            parameters,
        )
        regressor = _regressor(training, validation, columns, parameters)
        p_train = float(training["forecast_realized_up"].mean())
        ret_train = float(training["forecast_realized_return"].mean())

        if classifier is None:
            raw_calibration = np.full(len(calibration), p_train)
            raw_test = np.full(len(testing), p_train)
        else:
            raw_calibration = classifier.predict_proba(calibration[columns])[:, 1]
            raw_test = classifier.predict_proba(testing[columns])[:, 1]
            classifier_importances += classifier.feature_importances_
            trained_classifiers += 1

        calibrator = ProbabilityCalibrator(
            solver=parameters.string("xgboost.calibration_solver")
        )
        calibrator.fit(raw_calibration, calibration["forecast_realized_up"].to_numpy(int))

        testing["forecast_raw_p_up"] = np.clip(raw_test, 0.0, 1.0)
        testing["forecast_p_up"] = np.clip(
            calibrator.transform(raw_test), 0.0, 1.0,
        )
        scale = parameters.floating("forecast.regression_target_scale")
        testing["forecast_predicted_return"] = (
            regressor.predict(testing[columns]) / scale
            if regressor is not None else np.full(len(testing), ret_train)
        )
        calibration_info = calibration_diagnostics(
            raw_calibration=raw_calibration,
            calibration_labels=calibration["forecast_realized_up"].to_numpy(int),
            raw_test=raw_test,
            calibrated_test=testing["forecast_p_up"].to_numpy(float),
            test_labels=testing["forecast_realized_up"].to_numpy(int),
            baseline_probability=p_train,
            calibrator=calibrator,
        )
        regression_info = tree_diagnostics(
            regressor, testing["forecast_predicted_return"].to_numpy(float)
        )
        direction_info = tree_diagnostics(classifier, raw_test)
        if regressor is not None:
            regression_importances += regressor.feature_importances_
            trained_regressors += 1
        testing["forecast_baseline_p_up"] = p_train
        testing["forecast_baseline_mean_return"] = ret_train
        testing["forecast_fold"] = fold + 1
        testing["forecast_horizon_hours"] = horizon_hours

        window = {
            "fold": fold + 1,
            "horizon_hours": horizon_hours,
            "purge_candles_at_each_boundary": int(h),
            **_fold_boundaries(training, "training"),
            **_fold_boundaries(validation, "early_stopping"),
            **_fold_boundaries(calibration, "calibration"),
            **_fold_boundaries(testing, "test"),
            "model_diagnostics": {
                "regression": {
                    **regression_info,
                    "target_scale": scale,
                    "target_unit": "simple_return",
                    "fit_target_unit": "scaled_simple_return",
                },
                "direction": direction_info,
                "calibration": calibration_info,
            },
        }
        for left_name, right_name in (
            ("training", "early_stopping"), ("early_stopping", "calibration"),
            ("calibration", "test"),
        ):
            if not pd.Timestamp(window[left_name + "_last_label_at"]) < pd.Timestamp(
                window[right_name + "_start"]
            ):
                raise ValueError("Target horizon overlaps the next temporal partition")
        windows.append(window)
        forecasts.append(testing)
        if progress:
            progress(fold + 1, f"{horizon_hours}h · fold {fold + 1}/{fold_count}")

    if not forecasts:
        raise ValueError(f"{horizon_hours}h: no valid purged forecast folds")
    output = pd.concat(forecasts, ignore_index=True).sort_values("timestamp")
    output.drop_duplicates("timestamp", keep="last", inplace=True)
    output.reset_index(drop=True, inplace=True)
    importances = {
        "direction": sorted([
            {"feature": name, "importance": float(importance / max(1, trained_classifiers))}
            for name, importance in zip(columns, classifier_importances, strict=True)
        ], key=lambda item: item["importance"], reverse=True),
        "return": sorted([
            {"feature": name, "importance": float(importance / max(1, trained_regressors))}
            for name, importance in zip(columns, regression_importances, strict=True)
        ], key=lambda item: item["importance"], reverse=True),
    }
    return output, {"horizon_hours": horizon_hours, "horizon_candles": int(h), "fold_windows": windows, "feature_importance": importances}


def _history_coverage(
    decision: pd.DataFrame, request: ResolvedMLBacktestCreate, client: object,
) -> dict:
    seconds = TIMEFRAME_SECONDS[request.decision_timeframe]
    timestamps = pd.to_datetime(decision["timestamp"], utc=True)
    count = math.ceil(request.period_years * 365.25 * 86400 / seconds)
    diffs = timestamps.diff().dropna().dt.total_seconds() / seconds
    missing = int((diffs.round().clip(lower=1) - 1).sum())
    interruptions = getattr(client, "market_interruptions", {})
    documented = sum(
        int(item["missing_candles"])
        for item in interruptions.get(request.decision_timeframe, [])
    )
    expected = int(round((timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / seconds)) + 1
    coverage = min(1.0, expected / max(1, count))
    return {
        "source": request.history_source,
        "raw_candles": len(decision),
        "raw_start": timestamps.iloc[0].isoformat(),
        "raw_end": timestamps.iloc[-1].isoformat(),
        "requested_years": request.period_years,
        "requested_candles": count,
        "missing_intervals": missing,
        "documented_market_halt_candles": documented,
        "requested_period_coverage_ratio": coverage,
        "coverage_status": (
            "FULL" if coverage >= 0.99 and missing == 0 else
            "FULL_WITH_DOCUMENTED_HALT" if coverage >= 0.99 and missing == documented and documented else
            "PARTIAL"
        ),
        "market_interruptions": interruptions,
    }


def run_direction_backtest(
    request: ResolvedMLBacktestCreate, parameters: RuntimeParameters,
    progress: Callable[[int, str], None], cancelled: Callable[[], bool],
    audit_writer: Callable[[bytes], None] | None = None,
    historical_client: object | None = None,
) -> dict:
    """A forecasting research job; deliberately no portfolio simulation."""
    if request.experiment_type != EXPERIMENT or request.symbol != "BTCUSDT":
        raise ValueError("Direct BTC forecast requires the BTC_DIRECTION_RETURN experiment")
    client = historical_client if historical_client is not None else (
        BinancePublicArchiveClient(parameters, __version__)
        if request.history_source == "BINANCE_SPOT_ARCHIVE"
        else MEXCHistoryClient(parameters, __version__)
    )
    try:
        progress(3, "Downloading BTC forecasting history")
        decision = client.fetch_history(
            symbol=request.symbol, timeframe=request.decision_timeframe,
            period_years=request.period_years,
            progress=lambda value: progress(3 + int(value * .10), "Downloading decision candles"),
            cancelled=cancelled,
        )
        if request.trend_timeframe == request.decision_timeframe:
            trend = decision
        else:
            trend = client.fetch_history(
                symbol=request.symbol, timeframe=request.trend_timeframe,
                period_years=request.period_years,
                progress=lambda value: progress(14 + int(value * .07), "Downloading trend candles"),
                cancelled=cancelled,
            )
        featured = build_features(
            decision, parameters, trend,
            decision_timeframe=request.decision_timeframe,
            trend_timeframe=request.trend_timeframe,
        )
        progress(25, "Preparing direct BTC movement forecasts")
        horizons = [int(value) for value in parameters.list("forecast.horizons_hours")]
        all_predictions = []
        outputs = []
        window_metadata = []
        for index, horizon in enumerate(horizons):
            progress(27 + int(60 * index / len(horizons)), f"Forecasting {horizon}h direction and return")
            predicted, details = walk_forward_direction(
                featured, parameters,
                timeframe=request.decision_timeframe, horizon_hours=horizon,
                cancelled=cancelled,
            )
            all_predictions.append(predicted)
            window_metadata.extend(details["fold_windows"])
            stride = max(1, math.ceil(len(predicted) / MAX_CHART_POINTS))
            sampled = predicted.iloc[::stride]
            sample_columns = (
                "timestamp", "close", "forecast_future_at_utc",
                "forecast_realized_return", "forecast_realized_up",
                "forecast_raw_p_up", "forecast_p_up",
                "forecast_predicted_return",
                "forecast_baseline_p_up", "forecast_fold",
            )
            outputs.append({
                "horizon_hours": horizon,
                "horizon_candles": details["horizon_candles"],
                "metrics": _metrics(predicted),
                "confidence_intervals": temporal_block_confidence_intervals(
                    predicted, horizon_candles=details["horizon_candles"],
                    samples=parameters.integer("forecast.bootstrap_samples"),
                    random_state=parameters.integer("xgboost.random_state"),
                ),
                "folds": [
                    {
                        "fold": int(f),
                        "metrics": _metrics(group),
                        "model_diagnostics": next(
                            (
                                window.get("model_diagnostics")
                                for window in details["fold_windows"]
                                if int(window["fold"]) == int(f)
                            ), None,
                        ),
                    }
                    for f, group in predicted.groupby("forecast_fold", sort=True)
                ],
                "feature_importance": {
                    name: items[:15] for name, items in details["feature_importance"].items()
                },
                "sample": [
                    {
                        col: (
                            pd.Timestamp(row[col]).isoformat()
                            if col in {"timestamp", "forecast_future_at_utc"}
                            else int(row[col]) if col == "forecast_fold"
                            else float(row[col])
                        )
                        for col in sample_columns
                    } for _, row in sampled.iterrows()
                ],
            })
        coverage = _history_coverage(decision, request, client)
        progress(92, "Preparing research results and audit")
        result = {
            "experiment_type": EXPERIMENT,
            "summary": {"experiment_type": EXPERIMENT, "financial_trading_enabled": False},
            "execution": {
                "decision_timeframe": request.decision_timeframe,
                "trend_timeframe": request.trend_timeframe,
                "history_source": request.history_source,
                "history_coverage": coverage,
                "market_interruptions": getattr(client, "market_interruptions", {}),
                "experiment_type": EXPERIMENT,
                "walk_forward_folds": parameters.integer("forecast.maximum_folds"),
                "forecast_horizons_hours": horizons,
                "fold_windows": window_metadata,
                "parameter_source": "DATABASE_SNAPSHOT",
                "replay_original_audit_sha256": getattr(
                    client, "replay_source_sha256", None
                ),
            },
            "forecast": {"horizons": outputs},
            "comparisons": [],
            "trades": [],
            "equity_curve": [],
            "rotation_timeline": [],
            "insights": {
                "research_only": True,
                "no_portfolio_simulation": True,
                "prediction_target": "BTC close-to-close positive return and simple percentage return",
                "calibration": "Temporal calibration holdout after early stopping",
                "regression_target_scale": parameters.floating(
                    "forecast.regression_target_scale"
                ),
                "bootstrap": "Moving-block 95% CI per horizon, stratified by OOS fold",
                "model_diagnostics": "Raw and calibrated probability bins, best XGBoost tree iterations, actual split nodes and prediction dispersion per fold",
            },
        }
        if audit_writer is not None:
            audit_writer(build_direction_audit_archive(
                predictions=all_predictions,
                decision_history=decision, trend_history=trend,
                request=request, parameters=parameters,
                result=result,
                source_archives=getattr(client, "archive_manifest", None),
                archived_market_bytes=getattr(client, "market_bytes", None),
            ))
        progress(100, "Completed")
        return result
    finally:
        client.close()
