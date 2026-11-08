from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_score, roc_auc_score
from lightgbm import LGBMClassifier
from .lightgbm_models import fit_classifier

from . import __version__
from .audit_export import build_audit_archive
from .binance_archive import BinancePublicArchiveClient
from .features import atr_column, build_features, create_labels, feature_names, signal_ema_columns
from .mexc import MEXCHistoryClient, TIMEFRAME_SECONDS
from .rotation_analysis import build_rotation_timeline
from .schemas import ResolvedMLBacktestCreate
from .settings_service import RuntimeParameters


@dataclass
class ProbabilityCalibrator:
    """Calibrate each forecast horizon on its OWN chronology before OOS.

    Raw, Platt-on-probabilities and Platt-on-log-odds are evaluated on a
    held-out tail of the calibration partition. Only that partition's labels
    choose the method; the selection/test partition is never consulted.
    """
    solver: str
    model: LogisticRegression | None = None
    method: str = "identity"
    selection_brier: float | None = None
    selection_rows: int = 0
    calibration_rows: int = 0

    @staticmethod
    def _features(probabilities: np.ndarray, method: str) -> np.ndarray:
        raw = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        if method == "logistic_logit":
            raw = np.log(raw / (1.0 - raw))
        return raw.reshape(-1, 1)

    def fit(self, raw_probabilities: np.ndarray, labels: np.ndarray) -> None:
        from sklearn.metrics import brier_score_loss
        raw = np.clip(np.asarray(raw_probabilities, dtype=float), 1e-6, 1 - 1e-6)
        targets = np.asarray(labels, dtype=int)
        if len(raw) != len(targets):
            raise ValueError("Calibration predictions/labels length mismatch")
        self.model = None
        self.method = "identity"
        self.selection_brier = None
        self.selection_rows = 0
        self.calibration_rows = len(targets)
        if len(raw) < 100 or np.unique(targets).size < 2:
            return
        # Chronological selection: no randomized split or scoring-window labels.
        split = int(len(targets) * 0.75)
        fit_labels = targets[:split]
        holdout = targets[split:]
        if np.unique(fit_labels).size < 2 or len(holdout) < 25:
            return
        best_method = "identity"
        best_brier = float(brier_score_loss(holdout, raw[split:]))
        self.selection_rows = len(holdout)
        self.selection_brier = best_brier
        for candidate in ("logistic_raw", "logistic_logit"):
            try:
                trial = LogisticRegression(solver=self.solver, max_iter=500)
                trial.fit(self._features(raw[:split], candidate), fit_labels)
                values = trial.predict_proba(self._features(raw[split:], candidate))[:, 1]
                trial_brier = float(brier_score_loss(holdout, values))
                # A small minimum improvement discourages selection on noise.
                if trial_brier < best_brier - 1e-4:
                    best_brier, best_method = trial_brier, candidate
            except (ValueError, FloatingPointError):
                continue
        self.method = best_method
        self.selection_brier = best_brier
        if best_method != "identity":
            model = LogisticRegression(solver=self.solver, max_iter=500)
            model.fit(self._features(raw, best_method), targets)
            self.model = model

    def transform(self, raw_probabilities: np.ndarray) -> np.ndarray:
        raw = np.clip(np.asarray(raw_probabilities, dtype=float), 1e-6, 1 - 1e-6)
        if self.model is None:
            return raw
        return np.clip(
            self.model.predict_proba(self._features(raw, self.method))[:, 1],
            1e-6, 1 - 1e-6,
        )


def _constant_probability(labels: np.ndarray, size: int) -> np.ndarray:
    return np.full(size, float(np.mean(labels)) if len(labels) else 0.5)


def _fit_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    y_validation: np.ndarray,
    parameters: RuntimeParameters,
) -> LGBMClassifier | None:
    """Chronological LightGBM classification for both legacy and direct research."""
    return fit_classifier(
        x_train, y_train, x_validation, y_validation, parameters
    )


def walk_forward_predict(
    frame: pd.DataFrame,
    parameters: RuntimeParameters,
    progress: Callable[[int, str], None],
    cancelled: Callable[[], bool],
) -> tuple[pd.DataFrame, dict]:
    columns = feature_names(parameters)
    usable = frame.dropna(subset=[*columns, "entry_label", "exit_label"]).copy()
    usable.reset_index(drop=True, inplace=True)
    minimum_usable = parameters.integer("walk_forward.minimum_usable_candles")
    if len(usable) < minimum_usable:
        raise ValueError(
            f"Only {len(usable)} usable candles remain after feature engineering; "
            f"the database parameter walk_forward.minimum_usable_candles requires {minimum_usable}."
        )

    initial_train = max(
        parameters.integer("walk_forward.initial_train_minimum"),
        int(len(usable) * parameters.floating("walk_forward.initial_train_fraction")),
    )
    if initial_train >= len(usable):
        raise ValueError("The database walk-forward parameters leave no out-of-sample candles.")

    remaining = len(usable) - initial_train
    fold_count = min(
        parameters.integer("walk_forward.maximum_folds"),
        max(
            parameters.integer("walk_forward.minimum_folds"),
            remaining // parameters.integer("walk_forward.fold_size_divisor"),
        ),
    )
    fold_size = max(
        parameters.integer("walk_forward.minimum_fold_size"),
        remaining // max(1, fold_count),
    )
    predictions: list[pd.DataFrame] = []
    importance_accumulator = np.zeros(len(columns), dtype=float)
    trained_models = 0
    solver = parameters.string("lightgbm.calibration_solver")
    purge_candles = parameters.integer("labels.horizon_candles")
    fold_windows: list[dict] = []

    for fold in range(fold_count):
        if cancelled():
            raise InterruptedError("Backtest cancellation requested")
        train_end = initial_train + fold * fold_size
        test_end = len(usable) if fold == fold_count - 1 else min(len(usable), train_end + fold_size)
        if test_end - train_end < parameters.integer("walk_forward.minimum_test_size"):
            continue
        train_block = usable.iloc[:train_end]
        validation_size = max(
            parameters.integer("walk_forward.validation_minimum"),
            int(len(train_block) * parameters.floating("walk_forward.validation_fraction")),
        )
        # Purge the entire target horizon before both the calibration and
        # test partitions. No training label may observe a validation candle;
        # no calibration label may observe a test candle.
        validation_end = train_end - purge_candles
        validation_start = validation_end - validation_size
        training_end = validation_start - purge_candles
        if training_end <= purge_candles or validation_start < 0:
            continue
        training = usable.iloc[:training_end]
        validation = usable.iloc[validation_start:validation_end]
        testing = usable.iloc[train_end:test_end].copy()
        if len(validation) < parameters.integer("walk_forward.validation_minimum"):
            continue
        # Audit the actual timestamp of label availability, not merely row
        # offsets. Conservative purging also covers gaps in market candles.
        if "label_available_at" in usable:
            training = training[
                pd.to_datetime(training["label_available_at"], utc=True)
                < pd.Timestamp(validation.iloc[0]["timestamp"])
            ]
            validation = validation[
                pd.to_datetime(validation["label_available_at"], utc=True)
                < pd.Timestamp(testing.iloc[0]["timestamp"])
            ]
        if training.empty or validation.empty:
            continue

        x_train = training[columns]
        x_validation = validation[columns]
        x_test = testing[columns]

        entry_train = training["entry_label"].to_numpy(int)
        entry_validation = validation["entry_label"].to_numpy(int)
        exit_train = training["exit_label"].to_numpy(int)
        exit_validation = validation["exit_label"].to_numpy(int)

        entry_model = _fit_model(x_train, entry_train, x_validation, entry_validation, parameters)
        exit_model = _fit_model(x_train, exit_train, x_validation, exit_validation, parameters)

        if entry_model is None:
            raw_entry_validation = _constant_probability(entry_train, len(validation))
            raw_entry_test = _constant_probability(entry_train, len(testing))
        else:
            raw_entry_validation = entry_model.predict_proba(x_validation)[:, 1]
            raw_entry_test = entry_model.predict_proba(x_test)[:, 1]
            importance_accumulator += entry_model.feature_importances_
            trained_models += 1

        if exit_model is None:
            raw_exit_validation = _constant_probability(exit_train, len(validation))
            raw_exit_test = _constant_probability(exit_train, len(testing))
        else:
            raw_exit_validation = exit_model.predict_proba(x_validation)[:, 1]
            raw_exit_test = exit_model.predict_proba(x_test)[:, 1]
            importance_accumulator += exit_model.feature_importances_
            trained_models += 1

        entry_calibrator = ProbabilityCalibrator(solver=solver)
        entry_calibrator.fit(raw_entry_validation, entry_validation)
        exit_calibrator = ProbabilityCalibrator(solver=solver)
        exit_calibrator.fit(raw_exit_validation, exit_validation)
        testing["entry_probability"] = entry_calibrator.transform(raw_entry_test)
        testing["exit_probability"] = exit_calibrator.transform(raw_exit_test)
        # Reference probabilities depend only on the training partition.
        testing["entry_baseline_probability"] = float(np.mean(entry_train))
        testing["exit_baseline_probability"] = float(np.mean(exit_train))
        testing["walk_forward_fold"] = fold + 1
        fold_windows.append({
            "fold": fold + 1,
            "training_rows": len(training),
            "training_start": pd.Timestamp(training.iloc[0]["timestamp"]).isoformat(),
            "training_end": pd.Timestamp(training.iloc[-1]["timestamp"]).isoformat(),
            "training_last_label_available_at": (
                pd.Timestamp(training.iloc[-1]["label_available_at"]).isoformat()
                if "label_available_at" in training else None
            ),
            "validation_rows": len(validation),
            "validation_start": pd.Timestamp(validation.iloc[0]["timestamp"]).isoformat(),
            "validation_end": pd.Timestamp(validation.iloc[-1]["timestamp"]).isoformat(),
            "validation_last_label_available_at": (
                pd.Timestamp(validation.iloc[-1]["label_available_at"]).isoformat()
                if "label_available_at" in validation else None
            ),
            "test_rows": len(testing),
            "test_start": pd.Timestamp(testing.iloc[0]["timestamp"]).isoformat(),
            "test_end": pd.Timestamp(testing.iloc[-1]["timestamp"]).isoformat(),
            "purge_candles_at_each_boundary": purge_candles,
        })
        predictions.append(testing)
        progress(45 + int(28 * (fold + 1) / fold_count), f"Walk-forward fold {fold + 1}/{fold_count}")

    if not predictions:
        raise ValueError("No valid walk-forward test folds could be created from the database parameters.")

    out_of_sample = pd.concat(predictions, ignore_index=True).sort_values("timestamp")
    out_of_sample.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    average_importance = importance_accumulator / max(1, trained_models)
    feature_importance = sorted(
        [
            {"feature": feature, "importance": round(float(value), 6)}
            for feature, value in zip(columns, average_importance, strict=True)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )[: parameters.integer("walk_forward.feature_importance_limit")]
    return out_of_sample.reset_index(drop=True), {
        "fold_count": len(fold_windows),
        "fold_windows": fold_windows,
        "label_horizon_candles": purge_candles,
        "feature_importance": feature_importance,
        "feature_columns": columns,
    }


def _safe_model_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    parameters: RuntimeParameters,
    baseline_probabilities: np.ndarray | None = None,
) -> dict:
    epsilon = parameters.floating("simulation.probability_clip_epsilon")
    clipped = np.clip(probabilities, epsilon, 1 - epsilon)
    predicted = (clipped >= threshold).astype(int)
    both_classes = len(np.unique(labels)) > 1
    brier = float(brier_score_loss(labels, clipped))
    reference_brier = (
        float(brier_score_loss(labels, np.clip(
            baseline_probabilities, epsilon, 1 - epsilon
        )))
        if baseline_probabilities is not None else None
    )
    # Positive skill means better calibration than a training-only
    # event-frequency reference, evaluated on exactly the same OOS rows.
    brier_skill = (
        1.0 - brier / reference_brier
        if reference_brier is not None and reference_brier > 0 else None
    )
    return {
        "precision": round(float(precision_score(labels, predicted, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(labels, clipped)), 4) if both_classes else None,
        "roc_auc": round(float(roc_auc_score(labels, clipped)), 4) if both_classes else None,
        "log_loss": round(float(log_loss(labels, clipped, labels=[0, 1])), 4),
        "brier_score": round(brier, 4),
        "baseline_brier_score": round(reference_brier, 4) if reference_brier is not None else None,
        "brier_skill_score": round(brier_skill, 4) if brier_skill is not None else None,
        "event_prevalence_oos": round(float(np.mean(labels)), 6),
        "calibration": (
            "GOOD"
            if brier <= parameters.floating("simulation.calibration_good_brier")
            and (brier_skill is None or brier_skill > 0)
            else "REVIEW"
        ),
    }


def simulate_portfolio(
    frame: pd.DataFrame,
    request: ResolvedMLBacktestCreate,
    parameters: RuntimeParameters,
) -> dict:
    cash = request.initial_capital
    quantity = 0.0
    entry_price = 0.0
    entry_time = None
    position_allocation = 0.0
    stop_price = 0.0
    target_price = 0.0
    fees_paid = 0.0
    trades: list[dict] = []
    equity_rows: list[dict] = []
    rotation_rows: list[dict] = []
    exposed_bars = 0
    entry_candidates = 0
    entry_probability_passes = 0
    entry_trend_passes = 0
    entry_combined_passes = 0
    pending_action: str | None = None
    pending_reason: str | None = None

    atr_name = atr_column(parameters)
    _, medium_ema, slow_ema = signal_ema_columns(parameters)
    minimum_atr_fraction = parameters.floating("simulation.minimum_atr_price_fraction")
    initial_stop_atr = parameters.floating("simulation.initial_stop_atr")
    initial_target_atr = parameters.floating("simulation.initial_target_atr")
    trailing_stop_atr = parameters.floating("simulation.trailing_stop_atr")
    minimum_trend_gap = parameters.floating("simulation.minimum_trend_gap")

    first_close = float(frame.iloc[0]["close"])
    if first_close <= 0:
        raise ValueError("BTC close must be positive for the benchmark")
    # Benchmark purchases at first OOS close and values liquidation at
    # each close, applying the same fee and slippage rates as the strategy.
    buy_hold_quantity = (
        request.initial_capital * (1 - request.fee_rate)
        / (first_close * (1 + request.slippage_rate))
    )
    for index, row in frame.iterrows():
        bar_actions: list[dict] = []
        open_price = float(row["open"])
        close_price = float(row["close"])
        atr = max(float(row[atr_name]), close_price * minimum_atr_fraction)

        if pending_action == "BUY" and quantity == 0:
            allocation = cash if request.reinvestment else min(cash, request.initial_capital)
            execution = open_price * (1 + request.slippage_rate)
            entry_fee = allocation * request.fee_rate
            quantity = max(0.0, (allocation - entry_fee) / execution)
            cash -= allocation
            fees_paid += entry_fee
            position_allocation = allocation
            entry_price = execution
            entry_time = row["timestamp"]
            stop_price = entry_price - initial_stop_atr * atr
            target_price = entry_price + initial_target_atr * atr
            bar_actions.append({
                "action": "BUY", "execution_price": round(entry_price, 8),
                "reason": pending_reason or "LIGHTGBM_ENTRY",
            })
            pending_action = None
        elif pending_action == "SELL" and quantity > 0:
            execution = open_price * (1 - request.slippage_rate)
            gross = quantity * execution
            exit_fee = gross * request.fee_rate
            net_exit = gross - exit_fee
            cash += net_exit
            fees_paid += exit_fee
            pnl = net_exit - position_allocation
            trades.append(
                {
                    "entry_time": pd.Timestamp(entry_time).isoformat(),
                    "exit_time": pd.Timestamp(row["timestamp"]).isoformat(),
                    "entry_price": round(entry_price, 8),
                    "exit_price": round(execution, 8),
                    "net_pnl": round(pnl, 4),
                    "return_pct": round((net_exit / position_allocation - 1) * 100, 4),
                    "reason": pending_reason or "MODEL_EXIT",
                }
            )
            bar_actions.append({
                "action": "SELL", "execution_price": round(execution, 8),
                "reason": pending_reason or "MODEL_EXIT",
            })
            quantity = 0.0
            entry_price = 0.0
            entry_time = None
            position_allocation = 0.0
            pending_action = None
            pending_reason = None

        if quantity > 0:
            exposed_bars += 1
            intrabar_exit: tuple[float, str] | None = None
            if float(row["low"]) <= stop_price:
                intrabar_exit = (stop_price * (1 - request.slippage_rate), "PROTECTIVE_STOP")
            elif float(row["high"]) >= target_price:
                intrabar_exit = (target_price * (1 - request.slippage_rate), "TARGET")

            if intrabar_exit:
                execution, reason = intrabar_exit
                gross = quantity * execution
                exit_fee = gross * request.fee_rate
                net_exit = gross - exit_fee
                cash += net_exit
                fees_paid += exit_fee
                pnl = net_exit - position_allocation
                trades.append(
                    {
                        "entry_time": pd.Timestamp(entry_time).isoformat(),
                        "exit_time": pd.Timestamp(row["timestamp"]).isoformat(),
                        "entry_price": round(entry_price, 8),
                        "exit_price": round(execution, 8),
                        "net_pnl": round(pnl, 4),
                        "return_pct": round((net_exit / position_allocation - 1) * 100, 4),
                        "reason": reason,
                    }
                )
                bar_actions.append({
                    "action": "SELL", "execution_price": round(execution, 8),
                    "reason": reason,
                })
                quantity = 0.0
                entry_price = 0.0
                entry_time = None
                position_allocation = 0.0
                pending_action = None
                pending_reason = None
            else:
                stop_price = max(stop_price, close_price - trailing_stop_atr * atr)

        if index < len(frame) - 1:
            if quantity == 0 and pending_action is None:
                trend_ok = (
                    float(row[medium_ema]) > float(row[slow_ema])
                    and float(row["trend_ema_gap"]) > minimum_trend_gap
                )
                probability_ok = float(row["entry_probability"]) >= request.entry_threshold
                entry_candidates += 1
                entry_probability_passes += int(probability_ok)
                entry_trend_passes += int(trend_ok)
                entry_combined_passes += int(probability_ok and trend_ok)
                if probability_ok and trend_ok:
                    pending_action = "BUY"
                    pending_reason = "LIGHTGBM_ENTRY"
            elif quantity > 0 and pending_action is None:
                model_exit = float(row["exit_probability"]) >= request.exit_threshold
                trend_lost = close_price < float(row[medium_ema])
                if model_exit or trend_lost:
                    pending_action = "SELL"
                    pending_reason = "LIGHTGBM_EXIT" if model_exit else "TREND_INVALIDATION"

        equity = cash + quantity * close_price
        buy_hold = (
            buy_hold_quantity * close_price
            * (1 - request.slippage_rate) * (1 - request.fee_rate)
        )
        equity_rows.append(
            {
                "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                "equity": equity,
                "buy_hold": buy_hold,
            }
        )
        rotation_rows.append({
            "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
            "btc_close": close_price,
            "position": "BTC" if quantity > 0 else "CASH",
            "equity": equity,
            "buy_hold": buy_hold,
            "actions": bar_actions,
        })

    if quantity > 0:
        final_row = frame.iloc[-1]
        execution = float(final_row["close"]) * (1 - request.slippage_rate)
        gross = quantity * execution
        exit_fee = gross * request.fee_rate
        net_exit = gross - exit_fee
        cash += net_exit
        fees_paid += exit_fee
        pnl = net_exit - position_allocation
        trades.append(
            {
                "entry_time": pd.Timestamp(entry_time).isoformat(),
                "exit_time": pd.Timestamp(final_row["timestamp"]).isoformat(),
                "entry_price": round(entry_price, 8),
                "exit_price": round(execution, 8),
                "net_pnl": round(pnl, 4),
                "return_pct": round((net_exit / position_allocation - 1) * 100, 4),
                "reason": "END_OF_TEST",
            }
        )
        equity_rows[-1]["equity"] = cash
        rotation_rows[-1]["equity"] = cash
        rotation_rows[-1]["position"] = "CASH"
        rotation_rows[-1]["actions"].append({
            "action": "SELL", "execution_price": round(execution, 8),
            "reason": "END_OF_TEST",
        })

    return {
        "capital_final": cash,
        "fees_paid": fees_paid,
        "trades": trades,
        "equity_curve": equity_rows,
        "rotation_rows": rotation_rows,
        "exposure": exposed_bars / max(1, len(frame)),
        "signal_diagnostics": {
            "eligible_cash_candles": entry_candidates,
            "entry_probability_above_threshold": entry_probability_passes,
            "entry_trend_approved": entry_trend_passes,
            "entry_both_approved": entry_combined_passes,
            "entry_probability_max": float(frame["entry_probability"].max()),
            "entry_probability_mean": float(frame["entry_probability"].mean()),
            "buy_hold_fee_rate": request.fee_rate,
            "buy_hold_slippage_rate": request.slippage_rate,
            "benchmark_execution": "BUY_FIRST_OOS_CLOSE_SELL_LAST_OOS_CLOSE",
        },
    }


def _max_drawdown(values: np.ndarray) -> float:
    peaks = np.maximum.accumulate(values)
    drawdowns = values / np.where(peaks == 0, 1, peaks) - 1
    return float(np.min(drawdowns))


def calculate_results(
    frame: pd.DataFrame,
    request: ResolvedMLBacktestCreate,
    simulation: dict,
    model_metadata: dict,
    parameters: RuntimeParameters,
) -> dict:
    equity = np.array([row["equity"] for row in simulation["equity_curve"]], dtype=float)
    buy_hold_curve = np.array([row["buy_hold"] for row in simulation["equity_curve"]], dtype=float)
    # Compare drawdowns against the original capital before execution costs.
    strategy_drawdown = _max_drawdown(np.r_[request.initial_capital, equity])
    benchmark_drawdown = _max_drawdown(np.r_[request.initial_capital, buy_hold_curve])
    returns = pd.Series(equity).pct_change().dropna().to_numpy()
    downside = returns[returns < 0]
    periods_per_year = 365.25 * 86400 / TIMEFRAME_SECONDS[request.decision_timeframe]
    sharpe = (
        float(np.mean(returns) / np.std(returns) * math.sqrt(periods_per_year))
        if len(returns) > 2 and np.std(returns) > 0
        else 0.0
    )
    sortino = (
        float(np.mean(returns) / np.std(downside) * math.sqrt(periods_per_year))
        if len(downside) > 2 and np.std(downside) > 0
        else 0.0
    )
    start = pd.Timestamp(frame.iloc[0]["timestamp"])
    end = pd.Timestamp(frame.iloc[-1]["timestamp"])
    years = max((end - start).total_seconds() / (365.25 * 86400), 1 / 365.25)
    capital_final = float(simulation["capital_final"])
    total_return = capital_final / request.initial_capital - 1
    cagr = (capital_final / request.initial_capital) ** (1 / years) - 1 if capital_final > 0 else -1
    trades = simulation["trades"]
    wins = [trade for trade in trades if trade["net_pnl"] > 0]
    losses = [trade for trade in trades if trade["net_pnl"] <= 0]
    gross_profit = sum(trade["net_pnl"] for trade in wins)
    gross_loss = abs(sum(trade["net_pnl"] for trade in losses))

    entry_metrics = _safe_model_metrics(
        frame["entry_label"].to_numpy(int),
        frame["entry_probability"].to_numpy(float),
        request.entry_threshold,
        parameters,
        frame["entry_baseline_probability"].to_numpy(float)
        if "entry_baseline_probability" in frame else None,
    )
    exit_metrics = _safe_model_metrics(
        frame["exit_label"].to_numpy(int),
        frame["exit_probability"].to_numpy(float),
        request.exit_threshold,
        parameters,
        frame["exit_baseline_probability"].to_numpy(float)
        if "exit_baseline_probability" in frame else None,
    )

    maximum_points = parameters.integer("simulation.equity_curve_points")
    sample_step = max(1, len(simulation["equity_curve"]) // maximum_points)
    sampled_curve = simulation["equity_curve"][::sample_step]
    if sampled_curve[-1] != simulation["equity_curve"][-1]:
        sampled_curve.append(simulation["equity_curve"][-1])

    maximum_trades = parameters.integer("simulation.maximum_stored_trades")
    insight_features = parameters.integer("simulation.insight_top_features")
    return {
        "summary": {
            "capital_initial": request.initial_capital,
            "capital_final": round(capital_final, 4),
            "return_total": round(total_return, 6),
            "cagr": round(cagr, 6),
            "max_drawdown": round(strategy_drawdown, 6),
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else None,
            "trade_count": len(trades),
            "has_executed_trades": bool(trades),
            "win_rate": round(len(wins) / len(trades), 6) if trades else 0.0,
            "exposure": round(float(simulation["exposure"]), 6),
            "fees_paid": round(float(simulation["fees_paid"]), 4),
            "best_trade": max((trade["net_pnl"] for trade in trades), default=0.0),
            "worst_trade": min((trade["net_pnl"] for trade in trades), default=0.0),
        },
        "execution": {
            "candles_used": len(frame),
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
            "duration_years": round(years, 3),
            "walk_forward_folds": model_metadata["fold_count"],
            "decision_timeframe": request.decision_timeframe,
            "trend_timeframe": request.trend_timeframe,
            "model_family": "LightGBM",
            "parameter_source": "DATABASE_SNAPSHOT",
            "fold_windows": model_metadata.get("fold_windows", []),
            "label_horizon_candles": model_metadata.get("label_horizon_candles"),
            "history_coverage": model_metadata.get("history_coverage"),
            "history_source": request.history_source,
            "market_interruptions": model_metadata.get("market_interruptions", {}),
            "label_vs_execution": model_metadata.get("target_vs_simulation"),
        },
        "models": {
            "entry": entry_metrics,
            "exit": exit_metrics,
            "feature_importance": model_metadata["feature_importance"],
        },
        "comparisons": [
            {
                "code": "LIGHTGBM_WALK_FORWARD",
                "name": "LightGBM Walk-Forward",
                "capital_final": round(capital_final, 4),
                "return_total": round(total_return, 6),
                "max_drawdown": round(strategy_drawdown, 6),
                "sharpe": round(sharpe, 4),
            },
            {
                "code": "BUY_AND_HOLD",
                "name": "Buy and Hold",
                "capital_final": round(float(buy_hold_curve[-1]), 4),
                "return_total": round(float(buy_hold_curve[-1] / request.initial_capital - 1), 6),
                "max_drawdown": round(benchmark_drawdown, 6),
                "sharpe": None,
                "benchmark_costs_included": True,
            },
        ],
        "trades": trades[-maximum_trades:],
        "equity_curve": sampled_curve,
        "rotation_timeline": build_rotation_timeline(
            simulation.get("rotation_rows", []), maximum_points
        ),
        "insights": {
            "top_features": [
                item["feature"] for item in model_metadata["feature_importance"][:insight_features]
            ],
            "entry_threshold": request.entry_threshold,
            "exit_threshold": request.exit_threshold,
            "beats_buy_and_hold": capital_final > float(buy_hold_curve[-1]),
            "strategy_has_trades": bool(trades),
            "no_trade_reason": (
                "ENTRY_PROBABILITY_BELOW_THRESHOLD"
                if not trades and simulation.get("signal_diagnostics", {}).get("entry_probability_above_threshold") == 0
                else "ENTRY_FILTERS_NOT_SIMULTANEOUSLY_APPROVED"
                if not trades and simulation.get("signal_diagnostics", {}).get("entry_both_approved") == 0
                else "NO_COMPLETED_TRADES" if not trades else None
            ),
            "signal_diagnostics": simulation.get("signal_diagnostics", {}),
            "benchmark_costs_included": True,
        },
    }


def run_ml_backtest(
    request: ResolvedMLBacktestCreate,
    parameters: RuntimeParameters,
    progress: Callable[[int, str], None],
    cancelled: Callable[[], bool],
    audit_writer: Callable[[bytes], None] | None = None,
) -> dict:
    client = (
        BinancePublicArchiveClient(parameters, __version__)
        if request.history_source == "BINANCE_SPOT_ARCHIVE"
        else MEXCHistoryClient(parameters, __version__)
    )
    try:
        progress(3, "Downloading decision-timeframe history")
        decision = client.fetch_history(
            symbol=request.symbol,
            timeframe=request.decision_timeframe,
            period_years=request.period_years,
            progress=lambda value: progress(3 + int(value * 0.10), "Downloading decision-timeframe history"),
            cancelled=cancelled,
        )
        if request.trend_timeframe == request.decision_timeframe:
            trend = decision
        else:
            progress(14, "Downloading trend-timeframe history")
            trend = client.fetch_history(
                symbol=request.symbol,
                timeframe=request.trend_timeframe,
                period_years=request.period_years,
                progress=lambda value: progress(14 + int(value * 0.07), "Downloading trend-timeframe history"),
                cancelled=cancelled,
            )
        progress(24, "Calculating technical indicators")
        featured = build_features(
            decision, parameters, trend,
            decision_timeframe=request.decision_timeframe,
            trend_timeframe=request.trend_timeframe,
        )
        progress(32, "Creating entry and exit labels")
        labelled = create_labels(featured, parameters)
        progress(40, "Preparing walk-forward validation")
        predicted, metadata = walk_forward_predict(labelled, parameters, progress, cancelled)
        expected_seconds = TIMEFRAME_SECONDS[request.decision_timeframe]
        observed = pd.to_datetime(decision["timestamp"], utc=True)
        interval_counts = observed.diff().dropna().dt.total_seconds() / expected_seconds
        missing_intervals = int((interval_counts.round().clip(lower=1) - 1).sum())
        requested_candles = int(math.ceil(
            request.period_years * 365.25 * 86400 / expected_seconds
        ))
        raw_start = pd.Timestamp(observed.iloc[0])
        raw_end = pd.Timestamp(observed.iloc[-1])
        available_intervals = int(round(
            (raw_end - raw_start).total_seconds() / expected_seconds
        )) + 1
        observed_coverage = min(1.0, available_intervals / max(1, requested_candles))
        metadata["history_source"] = request.history_source
        metadata["source_archives"] = getattr(client, "archive_manifest", None)
        metadata["market_interruptions"] = getattr(client, "market_interruptions", {})
        documented_halt_candles = sum(
            int(event["missing_candles"])
            for event in metadata["market_interruptions"].get(
                request.decision_timeframe, []
            )
        )
        coverage_status = (
            "FULL"
            if observed_coverage >= 0.99 and missing_intervals == 0
            else "FULL_WITH_DOCUMENTED_HALT"
            if observed_coverage >= 0.99
            and documented_halt_candles > 0
            and missing_intervals == documented_halt_candles
            else "PARTIAL"
        )
        metadata["history_coverage"] = {
            "source": request.history_source,
            "requested_years": request.period_years,
            "requested_candles": requested_candles,
            "max_candles_parameter": parameters.integer("market.max_candles"),
            "raw_candles": len(decision),
            "raw_start": raw_start.isoformat(),
            "raw_end": raw_end.isoformat(),
            "raw_available_intervals": available_intervals,
            "missing_intervals": missing_intervals,
            "documented_market_halt_candles": documented_halt_candles,
            "market_interruptions": metadata["market_interruptions"],
            "max_observed_interval_gap": float(interval_counts.max()) if len(interval_counts) else 0.0,
            "requested_period_coverage_ratio": round(observed_coverage, 6),
            "coverage_status": coverage_status,
            "oos_candles": len(predicted),
            "oos_start": pd.Timestamp(predicted.iloc[0]["timestamp"]).isoformat(),
            "oos_end": pd.Timestamp(predicted.iloc[-1]["timestamp"]).isoformat(),
        }
        metadata["target_vs_simulation"] = {
            "label_horizon_candles": parameters.integer("labels.horizon_candles"),
            "label_entry_target_atr": parameters.floating("labels.entry_target_atr"),
            "label_entry_stop_atr": parameters.floating("labels.entry_stop_atr"),
            "simulation_initial_target_atr": parameters.floating("simulation.initial_target_atr"),
            "simulation_initial_stop_atr": parameters.floating("simulation.initial_stop_atr"),
        }
        progress(80, "Simulating out-of-sample operations")
        simulation = simulate_portfolio(predicted, request, parameters)
        progress(92, "Calculating financial and model metrics")
        result = calculate_results(predicted, request, simulation, metadata, parameters)
        if audit_writer is not None:
            progress(95, "Preparing full audit export")
            audit_writer(build_audit_archive(
                predictions=predicted,
                simulation=simulation,
                request=request,
                parameters=parameters,
                result=result,
                model_metadata=metadata,
                decision_history=decision,
                trend_history=trend,
            ))
        progress(100, "Completed")
        return result
    finally:
        client.close()
