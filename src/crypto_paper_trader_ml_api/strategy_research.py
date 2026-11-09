"""MCT-style BTC research: saved strategies, bounded tuning and audit reports.

Tuning uses ONLY candles before the first archived OOS test boundary.
The untouched OOS period is evaluated later by the existing walk-forward job.
No scripts, filesystem paths, brokerage calls, live orders or exchange downloads.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import BytesIO, StringIO
import json
import math
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sqlalchemy import desc

from .archive_replay import ArchivedMarketClient, inspect_source_archive
from .database import MLBacktestAudit, MLBacktestJob, MLResearchStrategy, SessionLocal
from .direct_forecast import create_direction_labels
from .features import build_features, feature_names
from .lightgbm_models import fit_classifier, fit_regressor
from .mexc import TIMEFRAME_SECONDS
from .settings_service import get_runtime_parameters, get_runtime_parameters_from_snapshot, RuntimeParameters
from .engine import ProbabilityCalibrator
from .btc_policy import DEFAULT_POLICY, aggregate_signals, simulate_rotations, validate_policy
from .alpaca_fees import ALPACA_FEE_MODEL, AlpacaFeeLedger, round_trip_cost
from .frozen_cost_replay import replay_frozen_rotations
from .forecast_quality import describe_return_predictions
from .btc_event_research import evaluate_archived_event_research, POLICIES as BTC_EVENT_POLICIES
from .btc_directional_change import evaluate_archived_directional_change, THRESHOLDS as BTC_DC_THRESHOLDS
from .btc_technical_research import evaluate_archived_technical_research, POLICIES as BTC_TECHNICAL_POLICIES
from .btc_trend_regimes import evaluate_archived_trend_regimes, POLICIES as BTC_INDICATOR_POLICIES, V14_POLICIES as BTC_TREND_POLICIES
from .selection_robustness import (
    assess_robustness, STRESS_FEE_MULTIPLIER, STRESS_SLIPPAGE_MULTIPLIER,
    MIN_COMPLETED_CYCLES, MIN_ACTIVE_SEGMENTS, MIN_PROFITABLE_SEGMENTS,
)


# Expose an intentionally small, well-bounded search space; optimize the
# decision-policy thresholds independently from tree topology in a later run.
SPACE = (
    ("lightgbm.max_depth", "int", 3, 6),
    ("lightgbm.num_leaves", "int", 7, 31),
    ("lightgbm.min_child_samples", "int", 15, 100),
    ("lightgbm.learning_rate", "float", 0.015, 0.12),
    ("lightgbm.reg_lambda", "float", 0.0, 4.0),
    ("forecast.regression_num_leaves", "int", 5, 15),
    ("forecast.regression_learning_rate", "float", 0.015, 0.12),
)
MAX_DEV_CANDLES = 48000
POLICY_SPACE = (
    ("policy.entry_probability", "float", 0.30, 0.60),
    ("policy.exit_probability", "float", 0.12, 0.45),
    ("policy.entry_edge", "float", 0.0, 0.002),
    ("policy.exit_return", "float", -0.018, 0.004),
    ("policy.min_positive_votes", "int", 1, 3),
    ("policy.stop_loss", "float", 0.012, 0.18),
    ("policy.trailing_stop", "float", 0.012, 0.16),
    ("policy.take_profit", "float", 0.02, 0.24),
    ("policy.min_hold_bars", "int", 0, 24),
    ("policy.cooldown_bars", "int", 0, 24),
    ("policy.weight_2h", "float", 0.1, 1.0),
    ("policy.weight_6h", "float", 0.1, 1.0),
    ("policy.weight_24h", "float", 0.1, 1.0),
)
# Each candidate jointly changes the forecasters and their BTC/CASH rotation policy.
SPACE = SPACE + POLICY_SPACE
MAX_TRIALS = 40
JOINT_HORIZONS = (2, 6, 24)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _decode(row: MLResearchStrategy) -> dict:
    return {
        "id": row.id, "name": row.name, "source_job_id": row.source_job_id,
        "horizon_hours": row.horizon_hours,
        "horizons_hours": list(JOINT_HORIZONS if row.horizon_hours == 0 else (row.horizon_hours,)),
        "status": row.status, "mode": row.mode,
        "progress": row.progress, "trials_target": row.trials_target,
        "trials": json.loads(row.trials_json or "[]"),
        "champion": json.loads(row.champion_json) if row.champion_json else None,
        "summary": json.loads(row.summary_json) if row.summary_json else None,
        "execution_job_id": row.execution_job_id,
        "error_message": row.error_message,
        "created_at": row.created_at.isoformat(),
        "updated_at": row.updated_at.isoformat(),
    }


def list_strategies() -> list[dict]:
    with SessionLocal() as session:
        rows = session.query(MLResearchStrategy).order_by(
            desc(MLResearchStrategy.created_at)
        ).limit(100).all()
        return [_decode(row) for row in rows]


def get_strategy(strategy_id: str) -> dict:
    with SessionLocal() as session:
        row = session.get(MLResearchStrategy, strategy_id)
        if row is None:
            raise LookupError("Estratégia de pesquisa não encontrada")
        return _decode(row)


def create_strategy(*, name: str, source_job_id: str, horizon_hours: int = 0) -> dict:
    with SessionLocal.begin() as session:
        job = session.get(MLBacktestJob, source_job_id)
        if job is None or job.status != "COMPLETED":
            raise ValueError("Selecione uma execução BTC concluída com auditoria")
        audit = session.get(MLBacktestAudit, source_job_id)
        if audit is None:
            raise ValueError("A execução selecionada não possui auditoria armazenada na ML API")
        source_request, source_settings = inspect_source_archive(audit.archive_zip)
        available = {int(x) for x in source_settings["forecast.horizons_hours"]}
        selected = JOINT_HORIZONS if horizon_hours == 0 else (horizon_hours,)
        missing = set(selected) - available
        if missing:
            raise ValueError(
                f"A auditoria selecionada não contém os horizontes {sorted(missing)}h; "
                "escolha uma execução com previsões de 2h, 6h e 24h."
            )
        with ZipFile(BytesIO(audit.archive_zip)) as archive:
            fold_horizons = {
                int(fold["horizon_hours"]) for fold in json.loads(archive.read("folds.json"))
            }
        missing_folds = set(selected) - fold_horizons
        if missing_folds:
            raise ValueError(
                f"A auditoria não contém folds OOS de {sorted(missing_folds)}h. "
                "Selecione outra execução com os três horizontes concluídos."
            )
        row = MLResearchStrategy(
            id=str(uuid4()), name=name.strip(), source_job_id=source_job_id,
            horizon_hours=horizon_hours, status="DRAFT", progress=0, trials_json="[]",
        )
        session.add(row)
        session.flush()
        return _decode(row)


def _candidate(point: np.ndarray) -> dict:
    settings = {}
    for (key, kind, lower, upper), unit in zip(SPACE, point, strict=True):
        val = lower + float(np.clip(unit, 0, 1)) * (upper - lower)
        settings[key] = int(round(val)) if kind == "int" else round(val, 6)
    settings["lightgbm.num_leaves"] = min(
        settings["lightgbm.num_leaves"], 2 ** settings["lightgbm.max_depth"]
    )
    settings["forecast.regression_num_leaves"] = min(
        settings["forecast.regression_num_leaves"], 16
    )
    settings["policy.exit_probability"] = min(
        settings["policy.exit_probability"],
        round(settings["policy.entry_probability"] - 0.02, 6),
    )
    settings["policy.enabled"] = True
    # Freeze the actual Alpaca fee model with each champion. Older strategy
    # snapshots (without this key) keep their historic fixed fee semantics.
    settings["policy.fee_model"] = ALPACA_FEE_MODEL
    settings["policy.alpaca_liquidity"] = "TAKER"
    settings["policy.alpaca_initial_30d_volume_usd"] = 0.0
    validate_policy(settings)
    return settings


def _next_candidate(
    index: int, mode: str, points: list[np.ndarray], scores: list[float],
    sampler: qmc.LatinHypercube, rng: np.random.Generator,
    observations: list[dict] | None = None,
) -> np.ndarray:
    """Crypto CARO: initial space filling, then risk-constrained GP search.

    Adapted from the MCT's unified exploration/refinement idea. It uses
    scikit-learn GPs, not the MCT's BoTorch four-outcome CARO engine.
    """
    if index == 0:
        # Fixed permissive research anchor, defined BEFORE seeing OOS data.
        # This separates genuinely untradable forecasts from sparse sampling
        # across a 20-dimensional hyperparameter search.
        seed = np.full(len(SPACE), 0.5, dtype=float)
        keys = [entry[0] for entry in SPACE]
        for name in (
            "policy.entry_probability", "policy.entry_edge",
            "policy.min_positive_votes",
        ):
            seed[keys.index(name)] = 0.0
        return seed
    if mode != "BAYESIAN" or len(points) < 8 or not observations:
        return sampler.random(1)[0]
    valid = [
        (point, row["metrics"], row["metrics"]["score"])
        for point, row in zip(points, observations, strict=True)
        if row.get("status") == "COMPLETED"
        and all(np.isfinite(float(row["metrics"].get(key, np.nan)))
                for key in ("capital_final", "sharpe", "max_drawdown", "worst_fold_return", "score"))
    ]
    if len(valid) < 8:
        return sampler.random(1)[0]
    try:
        best_point, benchmark, _ = max(valid, key=lambda item: item[2])
        x = np.stack([item[0] for item in valid])
        targets = (
            "capital_final", "sharpe", "max_drawdown", "worst_fold_return"
        )
        # Global exploration remains available, even after a strong champion.
        # Trust-region refinement shrinks when recent trials improve.
        history = [item[2] for item in valid]
        improvement_recently = max(history[-4:]) > max(history[:-4], default=-np.inf)
        radius = 0.15 if improvement_recently else 0.30
        local = np.clip(
            rng.normal(loc=best_point, scale=radius, size=(224, len(SPACE))),
            0.0, 1.0,
        )
        global_pool = rng.uniform(size=(128, len(SPACE)))
        pool = np.vstack([local, global_pool])
        posterior = {}
        for key in targets:
            kernel = ConstantKernel(1.0) * Matern(
                length_scale=np.ones(len(SPACE)), nu=2.5
            ) + WhiteKernel(noise_level=0.001)
            gp = GaussianProcessRegressor(
                kernel=kernel, random_state=42, normalize_y=True,
                alpha=1e-6, n_restarts_optimizer=0,
            )
            gp.fit(x, np.array([item[1][key] for item in valid], dtype=float))
            mean, uncertainty = gp.predict(pool, return_std=True)
            posterior[key] = (mean, np.maximum(uncertainty, 1e-9))

        cap_mean, cap_std = posterior["capital_final"]
        min_improvement = float(benchmark["capital_final"]) * 0.005
        threshold = float(benchmark["capital_final"]) + min_improvement
        delta = cap_mean - threshold
        z = delta / cap_std
        expected_improvement = delta * norm.cdf(z) + cap_std * norm.pdf(z)
        prob_profit = norm.sf((threshold - cap_mean) / cap_std)
        sharpe_mean, sharpe_std = posterior["sharpe"]
        dd_mean, dd_std = posterior["max_drawdown"]
        fold_mean, fold_std = posterior["worst_fold_return"]
        # Higher is better for all: max drawdown is a negative fraction.
        prob_risk = (
            norm.sf((float(benchmark["sharpe"]) - 0.05 - sharpe_mean) / sharpe_std)
            * norm.sf((float(benchmark["max_drawdown"]) - 0.03 - dd_mean) / dd_std)
            * norm.sf((float(benchmark["worst_fold_return"]) - 0.03 - fold_mean) / fold_std)
        )
        distance = np.linalg.norm(pool[:, None, :] - x[None, :, :], axis=2).min(axis=1)
        # Constrained expected capital improvement, with a small exploration
        # floor and novelty bonus to avoid getting stuck at the champion.
        utility = (
            expected_improvement / 10000.0 * prob_risk
            + 0.015 * prob_profit * prob_risk
            + 0.003 * cap_std / 10000.0
            + 0.003 * np.minimum(distance, 1.0)
        )
        utility[distance < 1e-5] = -np.inf
        position = int(np.argmax(utility))
        if not np.isfinite(utility[position]):
            return sampler.random(1)[0]
        return pool[position]
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return sampler.random(1)[0]


def _prepare(
    source_archive: bytes, horizon: int, parameters: RuntimeParameters,
    oos_cutoff: pd.Timestamp | None = None,
):
    """Freeze training strictly before the FIRST archived OOS test date."""
    request, _ = inspect_source_archive(source_archive)
    with ZipFile(BytesIO(source_archive)) as archive:
        windows = json.loads(archive.read("folds.json"))
    first_oos = min(
        pd.Timestamp(fold["test_start"])
        for fold in windows if int(fold["horizon_hours"]) == horizon
    )
    if oos_cutoff is not None:
        first_oos = min(first_oos, oos_cutoff)
    client = ArchivedMarketClient(source_archive, request)
    decision = client.fetch_history(
        symbol=request.symbol, timeframe=request.decision_timeframe,
        period_years=request.period_years,
    )
    trend = client.fetch_history(
        symbol=request.symbol, timeframe=request.trend_timeframe,
        period_years=request.period_years,
    )
    featured = build_features(
        decision, parameters, trend,
        decision_timeframe=request.decision_timeframe,
        trend_timeframe=request.trend_timeframe,
    )
    labelled = create_direction_labels(featured, request.decision_timeframe, horizon)
    cols = feature_names(parameters)
    usable = labelled.dropna(subset=[
        *cols, "forecast_realized_return", "forecast_realized_up",
        "forecast_future_at_utc",
    ]).copy()
    # Labels themselves must mature before the untouched OOS period begins.
    usable = usable.loc[
        (pd.to_datetime(usable["timestamp"], utc=True) < first_oos)
        & (pd.to_datetime(usable["forecast_future_at_utc"], utc=True) < first_oos)
    ].tail(MAX_DEV_CANDLES).reset_index(drop=True)
    if len(usable) < 900:
        raise ValueError("Não há dados suficientes antes do primeiro teste OOS para otimização")
    h = horizon * 3600 // TIMEFRAME_SECONDS[request.decision_timeframe]
    n = len(usable)
    # Purged internal research partitions: training / early stop /
    # calibration / scoring. Scoring samples are never training samples.
    # A longer independent PRE-OOS economic selection window across
    # market regimes, with purges between fitting and selection.
    start_early = int(n * .52)
    start_cal = int(n * .65)
    start_score = int(n * .75)
    train = usable.iloc[:start_early - h].copy()
    early = usable.iloc[start_early:start_cal - h].copy()
    calibration = usable.iloc[start_cal:start_score - h].copy()
    scoring = usable.iloc[start_score:].copy()
    sections = [train, early, calibration, scoring]
    for prior, following in zip(sections, sections[1:]):
        boundary = pd.Timestamp(following.iloc[0]["timestamp"])
        prior.drop(
            index=prior.index[
                pd.to_datetime(prior["forecast_future_at_utc"], utc=True) >= boundary
            ],
            inplace=True,
        )
    if min(len(part) for part in sections) < 80:
        raise ValueError("Janelas cronológicas de otimização insuficientes após o purge")
    return train, early, calibration, scoring, cols, request, first_oos


def _simulate(
    scoring: pd.DataFrame, probabilities: np.ndarray, forecasts: np.ndarray,
    fee: float, slippage: float, timeframe: str,
) -> dict:
    """Causal validation portfolio: signal at CLOSE, fill at NEXT OPEN."""
    open_ = scoring["open"].to_numpy(float)
    close = scoring["close"].to_numpy(float)
    timestamps = pd.to_datetime(scoring["timestamp"], utc=True)
    step = TIMEFRAME_SECONDS[timeframe]
    required_edge = (1 + slippage) / ((1 - fee)**2 * (1 - slippage)) - 1
    equity, btc = 10000.0, 0.0
    peak, maxdd, buys, sells = equity, 0.0, 0, 0
    curve = [equity]
    for index in range(1, len(scoring)):
        if (timestamps.iloc[index] - timestamps.iloc[index - 1]).total_seconds() == step:
            estimate = float(forecasts[index - 1])
            p_up = float(probabilities[index - 1])
            if btc == 0 and estimate > required_edge and p_up >= 0.5:
                btc = equity * (1 - fee) / (open_[index] * (1 + slippage))
                equity = 0.0
                buys += 1
            elif btc > 0 and (estimate <= 0 or p_up < 0.5):
                equity = btc * open_[index] * (1 - slippage) * (1 - fee)
                btc = 0.0
                sells += 1
        value = equity + btc * close[index]
        peak = max(value, peak)
        maxdd = min(maxdd, value / peak - 1)
        curve.append(value)
    if btc > 0:
        equity = btc * close[-1] * (1 - slippage) * (1 - fee)
        sells += 1
        curve[-1] = equity
    returns = np.diff(np.asarray(curve, dtype=float)) / np.maximum(1e-9, curve[:-1])
    std = float(np.std(returns)) if len(returns) else 0.0
    sharpe = float(np.mean(returns) / std * np.sqrt(365.25 * 48)) if std > 1e-12 else 0.0
    segments = np.array_split(np.arange(len(curve)), 3)
    fold_returns = [float(curve[segment[-1]] / curve[segment[0]] - 1)
                    for segment in segments if len(segment) > 1]
    worst_fold = min(fold_returns) if fold_returns else 0.0
    return {
        "capital_final": float(equity), "return_total": float(equity / 10000 - 1),
        "max_drawdown": float(maxdd), "buys": buys, "sells": sells,
        "sharpe": sharpe, "worst_fold_return": float(worst_fold),
    }


def _score(
    dataset: tuple, settings: dict, base: RuntimeParameters,
) -> dict:
    train, early, calibration, scoring, columns, request, _ = dataset
    parameters = RuntimeParameters({**base.snapshot(), **settings})
    classifier = fit_classifier(
        train[columns], train["forecast_realized_up"].to_numpy(int),
        early[columns], early["forecast_realized_up"].to_numpy(int), parameters,
    )
    regressor = fit_regressor(train, early, columns, parameters)
    baseline = float(train["forecast_realized_up"].mean())
    if classifier is None:
        raw_cal = np.full(len(calibration), baseline)
        raw_score = np.full(len(scoring), baseline)
    else:
        raw_cal = classifier.predict_proba(calibration[columns])[:, 1]
        raw_score = classifier.predict_proba(scoring[columns])[:, 1]
    calibrator = ProbabilityCalibrator(
        solver=parameters.string("lightgbm.calibration_solver")
    )
    calibrator.fit(raw_cal, calibration["forecast_realized_up"].to_numpy(int))
    probs = np.clip(calibrator.transform(raw_score), 1e-7, 1 - 1e-7)
    scale = parameters.floating("forecast.regression_target_scale")
    forecasts = (
        regressor.predict(scoring[columns]) / scale if regressor is not None
        else np.full(len(scoring), float(train["forecast_realized_return"].mean()))
    )
    y = scoring["forecast_realized_up"].to_numpy(int)
    actual = scoring["forecast_realized_return"].to_numpy(float)
    brier = float(brier_score_loss(y, probs))
    mae = float(mean_absolute_error(actual, forecasts))
    magnitude_quality = describe_return_predictions(
        forecasts, actual,
        training_baseline_return=float(train["forecast_realized_return"].mean()),
        # ResolvedMLBacktestCreate has no scalar horizon_hours; the archived
        # label timestamp is the ground truth for this prepared horizon.
        horizon_hours=int(round(
            (pd.Timestamp(scoring["forecast_future_at_utc"].iloc[0])
             - pd.Timestamp(scoring["timestamp"].iloc[0])).total_seconds() / 3600
        )),
    )
    capital = _simulate(
        scoring, probs, forecasts, request.fee_rate, request.slippage_rate,
        request.decision_timeframe,
    )
    # The selection objective is economic. Predictive errors are informative
    # tie-breakers; never fit/tune on the final OOS test.
    score = (
        math.log(max(1e-9, capital["capital_final"] / 10000))
        + 0.5 * capital["max_drawdown"]
        + 0.01 * np.clip(capital["sharpe"], -4, 4)
        + 0.3 * min(0.0, capital["worst_fold_return"])
        - 0.03 * brier
        - 0.03 * mae
        - (0.01 if capital["buys"] == 0 else 0)
    )
    predictions = {}
    for row, p, estimate in zip(scoring.itertuples(index=False), probs, forecasts, strict=True):
        predictions[pd.Timestamp(row.timestamp).isoformat()] = {
            "predicted_return": float(estimate), "p_up": float(p),
        }
    return {
        "score": float(score),
        "brier": brier, "mae": mae, **capital,
        "__return_quality": magnitude_quality,
        "__calibration": {
            "method": calibrator.method,
            "selection_brier": calibrator.selection_brier,
            "selection_rows": calibrator.selection_rows,
            "calibration_rows": calibrator.calibration_rows,
            "calibration_up_rate": float(calibration["forecast_realized_up"].mean()),
            "scoring_mean_p_up": float(np.mean(probs)),
            "scoring_max_p_up": float(np.max(probs)),
        },
        "__policy_predictions": predictions,
    }


def _entry_diagnostics(joint_signals: dict, cfg: dict, fee: float, slippage: float) -> dict:
    """Explain the opportunity funnel without using any realized returns."""
    if cfg.get("policy.fee_model") == ALPACA_FEE_MODEL:
        quoted = AlpacaFeeLedger(
            cfg["policy.alpaca_liquidity"],
            cfg["policy.alpaca_initial_30d_volume_usd"]
        ).quote(pd.Timestamp("2024-01-01", tz="UTC").to_pydatetime())
        fee = quoted["rate"]
    required_cost = round_trip_cost(fee, fee, slippage)
    samples = list(joint_signals.values())
    probabilities = np.asarray([float(row["p_up"]) for row in samples], dtype=float)
    predicted_returns = np.asarray([float(row["predicted_return"]) for row in samples], dtype=float)
    eligible = sum(int(
        row["positive_votes"] >= cfg["policy.min_positive_votes"]
        and row["p_up"] >= cfg["policy.entry_probability"]
        and row["predicted_return"] > required_cost + cfg["policy.entry_edge"]
    ) for row in samples)
    return {
        "total_aligned_signals": len(samples),
        "required_round_trip_return": float(required_cost),
        "required_entry_return": float(required_cost + cfg["policy.entry_edge"]),
        "passes_votes": sum(
            int(s["positive_votes"] >= cfg["policy.min_positive_votes"]) for s in samples
        ),
        "passes_probability": sum(
            int(s["p_up"] >= cfg["policy.entry_probability"]) for s in samples
        ),
        "passes_expected_return": sum(
            int(s["predicted_return"] > required_cost + cfg["policy.entry_edge"])
            for s in samples
        ),
        "passes_all_entry_filters": eligible,
        "max_probability": float(np.max(probabilities)) if len(samples) else None,
        "p90_probability": float(np.quantile(probabilities, .90)) if len(samples) else None,
        "max_predicted_return": float(np.max(predicted_returns)) if len(samples) else None,
        "p90_predicted_return": float(np.quantile(predicted_returns, .90)) if len(samples) else None,
        "diagnosis": (
            "NO_ALIGNED_SIGNALS" if not samples else
            "RETURN_FILTER_BLOCKED" if not any(
                s["predicted_return"] > required_cost + cfg["policy.entry_edge"]
                for s in samples
            ) else "PROBABILITY_OR_VOTE_BLOCKED" if eligible == 0
            else "HAS_ENTRY_OPPORTUNITIES"
        ),
    }




def _forecast_realization_diagnostics(
    candles: list[dict], signals: dict[int, dict], policy: dict,
    fee: float, slippage: float, bar_seconds: int,
) -> dict:
    """Compare predicted 6-hour equivalent return against observed 6-hour return.

    This is a historical SELECTION-WINDOW diagnostic, not a forecast feature
    and never an input to the live or final OOS decision logic. The last
    12-candle future path must be fully inside the pre-OOS scoring dataset.
    """
    if bar_seconds != 1800:
        horizon_bars = round(6 * 3600 / bar_seconds)
    else:
        horizon_bars = 12
    cfg = validate_policy(policy)
    predicted, realized, eligible = [], [], []
    initial_fee = (
        AlpacaFeeLedger(
            cfg["policy.alpaca_liquidity"],
            cfg["policy.alpaca_initial_30d_volume_usd"]
        ).quote(candles[0]["timestamp"])["rate"]
        if cfg.get("policy.fee_model") == ALPACA_FEE_MODEL else fee
    )
    hurdle = round_trip_cost(initial_fee, initial_fee, slippage) + cfg["policy.entry_edge"]
    for i, signal in signals.items():
        stop = i + horizon_bars - 1
        if i < 1 or stop >= len(candles):
            continue
        if (candles[stop]["timestamp"] - candles[i - 1]["timestamp"]).total_seconds() != horizon_bars * bar_seconds:
            continue
        first_close = float(candles[i - 1]["close"])
        last_close = float(candles[stop]["close"])
        if first_close <= 0 or last_close <= 0:
            continue
        prediction = float(signal["predicted_return"])
        outcome = last_close / first_close - 1
        predicted.append(prediction)
        realized.append(outcome)
        eligible.append(
            signal["positive_votes"] >= cfg["policy.min_positive_votes"]
            and signal["p_up"] >= cfg["policy.entry_probability"]
            and prediction > hurdle
        )
    if not predicted:
        return {"rows": 0, "horizon_hours": 6, "note": "Não há caminhos futuros completos dentro da validação pré-OOS"}
    p, y, mask = np.asarray(predicted), np.asarray(realized), np.asarray(eligible)
    correlation = (
        float(np.corrcoef(p, y)[0, 1]) if len(p) >= 3
        and np.std(p) > 1e-12 and np.std(y) > 1e-12 else None
    )
    return {
        "rows": len(p), "horizon_hours": 6,
        "mean_predicted_return": float(np.mean(p)),
        "mean_realized_return": float(np.mean(y)),
        "mean_prediction_bias": float(np.mean(p - y)),
        "return_mae": float(np.mean(np.abs(p - y))),
        "return_correlation": correlation,
        "direction_accuracy": float(np.mean((p > 0) == (y > 0))),
        "entry_eligible_rows": int(np.count_nonzero(mask)),
        "entry_eligible_mean_predicted_return": float(np.mean(p[mask])) if np.any(mask) else None,
        "entry_eligible_mean_realized_return": float(np.mean(y[mask])) if np.any(mask) else None,
        "note": "Descritivo: retorno 6h real só é observado após a previsão e nunca participa dos sinais.",
    }




def _combine_horizon_metrics(
    results: dict[str, dict],
    *, scoring: pd.DataFrame | None = None, policy: dict | None = None,
    fee: float = .001, slippage: float = .0005, bar_seconds: int = 1800,
) -> dict:
    """Score ONE fully reinvested multi-horizon portfolio, not mean of accounts."""
    if not results:
        raise ValueError("Nenhuma previsão disponível para otimização")
    if len(results) == 1:
        row = next(iter(results.values()))
        clean = {k: v for k, v in row.items() if not k.startswith("__")}
        return {**clean, "per_horizon": {"2": clean}, "score_method": "single_legacy"}
    if scoring is None or policy is None or len(results) != 3:
        raise ValueError("A estratégia conjunta exige os três horizontes e candles alinhados")
    cfg = validate_policy(policy)
    candles, indices = [], {}
    for i, row in enumerate(scoring.itertuples(index=False)):
        stamp = pd.Timestamp(row.timestamp)
        indices[stamp.isoformat()] = i
        candles.append({"timestamp": stamp.to_pydatetime(),
                        "open": float(row.open), "close": float(row.close)})
    signals_by_horizon = {}
    for h in JOINT_HORIZONS:
        horizon_signals = {}
        for timestamp, prediction in results[str(h)]["__policy_predictions"].items():
            prior = indices.get(timestamp)
            if prior is None or prior + 1 >= len(candles):
                continue
            nxt = prior + 1
            if (candles[nxt]["timestamp"] - candles[prior]["timestamp"]).total_seconds() != bar_seconds:
                continue
            horizon_signals[nxt] = {**prediction, "signal_at": candles[nxt]["timestamp"].isoformat()}
        signals_by_horizon[h] = horizon_signals
    joint_signals = aggregate_signals(signals_by_horizon, cfg)
    diagnostics = _entry_diagnostics(joint_signals, cfg, fee, slippage)
    forecast_quality = _forecast_realization_diagnostics(
        candles, joint_signals, cfg, fee, slippage, bar_seconds,
    )
    simulated = simulate_rotations(
        candles, joint_signals, cfg, initial_capital=10000,
        fee=fee, slippage=slippage, bar_seconds=bar_seconds,
        # Cost-stress needs the COMPLETE pre-OOS fill schedule, not the 200
        # recent trades kept by the normal user-facing audit.
        audit_limit=2 * len(candles),
    )
    if simulated["actions_truncated"]:
        raise ValueError("Agenda de operações incompleta: impossível testar os mesmos negócios")
    # Replay at the identical fills without reconsidering the entry hurdle.
    # A stressed policy that simply stays in CASH is NOT robustness.
    stressed = replay_frozen_rotations(
        candles, joint_signals, simulated["actions"], cfg,
        initial_capital=10000, fee=fee,
        slippage=min(.05, slippage * STRESS_SLIPPAGE_MULTIPLIER),
        fee_multiplier=STRESS_FEE_MULTIPLIER,
        bar_seconds=bar_seconds,
    )
    robustness = assess_robustness(simulated, stressed)
    capital = simulated["capital_final"]
    dd = simulated["max_drawdown"]
    worst = simulated["worst_fold_return"]
    sharpe = simulated["sharpe"]
    base_score = (
        math.log(max(1e-9, capital / 10000))
        + .5 * dd + .28 * min(0.0, worst)
        + .01 * float(np.clip(sharpe, -4, 4))
        - .01 * float(simulated["completed_cycles"] == 0)
    )
    stressed_score = (
        math.log(max(1e-9, stressed["capital_final"] / 10000))
        + .5 * stressed["max_drawdown"]
        + .28 * min(0.0, stressed["worst_fold_return"])
        + .01 * float(np.clip(stressed["sharpe"], -4, 4))
    )
    # Candidates are ranked by their WORSE risk-adjusted performance.
    # Hard quality requirements apply separately, before winner selection.
    score = min(base_score, stressed_score)
    return {
        "score": float(score),
        "base_score": float(base_score),
        "stress_score": float(stressed_score),
        "score_method": "joint_compounded_capital_risk_adjusted_v9_stress_min",
        "robustness": robustness,
        "forecast_realization_diagnostics": forecast_quality,
        "stress_capital_final": stressed["capital_final"],
        "stress_net_profit_usd": stressed["net_profit_usd"],
        "stress_fees_paid": stressed["fees_paid"],
        "stress_slippage_usd_estimate": stressed["slippage_usd_estimate"],
        "stress_same_trade_count": stressed["completed_cycles"],
        "stress_same_trade_schedule": stressed["same_trade_schedule"],
        "stress_method": stressed["stress_method"],
        "stress_reprice_frozen_actions": True,
        "stress_fees_hypothetical_only": True,
        "segment_cycle_counts": simulated["segment_cycle_counts"],
        "segment_returns": simulated["segment_returns"],
        "average_winning_cycle_usd": simulated["average_winning_cycle_usd"],
        "average_losing_cycle_usd": simulated["average_losing_cycle_usd"],
        "average_win_loss_ratio": simulated["average_win_loss_ratio"],
        "largest_win_share_of_gross_wins": simulated["largest_win_share_of_gross_wins"],
        "capital_final": capital,
        "return_total": simulated["total_return"],
        "max_drawdown": dd,
        "worst_fold_return": worst,
        "sharpe": sharpe,
        "buys": simulated["completed_cycles"],
        "sells": simulated["completed_cycles"],
        "wins": simulated["wins"],
        "losses": simulated["losses"],
        "win_rate": simulated["win_rate"],
        "profit_factor": simulated["profit_factor"],
        "fees_paid": simulated["fees_paid"],
        "alpaca_fees_usd": simulated["alpaca_fees_usd"],
        "slippage_usd_estimate": simulated["slippage_usd_estimate"],
        "net_profit_usd": simulated["net_profit_usd"],
        "fee_model": simulated["fee_model"],
        "fee_tier_history": simulated["fee_tier_history"],
        "exposure": simulated["exposure"],
        "switch_count": simulated["switch_count"],
        "completed_cycles": simulated["completed_cycles"],
        "brier": float(np.mean([v["brier"] for v in results.values()])),
        "mae": float(np.mean([v["mae"] for v in results.values()])),
        "per_horizon": {
            str(h): {
                "brier": results[str(h)]["brier"],
                "mae": results[str(h)]["mae"],
                "signal_count": len(signals_by_horizon[h]),
                "calibration": results[str(h)].get("__calibration"),
                "return_quality": results[str(h)].get("__return_quality"),
            } for h in JOINT_HORIZONS
        },
        "capital_note": "Um patrimônio BTC/CASH reinvestido; custos incluídos",
        "entry_diagnostics": {
            **diagnostics,
            "diagnosis": (
                "ELIGIBLE_BUT_NOT_EXECUTED"
                if simulated["completed_cycles"] == 0
                and diagnostics["passes_all_entry_filters"] > 0
                else diagnostics["diagnosis"]
            ),
        },
        "policy": cfg,
    }



def _update_strategy(strategy_id: str, **fields) -> None:
    with SessionLocal.begin() as session:
        row = session.get(MLResearchStrategy, strategy_id)
        if row is None:
            return
        for name, value in fields.items():
            setattr(row, name, value)
        row.updated_at = _now()


def _run_event_research(strategy_id: str) -> None:
    """A separate, auditable event experiment; no LightGBM or automatic OOS."""
    try:
        with SessionLocal() as session:
            row = session.get(MLResearchStrategy, strategy_id)
            if row is None or row.status == "CANCEL_REQUESTED":
                _update_strategy(strategy_id, status="CANCELLED", error_message=None)
                return
            audit = session.get(MLBacktestAudit, row.source_job_id)
            if audit is None:
                raise ValueError("Auditoria BTC histórica indisponível")
            archive = audit.archive_zip
            research_mode = row.mode
        if research_mode == "DIRECTIONAL_CHANGE":
            experiment = evaluate_archived_directional_change(archive)
            candidate_count = len(BTC_DC_THRESHOLDS)
        elif research_mode == "TECHNICAL":
            experiment = evaluate_archived_technical_research(archive)
            candidate_count = len(BTC_TECHNICAL_POLICIES)
        elif research_mode == "TREND_REGIMES":
            experiment = evaluate_archived_trend_regimes(archive, policy_set=BTC_TREND_POLICIES)
            candidate_count = len(BTC_TREND_POLICIES)
        elif research_mode == "INDICATOR_LAB":
            experiment = evaluate_archived_trend_regimes(archive)
            candidate_count = len(BTC_INDICATOR_POLICIES)
        else:
            experiment = evaluate_archived_event_research(archive)
            candidate_count = len(BTC_EVENT_POLICIES)
        with SessionLocal() as session:
            row = session.get(MLResearchStrategy, strategy_id)
            if row is None or row.status == "CANCEL_REQUESTED":
                _update_strategy(strategy_id, status="CANCELLED", error_message=None)
                return
        summary = {
            **experiment["summary"],
            "trials_target": candidate_count,
            "completed_trials": len(experiment["trials"]),
            "selection_warning": (
                "Pesquisa BTC com regras fixas e validação cronológica pré-OOS. "
                "Topos e fundos históricos só são confirmados depois do movimento. "
                "Não existe modelo aprovado para negociação nem execução OOS automática nesta modalidade."
            ),
        }
        _update_strategy(
            strategy_id, status="RESEARCH_COMPLETE", progress=100,
            trials_json=json.dumps(experiment["trials"], allow_nan=False),
            champion_json=None,
            summary_json=json.dumps(summary, allow_nan=False),
            error_message=None,
        )
    except Exception as exc:
        _update_strategy(strategy_id, status="FAILED", error_message=str(exc)[:1000])


def _run_optimization(strategy_id: str) -> None:
    try:
        with SessionLocal() as session:
            row = session.get(MLResearchStrategy, strategy_id)
            if row is None:
                return
            record = session.get(MLBacktestAudit, row.source_job_id)
            if record is None:
                raise ValueError("Auditoria original indisponível no banco")
            original, snapshot = inspect_source_archive(record.archive_zip)
            archive = record.archive_zip
            mode, horizon, trials_count = row.mode, row.horizon_hours, row.trials_target
            # Persist intent in the existing summary column, without schema migrations.
            automatic_execution = bool(json.loads(row.summary_json or "{}").get("auto_execute", False))
        base = get_runtime_parameters_from_snapshot(snapshot)
        # New learner settings are drawn from the current DB. All features
        # and original chronological splits stay pinned to the source archive.
        current = get_runtime_parameters().snapshot()
        values = base.snapshot()
        values.update({k: v for k, v in current.items() if k.startswith("lightgbm.")
                       or k.startswith("forecast.regression_")})
        base = RuntimeParameters(values)
        horizons = JOINT_HORIZONS if horizon == 0 else (horizon,)
        with ZipFile(BytesIO(archive)) as z:
            fold_windows = json.loads(z.read("folds.json"))
        boundaries = [
            pd.Timestamp(fold["test_start"]) for fold in fold_windows
            if int(fold["horizon_hours"]) in horizons
        ]
        if set(horizons) - {int(fold["horizon_hours"]) for fold in fold_windows}:
            raise ValueError("Faltam folds OOS de algum horizonte na auditoria histórica")
        first_oos = min(boundaries)
        datasets = {h: _prepare(archive, h, base, first_oos) for h in horizons}
        training_counts = {
            str(h): {
                "training_rows": len(data[0]),
                "early_stopping_rows": len(data[1]),
                "calibration_rows": len(data[2]),
                "selection_rows": len(data[3]),
            "selection_start": pd.Timestamp(data[3]["timestamp"].iloc[0]).isoformat(),
            "selection_end": pd.Timestamp(data[3]["timestamp"].iloc[-1]).isoformat(),
            }
            for h, data in datasets.items()
        }
        sampler = qmc.LatinHypercube(d=len(SPACE), seed=42)
        rng = np.random.default_rng(42)
        points: list[np.ndarray] = []
        scores: list[float] = []
        records: list[dict] = []
        winner: dict | None = None
        best_score = -float("inf")
        for index in range(trials_count):
            with SessionLocal() as session:
                current = session.get(MLResearchStrategy, strategy_id)
                if current is None or current.status == "CANCEL_REQUESTED":
                    _update_strategy(strategy_id, status="CANCELLED", error_message=None)
                    return
            vector = _next_candidate(index, mode, points, scores, sampler, rng, records)
            candidate = _candidate(vector)
            try:
                per_horizon = {
                    str(h): _score(datasets[h], candidate, base) for h in horizons
                }
                metrics = _combine_horizon_metrics(
                    per_horizon, scoring=datasets[2][3] if len(horizons) == 3 else None,
                    policy=candidate, fee=original.fee_rate,
                    slippage=original.slippage_rate,
                    bar_seconds=TIMEFRAME_SECONDS[original.decision_timeframe],
                )
                score = metrics["score"]
                # One profitable cycle is evidence of an isolated event, not
                # a stable strategy. No OOS run until ALL pre-OOS gates pass.
                cycles = metrics.get("completed_cycles", metrics.get("buys", 0))
                eligible = (bool(metrics.get("robustness", {}).get("passed"))
                            if len(horizons) == 3 else bool(cycles))
                status = "NO_TRADES" if not cycles else (
                    "COMPLETED" if eligible else "INSUFFICIENT_EVIDENCE"
                )
                entry = {"trial": index + 1, "status": status,
                         "parameters": candidate, "metrics": metrics}
            except Exception as exc:
                score = -1000.0
                entry = {"trial": index + 1, "status": "FAILED",
                         "parameters": candidate, "error": str(exc)[:250]}
            points.append(vector)
            scores.append(score)
            records.append(entry)
            if entry["status"] == "COMPLETED" and score > best_score:
                best_score, winner = score, entry
            _update_strategy(
                strategy_id, progress=round(100 * (index + 1) / trials_count),
                trials_json=json.dumps(records, allow_nan=False),
                champion_json=json.dumps(winner, allow_nan=False) if winner else None,
            )
        with SessionLocal() as session:
            current = session.get(MLResearchStrategy, strategy_id)
            if current is not None and current.status == "CANCEL_REQUESTED":
                _update_strategy(strategy_id, status="CANCELLED", error_message=None)
                return
        if winner is None:
            inactive = [entry for entry in records if entry["status"] == "NO_TRADES"]
            insufficient = [entry for entry in records if entry["status"] == "INSUFFICIENT_EVIDENCE"]
            unsuccessful = [entry for entry in records if entry["status"] == "FAILED"]
            if insufficient and not any(item["status"] == "COMPLETED" for item in records):
                summary = {
                    "validation_kind": "INTERNAL_PRE_OOS_CHRONOLOGICAL",
                    "oos_untouched_from": first_oos.isoformat(),
                    "horizons_hours": list(horizons),
                    "optimization_target": "NET_COMPOUNDED_BTC_CASH_CAPITAL",
                    "completed_trials": 0,
                    "inactive_trials": len(inactive),
                    "insufficient_evidence_trials": len(insufficient),
                    "failed_trials": len(unsuccessful),
                    "quality_gate": "V10_PRE_OOS_FROZEN_TRADE_COST_STRESS",
                    "quality_by_trial": [
                        {"trial": item["trial"], "status": item["status"],
                         "robustness": item["metrics"].get("robustness")}
                        for item in records if item.get("metrics")
                    ],
                    "research_iteration_warning": "V9 foi projetada depois da inspeção de OOS da V8. Este OOS permanece excluído do treinamento, mas não é um teste novo independente entre versões.",
                    "selection_warning": (
                        "Houve compras na validação, mas nenhum candidato "
                        "demonstrou consistência suficiente antes do OOS."
                    ),
                }
                _update_strategy(
                    strategy_id, status="INSUFFICIENT_EVIDENCE", progress=100,
                    champion_json=None,
                    summary_json=json.dumps(summary, allow_nan=False),
                    error_message=(
                        "A pesquisa encontrou operações, mas nenhum candidato atingiu "
                        "os critérios de consistência temporal, resultado líquido "
                        "e resistência a taxas/slippage maiores. "
                        "Nenhuma execução OOS foi iniciada; consulte os diagnósticos."
                    ),
                )
                return
            if inactive:
                diagnostics_by_trial = [
                    {
                        "trial": entry["trial"],
                        "diagnostics": entry["metrics"].get("entry_diagnostics", {}),
                    }
                    for entry in inactive
                ]
                diagnostic = max(
                    (entry["diagnostics"] for entry in diagnostics_by_trial),
                    key=lambda item: (
                        item.get("passes_all_entry_filters", 0),
                        item.get("passes_expected_return", 0),
                        item.get("passes_probability", 0),
                    ),
                    default={},
                )
                summary = {
                    "validation_kind": "INTERNAL_PRE_OOS_CHRONOLOGICAL",
                    "oos_untouched_from": first_oos.isoformat(),
                    "horizons_hours": list(horizons),
                    "optimization_target": "NET_COMPOUNDED_BTC_CASH_CAPITAL",
                    "completed_trials": 0,
                    "inactive_trials": len(inactive),
                    "failed_trials": len(unsuccessful),
                    "diagnostic_example": diagnostic,
                    "diagnostics_by_trial": diagnostics_by_trial,
                    "reason_code": diagnostic.get("diagnosis", "UNKNOWN"),
                    "selection_warning": "Sem compras na validação; não há candidato econômico vencedor.",
                }
                _update_strategy(
                    strategy_id, status="NO_TRADES", progress=100,
                    champion_json=None,
                    summary_json=json.dumps(summary, allow_nan=False),
                    error_message=(
                        f"Os {len(inactive)} candidatos treinados não realizaram compras BTC/CASH. "
                        "Consulte os filtros de entrada por tentativa. "
                        "Não existe vencedor para executar fora da amostra."
                    ),
                )
                return
            raise ValueError("Nenhum candidato LightGBM conseguiu ser treinado")
        # Freeze ALL active learner parameters, not just the seven searched
        # dimensions. Later DB edits must not silently change this champion.
        winner["model_snapshot"] = {
            **{
                key: value for key, value in base.snapshot().items()
                if key.startswith("lightgbm.") or key.startswith("forecast.regression_")
            },
            **winner["parameters"],
        }
        _update_strategy(strategy_id, champion_json=json.dumps(winner, allow_nan=False))
        summary = {
            "validation_kind": "INTERNAL_PRE_OOS_CHRONOLOGICAL",
            "oos_untouched_from": first_oos.isoformat(),
            "horizons_hours": list(horizons),
            "training_rows_by_horizon": training_counts,
            "aggregation_method": "Single compounded BTC/CASH portfolio; final capital with risk and temporal stability",
            "selection_segments": 3,
            "selection_window_rule": "last 25 percent of pre-OOS candles; up to 48000 archived candles; purged boundaries",
            "optimization_target": "NET_COMPOUNDED_BTC_CASH_CAPITAL",
            "optimization_dimensions": len(SPACE),
            "learner_dimensions": len(SPACE) - len(POLICY_SPACE),
            "policy_dimensions": len(POLICY_SPACE),
            "optimized_decision_policy": True,
            "capitalization_engine": "btc_policy.simulate_rotations",
            "quality_gate": "V10_PRE_OOS_FROZEN_TRADE_COST_STRESS",
            "research_iteration_warning": "V9 foi projetada depois da inspeção de OOS da V8. Este OOS permanece excluído do treinamento, mas não é um teste novo independente entre versões.",
            "minimum_completed_cycles": MIN_COMPLETED_CYCLES,
            "minimum_active_segments": MIN_ACTIVE_SEGMENTS,
            "minimum_profitable_segments": MIN_PROFITABLE_SEGMENTS,
            "fee_stress_multiplier": STRESS_FEE_MULTIPLIER,
            "stress_schedule": "All original pre-OOS buys and sells repriced at the identical candles without recomputing entries",
            "regression_diagnostic": "Per-horizon forecasting error versus training-mean baseline; 5 quantile buckets and amplitude, development validation only",
            "slippage_stress_multiplier": STRESS_SLIPPAGE_MULTIPLIER,
            "quality_by_trial": [
                {"trial": item["trial"], "status": item["status"],
                 "robustness": item["metrics"].get("robustness")}
                for item in records if item.get("metrics")
            ],
            "objective_note": "Stops confirmed at close; filled at next open; final OOS untouched",
            "model_family": "LightGBM",
            "optimization_algorithm": ("CRYPTO_CARO_GP_CONSTRAINED_EI" if mode == "BAYESIAN" else "LATIN_HYPERCUBE"),
            "caro_note": "Crypto adaptation using sklearn GPs, not the MCT BoTorch engine.",
            "selected_score": best_score,
            "candidate_policy": validate_policy(winner["parameters"]),
            "capital_definition": "10,000 USD equivalentes, BTC/CASH, reinvestimento integral, taxas maker/taker Alpaca por 30D, slippage separado",
            "fee_model": ALPACA_FEE_MODEL,
            "fee_liquidity": "TAKER",
            "fee_assumptions": "Crypto 8-tier 30d USD volume; reset 03:00 NY daily; only this strategy known; source BTCUSDT denominated in USDT approximates future BTCUSD",
            "initial_capital_for_selection": 10000,
            "completed_trials": sum(item["status"] == "COMPLETED" for item in records),
            "inactive_trials": sum(item["status"] == "NO_TRADES" for item in records),
            "insufficient_evidence_trials": sum(
                item["status"] == "INSUFFICIENT_EVIDENCE" for item in records
            ),
            "failed_trials": sum(item["status"] == "FAILED" for item in records),
            "historical_source_job_id": row.source_job_id,
            "selection_warning": "Resultados de seleção interna, não são o teste final OOS.",
        }
        _update_strategy(
            strategy_id, status="OPTIMIZED", progress=100,
            summary_json=json.dumps({**summary, "auto_execute": automatic_execution}, allow_nan=False),
            error_message=None,
        )
        # The same server-side workflow schedules the final replay. No second
        # frontend click or dependency on an open browser tab is required.
        if automatic_execution:
            try:
                execute_strategy(strategy_id)
            except Exception as exc:
                # Retain the optimized champion so execution can be recovered;
                # do not silently convert a successful optimization into FAILED.
                _update_strategy(strategy_id, status="OPTIMIZED",
                                 error_message="Execução OOS automática não iniciada: " + str(exc)[:800])
    except Exception as exc:
        _update_strategy(strategy_id, status="FAILED", error_message=str(exc)[:1000])


def start_optimization(strategy_id: str, *, mode: str, trials: int, auto_execute: bool = False) -> dict:
    if mode not in {"QUICK", "BAYESIAN", "EVENTS", "DIRECTIONAL_CHANGE", "TECHNICAL", "TREND_REGIMES", "INDICATOR_LAB"}:
        raise ValueError("Modo de otimização inválido")
    if mode == "EVENTS" and trials != len(BTC_EVENT_POLICIES):
        raise ValueError("A pesquisa por eventos usa quatro políticas pré-definidas")
    if mode == "DIRECTIONAL_CHANGE" and trials != len(BTC_DC_THRESHOLDS):
        raise ValueError("Directional Change usa quatro limites fixados previamente")
    if mode == "TECHNICAL" and trials != len(BTC_TECHNICAL_POLICIES):
        raise ValueError("A análise técnica compara as seis políticas pré-definidas")
    if mode == "TREND_REGIMES" and trials != len(BTC_TREND_POLICIES):
        raise ValueError("Tendência e regimes exigem seis hipóteses pré-definidas")
    if mode == "INDICATOR_LAB" and trials != len(BTC_INDICATOR_POLICIES):
        raise ValueError("V15 exige doze hipóteses técnicas pré-definidas")
    if not 4 <= trials <= MAX_TRIALS:
        raise ValueError("Quantidade de tentativas fora dos limites")
    with SessionLocal.begin() as session:
        row = session.get(MLResearchStrategy, strategy_id)
        if row is None:
            raise LookupError("Estratégia não encontrada")
        if row.status == "RUNNING":
            raise ValueError("Esta estratégia já possui uma otimização em andamento")
        if row.execution_job_id:
            raise ValueError("Crie uma nova estratégia para uma otimização após execução final")
        row.status = "RUNNING"
        row.mode = mode
        row.trials_target = trials
        row.progress = 0
        row.trials_json = "[]"
        row.champion_json = None
        row.summary_json = json.dumps({"auto_execute": bool(auto_execute)})
        row.error_message = None
        row.updated_at = _now()
    # Shared job executor; no background promises and no shell scripts.
    from .service import _get_executor
    from .settings_service import get_runtime_parameters
    _get_executor(get_runtime_parameters()).submit(
        _run_event_research if mode in {"EVENTS", "DIRECTIONAL_CHANGE", "TECHNICAL", "TREND_REGIMES", "INDICATOR_LAB"} else _run_optimization,
        strategy_id,
    )
    return get_strategy(strategy_id)


def cancel_optimization(strategy_id: str) -> dict:
    """Cancel the ongoing campaign at the next completed trial."""
    with SessionLocal.begin() as session:
        row = session.get(MLResearchStrategy, strategy_id)
        if row is None:
            raise LookupError("Estratégia de pesquisa não encontrada")
        if row.status != "RUNNING":
            raise ValueError("Não há otimização em andamento para cancelar")
        row.status = "CANCEL_REQUESTED"
        row.updated_at = _now()
        session.flush()
        return _decode(row)


def execute_strategy(strategy_id: str) -> dict:
    from .service import create_strategy_job
    with SessionLocal.begin() as session:
        row = session.get(MLResearchStrategy, strategy_id)
        if row is None:
            raise LookupError("Estratégia não encontrada")
        if row.status != "OPTIMIZED" or not row.champion_json:
            raise ValueError("Primeiro conclua uma otimização com candidato válido")
        if row.execution_job_id:
            raise ValueError("Esta estratégia já possui execução final")
        champion = json.loads(row.champion_json)
        if (
            row.horizon_hours == 0
            and champion.get("metrics", {}).get("completed_cycles") == 0
        ):
            raise ValueError(
                "Esta campanha terminou sem compras BTC/CASH. "
                "Não existe política de capitalização válida para executar OOS. "
                "Inicie uma nova campanha e confira os filtros de entrada."
            )
        source = row.source_job_id
        overrides = champion.get("model_snapshot") or champion["parameters"]
        # Source/model metadata are recorded by the actual job snapshot.
        created = create_strategy_job(source, overrides)
        row.execution_job_id = created["id"]
        row.status = "EXECUTING"
        row.updated_at = _now()
        return created


def export_strategy_report(strategy_id: str) -> bytes:
    strategy = get_strategy(strategy_id)
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as z:
        z.writestr("strategy.json", json.dumps(strategy, ensure_ascii=False, indent=2))
        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["trial", "status", "score", "capital_final", "max_drawdown",
                         "sharpe", "worst_fold_return", "brier", "mae", "parameters_json", "error"])
        for item in strategy["trials"]:
            metrics = item.get("metrics") or {}
            writer.writerow([
                item["trial"], item["status"], metrics.get("score"),
                metrics.get("capital_final"), metrics.get("max_drawdown"),
                metrics.get("sharpe"), metrics.get("worst_fold_return"),
                metrics.get("brier"), metrics.get("mae"),
                json.dumps(item["parameters"], sort_keys=True), item.get("error"),
            ])
        z.writestr("optimization_trials.csv", csv_buffer.getvalue())
        if strategy["mode"] == "EVENTS":
            z.writestr(
                "event_research.json",
                json.dumps({
                    "summary": strategy["summary"],
                    "trials": strategy["trials"],
                }, ensure_ascii=False, indent=2, allow_nan=False),
            )
            event_csv = StringIO()
            event_writer = csv.writer(event_csv)
            event_writer.writerow([
                "policy", "period", "capital_final", "net_profit_usd",
                "completed_cycles", "max_drawdown", "fees_paid", "slippage",
                "ambiguous_exits", "gap_exits",
            ])
            for item in strategy["trials"]:
                metrics = item.get("metrics") or {}
                for part, row in (("development", metrics.get("development", {})),
                                  ("validation", metrics)):
                    event_writer.writerow([
                        item["parameters"]["policy"], part, row.get("capital_final"),
                        row.get("net_profit_usd"), row.get("completed_cycles"),
                        row.get("max_drawdown"), row.get("fees_paid"),
                        row.get("slippage_usd_estimate"), row.get("ambiguous_barrier_exits"),
                        row.get("gap_exits"),
                    ])
            z.writestr("event_research_portfolios.csv", event_csv.getvalue())
        if strategy["mode"] == "DIRECTIONAL_CHANGE":
            z.writestr(
                "directional_change.json",
                json.dumps({"summary": strategy["summary"], "trials": strategy["trials"]},
                           ensure_ascii=False, indent=2, allow_nan=False),
            )
            dc_csv = StringIO()
            dc_writer = csv.writer(dc_csv)
            dc_writer.writerow([
                "policy", "period", "capital_final", "net_profit_usd",
                "completed_cycles", "confirmed_bottoms", "confirmed_tops",
                "median_lag_candles", "max_drawdown", "fees_paid", "slippage",
            ])
            for item in strategy["trials"]:
                for part, m in (("development", item["metrics"]["development"]),
                                ("validation", item["metrics"])):
                    dc_writer.writerow([
                        item["parameters"]["policy"], part, m.get("capital_final"),
                        m.get("net_profit_usd"), m.get("completed_cycles"),
                        m.get("confirmed_bottoms"), m.get("confirmed_tops"),
                        m.get("median_confirmation_lag_candles"),
                        m.get("max_drawdown"), m.get("fees_paid"),
                        m.get("slippage_usd_estimate"),
                    ])
            z.writestr("directional_change_portfolios.csv", dc_csv.getvalue())
        if strategy["mode"] in {"TREND_REGIMES", "INDICATOR_LAB"}:
            z.writestr("trend_regimes.json", json.dumps({
                "summary": strategy["summary"], "trials": strategy["trials"],
            }, ensure_ascii=False, indent=2, allow_nan=False))
            report_csv = StringIO()
            writer = csv.writer(report_csv)
            keys = ["policy", "partition", "approved", "capital_final", "return_total",
                    "max_drawdown", "sharpe", "buy_count", "sell_count",
                    "completed_cycles", "operation_count", "exposure",
                    "fees_paid", "slippage_usd_estimate", "quality_checks_json"]
            writer.writerow(keys)
            for trial in strategy["trials"]:
                for part, row in (("development", trial["metrics"]["development"]),
                                  ("validation", trial["metrics"])):
                    writer.writerow([
                        trial["parameters"]["policy"], part,
                        trial["parameters"]["policy"] in strategy["summary"].get("approved_policies", []),
                        row.get("capital_final"), row.get("return_total"),
                        row.get("max_drawdown"), row.get("sharpe"),
                        row.get("buy_count"), row.get("sell_count"),
                        row.get("completed_cycles"), row.get("operation_count"),
                        row.get("exposure"), row.get("fees_paid"), row.get("slippage_usd_estimate"),
                        json.dumps(trial["metrics"]["quality_checks"], ensure_ascii=False),
                    ])
            z.writestr("trend_regimes_portfolios.csv", report_csv.getvalue())
            if strategy["mode"] == "INDICATOR_LAB":
                chart = strategy["summary"].get("chart") or {}
                markers_csv = StringIO()
                marker_writer = csv.writer(markers_csv)
                marker_writer.writerow(["type", "action", "timestamp", "price",
                                        "recognized_at", "execution_at", "diagnostic_only", "reason"])
                for item in chart.get("inflections", []):
                    marker_writer.writerow(["HISTORICAL_INFLECTION", item.get("kind"),
                        item.get("pivot_timestamp"), item.get("pivot_close"),
                        item.get("confirmed_at"), "", True, "confirmed after pivot"])
                    marker_writer.writerow(["INFLECTION_CONFIRMED", item.get("kind"),
                        item.get("confirmed_at"), item.get("confirmed_close"),
                        item.get("confirmed_at"), "", True, "2% reversal"])
                for item in chart.get("signals", []):
                    marker_writer.writerow(["POLICY_SIGNAL", item.get("action"),
                        item.get("timestamp"), item.get("price"), item.get("timestamp"),
                        item.get("execution_at"), chart.get("diagnostic_only"), item.get("reason")])
                for item in chart.get("actions", []):
                    marker_writer.writerow(["SIMULATED_FILL", item.get("action"),
                        item.get("timestamp"), item.get("execution_price"), item.get("signal_at"),
                        item.get("timestamp"), chart.get("diagnostic_only"), item.get("reason")])
                z.writestr("indicator_lab_markers.csv", markers_csv.getvalue())
        if strategy["mode"] == "TECHNICAL":
            z.writestr(
                "technical_research.json",
                json.dumps({"summary": strategy["summary"], "trials": strategy["trials"]},
                           ensure_ascii=False, indent=2, allow_nan=False),
            )
            technical_csv = StringIO()
            writer = csv.writer(technical_csv)
            writer.writerow(["policy", "partition", "capital_final", "return_total",
                             "max_drawdown", "sharpe", "completed_cycles",
                             "exposure", "fees_paid", "slippage"])
            for trial in strategy["trials"]:
                for part, row in (("development", trial["metrics"]["development"]),
                                  ("validation", trial["metrics"])):
                    writer.writerow([
                        trial["parameters"]["policy"], part, row.get("capital_final"),
                        row.get("return_total"), row.get("max_drawdown"),
                        row.get("sharpe"), row.get("completed_cycles"),
                        row.get("exposure"), row.get("fees_paid"),
                        row.get("slippage_usd_estimate"),
                    ])
            z.writestr("technical_research_portfolios.csv", technical_csv.getvalue())
        z.writestr("README.txt", (
            "BTC/CASH audited research: LightGBM or BTC-specific CUSUM event study.\n"
            "EVENTS: event_research.json, event_research_portfolios.csv and event_trade_actions.csv.\\n"
            "DIRECTIONAL_CHANGE: directional_change.json, _portfolios.csv, _extrema.csv, _trades.csv.\\n"
            "TECHNICAL: technical_research.json, _portfolios.csv, _trades.csv.\\n"
            "TREND_REGIMES: trend_regimes.json, _portfolios.csv, _trades.csv.\\n"
            "INDICATOR_LAB: trend_regimes.json, _portfolios.csv, _trades.csv, indicator_lab_markers.csv.\\n"
            "strategy.json: strategy and optimization metadata\n"
            "optimization_trials.csv: model and BTC/CASH policy candidates\n"
            "source_historical_audit.zip: immutable source data and labels\n"
            "final_execution_result.json: final OOS metrics when completed\n"
            "final_execution_parameters.json: LightGBM champion snapshot\n"
            "final_oos_audit.zip: all OOS predictions and OHLCV of the executed candidate\n"
            "final_oos_capitalization.json: reinvested BTC/CASH curve, actions, risk and Buy-and-Hold\n"
            "Internal pre-OOS tuning is not proof of live profitability.\n"
        ))
        with SessionLocal() as session:
            source = session.get(MLBacktestAudit, strategy["source_job_id"])
            if source is not None:
                z.writestr("source_historical_audit.zip", source.archive_zip)
                if strategy["mode"] == "EVENTS":
                    full_replay = evaluate_archived_event_research(
                        source.archive_zip, include_trades=True,
                    )
                    if full_replay["trials"] != strategy["trials"]:
                        raise ValueError("Auditoria histórica de eventos diverge da pesquisa persistida")
                    trades_csv = StringIO()
                    trade_writer = csv.writer(trades_csv)
                    trade_writer.writerow([
                        "policy", "partition", "timestamp", "action", "reason",
                        "signal_candle_index", "price", "fee_rate", "fee_tier",
                        "fee_usd", "slippage_usd", "capital_after",
                    ])
                    for trade in full_replay["event_trades"]:
                        trade_writer.writerow([trade.get(key) for key in (
                            "policy", "partition", "timestamp", "action", "reason",
                            "signal_candle_index", "price", "fee_rate", "fee_tier",
                            "fee_usd", "slippage_usd", "capital_after",
                        )])
                    z.writestr("event_trade_actions.csv", trades_csv.getvalue())

                if strategy["mode"] in {"TREND_REGIMES", "INDICATOR_LAB"}:
                    research = evaluate_archived_trend_regimes(
                        source.archive_zip, include_details=True,
                        policy_set=BTC_TREND_POLICIES if strategy["mode"] == "TREND_REGIMES" else None,
                    )
                    if research["trials"] != strategy["trials"]:
                        raise ValueError("Auditoria de tendências diverge da execução persistida")
                    trade_csv = StringIO()
                    writer = csv.writer(trade_csv)
                    fields = ["policy", "partition", "action", "timestamp", "signal_at",
                              "execution_candle_index", "execution_price",
                              "target_fraction", "reason", "signal_price", "fee_rate", "fee_tier",
                              "fee_usd", "slippage_usd", "capital_after"]
                    writer.writerow(fields)
                    for action in research["trades"]:
                        writer.writerow([action.get(k) for k in fields])
                    z.writestr("trend_regimes_trades.csv", trade_csv.getvalue())
                if strategy["mode"] == "TECHNICAL":
                    research = evaluate_archived_technical_research(
                        source.archive_zip, include_details=True,
                    )
                    if research["trials"] != strategy["trials"]:
                        raise ValueError("Auditoria técnica diverge da pesquisa persistida")
                    actions_csv = StringIO()
                    writer = csv.writer(actions_csv)
                    fields = ["policy", "partition", "action", "timestamp", "signal_at",
                              "execution_candle_index", "execution_price", "reason",
                              "fee_rate", "fee_tier", "fee_usd", "slippage_usd",
                              "capital_after", "net_profit_usd"]
                    writer.writerow(fields)
                    for action in research["trades"]:
                        writer.writerow([action.get(k) for k in fields])
                    z.writestr("technical_research_trades.csv", actions_csv.getvalue())

                if strategy["mode"] == "DIRECTIONAL_CHANGE":
                    dc = evaluate_archived_directional_change(
                        source.archive_zip, include_details=True,
                    )
                    if dc["trials"] != strategy["trials"]:
                        raise ValueError("Auditoria histórica de extremos não reproduz a pesquisa persistida")
                    extrema_csv = StringIO()
                    extrema_writer = csv.writer(extrema_csv)
                    extrema_fields = [
                        "policy", "partition", "kind", "pivot_index",
                        "pivot_timestamp", "pivot_close", "confirmed_index",
                        "confirmed_at", "confirmed_close", "threshold_fraction",
                        "lag_candles", "confirmation_move_fraction",
                    ]
                    extrema_writer.writerow(extrema_fields)
                    for item in dc["extrema"]:
                        extrema_writer.writerow([item.get(key) for key in extrema_fields])
                    z.writestr("directional_change_extrema.csv", extrema_csv.getvalue())

                    trade_csv = StringIO()
                    trade_writer = csv.writer(trade_csv)
                    trade_fields = [
                        "policy", "partition", "action", "timestamp",
                        "execution_candle_index", "execution_price", "reason",
                        "signal_type", "pivot_timestamp", "pivot_close",
                        "confirmed_at", "confirmed_close",
                        "confirmation_lag_candles", "threshold_fraction",
                        "fee_rate", "fee_tier", "fee_usd", "slippage_usd",
                        "notional_usd", "capital_after", "cycle_net_profit_usd",
                    ]
                    trade_writer.writerow(trade_fields)
                    for item in dc["trades"]:
                        trade_writer.writerow([item.get(key) for key in trade_fields])
                    z.writestr("directional_change_trades.csv", trade_csv.getvalue())

            job = session.get(MLBacktestJob, strategy["execution_job_id"]) if strategy["execution_job_id"] else None
            audit = session.get(MLBacktestAudit, job.id) if job and job.status == "COMPLETED" else None
            if job and job.result_json:
                z.writestr("final_execution_result.json", job.result_json)
                z.writestr("final_execution_parameters.json", job.settings_snapshot_json or "{}")
            if audit is not None:
                z.writestr("final_oos_audit.zip", audit.archive_zip)
                frozen = json.loads(job.settings_snapshot_json or "{}")
                if frozen.get("policy.enabled") and source is not None:
                    from .capitalization import compare_capitalization
                    request = json.loads(job.request_json or "{}")
                    financial = compare_capitalization(
                        source.archive_zip, audit.archive_zip,
                        horizon=0, initial_capital=10000,
                        fee_rate=float(request.get("fee_rate", 0.001)),
                        slippage_rate=float(request.get("slippage_rate", 0.0005)),
                        entry_edge=0.0, min_probability=0.5, exit_return=0.0,
                        full_audit=True,
                        optimized_policy={
                            k: v for k, v in frozen.items() if k.startswith("policy.")
                        },
                    )
                    z.writestr(
                        "final_oos_capitalization.json",
                        json.dumps(financial, ensure_ascii=False, indent=2, allow_nan=False),
                    )
    return buffer.getvalue()
