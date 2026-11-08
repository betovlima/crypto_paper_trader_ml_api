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
MAX_DEV_CANDLES = 16000
MAX_TRIALS = 40


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _decode(row: MLResearchStrategy) -> dict:
    return {
        "id": row.id, "name": row.name, "source_job_id": row.source_job_id,
        "horizon_hours": row.horizon_hours, "status": row.status, "mode": row.mode,
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


def create_strategy(*, name: str, source_job_id: str, horizon_hours: int) -> dict:
    with SessionLocal.begin() as session:
        job = session.get(MLBacktestJob, source_job_id)
        if job is None or job.status != "COMPLETED":
            raise ValueError("Selecione uma execução BTC concluída com auditoria")
        audit = session.get(MLBacktestAudit, source_job_id)
        if audit is None:
            raise ValueError("A execução selecionada não possui auditoria armazenada na ML API")
        source_request, source_settings = inspect_source_archive(audit.archive_zip)
        if horizon_hours not in [int(x) for x in source_settings["forecast.horizons_hours"]]:
            raise ValueError("Horizonte não está disponível no histórico selecionado")
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
    return settings


def _next_candidate(
    index: int, mode: str, points: list[np.ndarray], scores: list[float],
    sampler: qmc.LatinHypercube, rng: np.random.Generator,
) -> np.ndarray:
    if mode != "BAYESIAN" or len(points) < 8:
        return sampler.random(1)[0]
    try:
        kernel = ConstantKernel(1.0, (0.01, 100.0)) * Matern(
            length_scale=np.ones(len(SPACE)), nu=2.5
        ) + WhiteKernel(noise_level=0.001)
        gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, random_state=42,
            n_restarts_optimizer=1,
        )
        gp.fit(np.stack(points), np.asarray(scores))
        choices = rng.uniform(size=(320, len(SPACE)))
        means, uncertainty = gp.predict(choices, return_std=True)
        uncertainty = np.maximum(uncertainty, 1e-9)
        improvement = means - max(scores) - 0.001
        z = improvement / uncertainty
        expected_improvement = improvement * norm.cdf(z) + uncertainty * norm.pdf(z)
        return choices[int(np.argmax(expected_improvement))]
    except (ValueError, np.linalg.LinAlgError):
        return sampler.random(1)[0]


def _prepare(source_archive: bytes, horizon: int, parameters: RuntimeParameters):
    """Freeze training strictly before the FIRST archived OOS test date."""
    request, _ = inspect_source_archive(source_archive)
    with ZipFile(BytesIO(source_archive)) as archive:
        windows = json.loads(archive.read("folds.json"))
    first_oos = min(
        pd.Timestamp(fold["test_start"])
        for fold in windows if int(fold["horizon_hours"]) == horizon
    )
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
    start_early = int(n * .60)
    start_cal = int(n * .77)
    start_score = int(n * .87)
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
    if btc > 0:
        equity = btc * close[-1] * (1 - slippage) * (1 - fee)
        sells += 1
    return {
        "capital_final": float(equity), "return_total": float(equity / 10000 - 1),
        "max_drawdown": float(maxdd), "buys": buys, "sells": sells,
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
    capital = _simulate(
        scoring, probs, forecasts, request.fee_rate, request.slippage_rate,
        request.decision_timeframe,
    )
    # The selection objective is economic. Predictive errors are informative
    # tie-breakers; never fit/tune on the final OOS test.
    score = (
        math.log(max(1e-9, capital["capital_final"] / 10000))
        + 0.5 * capital["max_drawdown"]
        - 0.03 * brier
        - 0.03 * mae
        - (0.01 if capital["buys"] == 0 else 0)
    )
    return {
        "score": float(score),
        "brier": brier, "mae": mae, **capital,
    }


def _update_strategy(strategy_id: str, **fields) -> None:
    with SessionLocal.begin() as session:
        row = session.get(MLResearchStrategy, strategy_id)
        if row is None:
            return
        for name, value in fields.items():
            setattr(row, name, value)
        row.updated_at = _now()


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
        base = get_runtime_parameters_from_snapshot(snapshot)
        # New learner settings are drawn from the current DB. All features
        # and original chronological splits stay pinned to the source archive.
        current = get_runtime_parameters().snapshot()
        values = base.snapshot()
        values.update({k: v for k, v in current.items() if k.startswith("lightgbm.")
                       or k.startswith("forecast.regression_")})
        base = RuntimeParameters(values)
        dataset = _prepare(archive, horizon, base)
        train, early, calibration, scoring, _, _, first_oos = dataset
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
            vector = _next_candidate(index, mode, points, scores, sampler, rng)
            candidate = _candidate(vector)
            try:
                metrics = _score(dataset, candidate, base)
                score = metrics["score"]
                entry = {"trial": index + 1, "status": "COMPLETED",
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
        if winner is None:
            raise ValueError("Nenhum candidato LightGBM conseguiu ser treinado")
        summary = {
            "validation_kind": "INTERNAL_PRE_OOS_CHRONOLOGICAL",
            "oos_untouched_from": first_oos.isoformat(),
            "training_rows": len(train),
            "early_stopping_rows": len(early),
            "calibration_rows": len(calibration),
            "selection_rows": len(scoring),
            "model_family": "LightGBM",
            "selected_score": best_score,
            "completed_trials": sum(item["status"] == "COMPLETED" for item in records),
            "failed_trials": sum(item["status"] == "FAILED" for item in records),
            "historical_source_job_id": row.source_job_id,
            "selection_warning": "Resultados de seleção interna, não são o teste final OOS.",
        }
        _update_strategy(
            strategy_id, status="OPTIMIZED", progress=100,
            summary_json=json.dumps(summary, allow_nan=False), error_message=None,
        )
    except Exception as exc:
        _update_strategy(strategy_id, status="FAILED", error_message=str(exc)[:1000])


def start_optimization(strategy_id: str, *, mode: str, trials: int) -> dict:
    if mode not in {"QUICK", "BAYESIAN"}:
        raise ValueError("Modo de otimização inválido")
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
        row.summary_json = None
        row.error_message = None
        row.updated_at = _now()
    # Shared job executor; no background promises and no shell scripts.
    from .service import _get_executor
    from .settings_service import get_runtime_parameters
    _get_executor(get_runtime_parameters()).submit(_run_optimization, strategy_id)
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
        source = row.source_job_id
        overrides = champion["parameters"]
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
                         "brier", "mae", "parameters_json", "error"])
        for item in strategy["trials"]:
            metrics = item.get("metrics") or {}
            writer.writerow([
                item["trial"], item["status"], metrics.get("score"),
                metrics.get("capital_final"), metrics.get("max_drawdown"),
                metrics.get("brier"), metrics.get("mae"),
                json.dumps(item["parameters"], sort_keys=True), item.get("error"),
            ])
        z.writestr("optimization_trials.csv", csv_buffer.getvalue())
        with SessionLocal() as session:
            job = session.get(MLBacktestJob, strategy["execution_job_id"]) if strategy["execution_job_id"] else None
            audit = session.get(MLBacktestAudit, job.id) if job and job.status == "COMPLETED" else None
            if job and job.result_json:
                z.writestr("final_execution_result.json", job.result_json)
                z.writestr("final_execution_parameters.json", job.settings_snapshot_json or "{}")
            if audit is not None:
                z.writestr("final_oos_audit.zip", audit.archive_zip)
    return buffer.getvalue()
