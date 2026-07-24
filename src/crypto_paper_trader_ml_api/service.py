from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4

from sqlalchemy import desc

from .database import MLBacktestJob, SessionLocal
from .engine import run_ml_backtest
from .schemas import MLBacktestCreate, ResolvedMLBacktestCreate
from .settings_service import (
    RuntimeParameters,
    get_runtime_parameters,
    get_runtime_parameters_from_snapshot,
)

logger = logging.getLogger(__name__)
_executor: ThreadPoolExecutor | None = None
_executor_workers: int | None = None
_executor_lock = Lock()
_submit_lock = Lock()


def _get_executor(parameters: RuntimeParameters) -> ThreadPoolExecutor:
    global _executor, _executor_workers
    workers = max(1, parameters.integer("worker.max_parallel_jobs"))
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ml-backtest")
            _executor_workers = workers
        elif _executor_workers != workers:
            logger.warning(
                "worker.max_parallel_jobs changed in the database from %s to %s; restart the ML API to apply it.",
                _executor_workers,
                workers,
            )
        return _executor


def _serialize_job(job: MLBacktestJob, include_result: bool = False) -> dict:
    payload = {
        "id": job.id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "symbol": job.symbol,
        "decision_timeframe": job.decision_timeframe,
        "trend_timeframe": job.trend_timeframe,
        "period_years": job.period_years,
        "initial_capital": job.initial_capital,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
    }
    if include_result:
        payload["request"] = json.loads(job.request_json)
        payload["settings_snapshot"] = (
            json.loads(job.settings_snapshot_json) if job.settings_snapshot_json else None
        )
        payload["result"] = json.loads(job.result_json) if job.result_json else None
    return payload


def _resolve_request(request: MLBacktestCreate, parameters: RuntimeParameters) -> ResolvedMLBacktestCreate:
    values = {
        "symbol": request.symbol or parameters.string("backtest.default_symbol"),
        "period_years": (
            request.period_years
            if request.period_years is not None
            else parameters.integer("backtest.default_period_years")
        ),
        "decision_timeframe": (
            request.decision_timeframe
            or parameters.string("backtest.default_decision_timeframe")
        ),
        "trend_timeframe": (
            request.trend_timeframe
            or parameters.string("backtest.default_trend_timeframe")
        ),
        "initial_capital": (
            request.initial_capital
            if request.initial_capital is not None
            else parameters.floating("backtest.default_initial_capital")
        ),
        "fee_rate": (
            request.fee_rate
            if request.fee_rate is not None
            else parameters.floating("backtest.default_fee_rate")
        ),
        "slippage_rate": (
            request.slippage_rate
            if request.slippage_rate is not None
            else parameters.floating("backtest.default_slippage_rate")
        ),
        "entry_threshold": (
            request.entry_threshold
            if request.entry_threshold is not None
            else parameters.floating("backtest.default_entry_threshold")
        ),
        "exit_threshold": (
            request.exit_threshold
            if request.exit_threshold is not None
            else parameters.floating("backtest.default_exit_threshold")
        ),
        "reinvestment": (
            request.reinvestment
            if request.reinvestment is not None
            else parameters.boolean("backtest.default_reinvestment")
        ),
    }
    resolved = ResolvedMLBacktestCreate.model_validate(values)

    symbol_minimum = parameters.integer("backtest.symbol_min_length")
    symbol_maximum = parameters.integer("backtest.symbol_max_length")
    if not symbol_minimum <= len(resolved.symbol) <= symbol_maximum:
        raise ValueError(
            f"symbol length must be between {symbol_minimum} and {symbol_maximum} characters"
        )
    allowed_decision_timeframes = {
        str(value) for value in parameters.list("backtest.allowed_decision_timeframes")
    }
    allowed_trend_timeframes = {
        str(value) for value in parameters.list("backtest.allowed_trend_timeframes")
    }
    if resolved.decision_timeframe not in allowed_decision_timeframes:
        raise ValueError("decision_timeframe is not enabled in the database")
    if resolved.trend_timeframe not in allowed_trend_timeframes:
        raise ValueError("trend_timeframe is not enabled in the database")

    minimum_years = parameters.integer("backtest.minimum_period_years")
    maximum_years = parameters.integer("market.max_backtest_years")
    if not minimum_years <= resolved.period_years <= maximum_years:
        raise ValueError(f"period_years must be between {minimum_years} and {maximum_years}")
    if not 0 < resolved.initial_capital <= parameters.floating("backtest.maximum_initial_capital"):
        raise ValueError("initial_capital is outside the database limits")
    if not 0 <= resolved.fee_rate <= parameters.floating("backtest.maximum_fee_rate"):
        raise ValueError("fee_rate is outside the database limits")
    if not 0 <= resolved.slippage_rate <= parameters.floating("backtest.maximum_slippage_rate"):
        raise ValueError("slippage_rate is outside the database limits")
    threshold_minimum = parameters.floating("backtest.minimum_probability_threshold")
    threshold_maximum = parameters.floating("backtest.maximum_probability_threshold")
    if not threshold_minimum <= resolved.entry_threshold <= threshold_maximum:
        raise ValueError("entry_threshold is outside the database limits")
    if not threshold_minimum <= resolved.exit_threshold <= threshold_maximum:
        raise ValueError("exit_threshold is outside the database limits")
    return resolved


def create_job(request: MLBacktestCreate) -> dict:
    parameters = get_runtime_parameters()
    resolved = _resolve_request(request, parameters)
    now = datetime.now(timezone.utc)
    job = MLBacktestJob(
        id=str(uuid4()),
        status="QUEUED",
        stage="Queued",
        progress=0,
        symbol=resolved.symbol,
        decision_timeframe=resolved.decision_timeframe,
        trend_timeframe=resolved.trend_timeframe,
        period_years=resolved.period_years,
        initial_capital=resolved.initial_capital,
        request_json=resolved.model_dump_json(),
        settings_snapshot_json=json.dumps(parameters.snapshot(), separators=(",", ":")),
        created_at=now,
    )
    with SessionLocal.begin() as session:
        session.add(job)
    with _submit_lock:
        _get_executor(parameters).submit(_run_job, job.id)
    return _serialize_job(job)


def list_jobs(limit: int) -> list[dict]:
    with SessionLocal() as session:
        jobs = session.query(MLBacktestJob).order_by(desc(MLBacktestJob.created_at)).limit(limit).all()
        return [_serialize_job(job) for job in jobs]


def get_job(job_id: str, include_result: bool = True) -> dict | None:
    with SessionLocal() as session:
        job = session.get(MLBacktestJob, job_id)
        return _serialize_job(job, include_result=include_result) if job else None


def cancel_job(job_id: str) -> dict | None:
    with SessionLocal.begin() as session:
        job = session.get(MLBacktestJob, job_id)
        if not job:
            return None
        if job.status in {"QUEUED", "RUNNING"}:
            job.cancel_requested = 1
            job.stage = "Cancellation requested"
        return _serialize_job(job)


def _is_cancelled(job_id: str) -> bool:
    with SessionLocal() as session:
        job = session.get(MLBacktestJob, job_id)
        return not job or bool(job.cancel_requested)


def _update_progress(job_id: str, progress: int, stage: str) -> None:
    with SessionLocal.begin() as session:
        job = session.get(MLBacktestJob, job_id)
        if not job:
            return
        job.progress = max(job.progress, min(100, int(progress)))
        job.stage = stage


def _run_job(job_id: str) -> None:
    with SessionLocal.begin() as session:
        job = session.get(MLBacktestJob, job_id)
        if not job:
            return
        job.status = "RUNNING"
        job.stage = "Starting"
        job.progress = 1
        job.started_at = datetime.now(timezone.utc)
        request = ResolvedMLBacktestCreate.model_validate_json(job.request_json)
        snapshot = json.loads(job.settings_snapshot_json or "{}")
        parameters = get_runtime_parameters_from_snapshot(snapshot)

    try:
        result = run_ml_backtest(
            request,
            parameters,
            lambda value, stage: _update_progress(job_id, value, stage),
            lambda: _is_cancelled(job_id),
        )
        with SessionLocal.begin() as session:
            job = session.get(MLBacktestJob, job_id)
            if not job:
                return
            if job.cancel_requested:
                job.status = "CANCELLED"
                job.stage = "Cancelled"
            else:
                job.status = "COMPLETED"
                job.stage = "Completed"
                job.progress = 100
                job.result_json = json.dumps(result, separators=(",", ":"), allow_nan=False)
            job.completed_at = datetime.now(timezone.utc)
    except InterruptedError:
        with SessionLocal.begin() as session:
            job = session.get(MLBacktestJob, job_id)
            if job:
                job.status = "CANCELLED"
                job.stage = "Cancelled"
                job.completed_at = datetime.now(timezone.utc)
    except Exception as exc:
        logger.exception("ML backtest %s failed", job_id)
        with SessionLocal.begin() as session:
            job = session.get(MLBacktestJob, job_id)
            if job:
                job.status = "FAILED"
                job.stage = "Failed"
                job.error_message = str(exc)
                job.completed_at = datetime.now(timezone.utc)
