from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from . import __version__
from .cors import DatabaseCORSMiddleware
from .database import initialize_database
from .schemas import (
    HealthResponse,
    MLBacktestCreate,
    MLBacktestJobDetail,
    MLBacktestJobResponse,
    PublicMLConfigurationResponse,
    RuntimeParametersResponse,
    RuntimeParametersUpdateRequest,
)
from .service import cancel_job, create_job, get_job, list_jobs
from .settings_service import (
    ParameterValidationError,
    get_runtime_parameters,
    list_parameter_rows,
    public_configuration_payload,
    update_parameter_rows,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
initialize_database()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield


app = FastAPI(
    title="Crypto Paper Trader ML API",
    version=__version__,
    description=(
        "Independent XGBoost walk-forward backtest service for Crypto Paper Trader. "
        "All runtime, model, market, feature and simulation parameters are stored in its database."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    DatabaseCORSMiddleware,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="crypto-paper-trader-ml-api",
        version=__version__,
        parameter_source="DATABASE",
    )


@app.get("/v1/ml-configuration", response_model=PublicMLConfigurationResponse)
def public_ml_configuration() -> dict:
    return public_configuration_payload()


@app.get("/v1/ml-settings", response_model=RuntimeParametersResponse)
def ml_settings() -> dict:
    return {"source": "DATABASE", "parameters": list_parameter_rows()}


@app.put("/v1/ml-settings", response_model=RuntimeParametersResponse)
def update_ml_settings(request: RuntimeParametersUpdateRequest) -> dict:
    try:
        rows = update_parameter_rows([item.model_dump() for item in request.parameters])
    except ParameterValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"source": "DATABASE", "parameters": rows}


@app.post("/v1/ml-backtests", response_model=MLBacktestJobResponse, status_code=202)
def start_backtest(request: MLBacktestCreate) -> dict:
    try:
        return create_job(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/v1/ml-backtests", response_model=list[MLBacktestJobResponse])
def recent_backtests(limit: int | None = Query(default=None, ge=1)) -> list[dict]:
    parameters = get_runtime_parameters()
    resolved_limit = (
        int(limit)
        if limit is not None
        else parameters.integer("api.recent_jobs_default_limit")
    )
    maximum_limit = parameters.integer("api.recent_jobs_maximum_limit")
    if resolved_limit > maximum_limit:
        raise HTTPException(
            status_code=422,
            detail=f"limit exceeds the database maximum of {maximum_limit}",
        )
    return list_jobs(resolved_limit)


@app.get("/v1/ml-backtests/{job_id}", response_model=MLBacktestJobDetail)
def backtest_detail(job_id: str) -> dict:
    job = get_job(job_id, include_result=True)
    if not job:
        raise HTTPException(status_code=404, detail="ML backtest not found")
    return job


@app.get("/v1/ml-backtests/{job_id}/progress", response_model=MLBacktestJobResponse)
def backtest_progress(job_id: str) -> dict:
    job = get_job(job_id, include_result=False)
    if not job:
        raise HTTPException(status_code=404, detail="ML backtest not found")
    return job


@app.get("/v1/ml-backtests/{job_id}/results")
def backtest_results(job_id: str) -> dict:
    job = get_job(job_id, include_result=True)
    if not job:
        raise HTTPException(status_code=404, detail="ML backtest not found")
    if job["status"] != "COMPLETED":
        raise HTTPException(status_code=409, detail="ML backtest has not completed")
    return job["result"]


@app.get("/v1/ml-backtests/{job_id}/trades")
def backtest_trades(job_id: str) -> list[dict]:
    job = get_job(job_id, include_result=True)
    if not job:
        raise HTTPException(status_code=404, detail="ML backtest not found")
    return (job.get("result") or {}).get("trades", [])


@app.delete("/v1/ml-backtests/{job_id}", response_model=MLBacktestJobResponse)
def request_cancellation(job_id: str) -> dict:
    job = cancel_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="ML backtest not found")
    return job
