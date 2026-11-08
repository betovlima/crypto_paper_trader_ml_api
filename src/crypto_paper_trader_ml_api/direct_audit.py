"""Untruncated, immutable-at-execution audit for BTC direct forecast research."""

from __future__ import annotations

from io import BytesIO
import json
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from .schemas import ResolvedMLBacktestCreate
from .settings_service import RuntimeParameters


def _json(value: object) -> bytes:
    return json.dumps(
        value, indent=2, ensure_ascii=False,
        allow_nan=False, default=str,
    ).encode("utf-8")


def _csv(frame: pd.DataFrame) -> bytes:
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[column]):
            result[column] = (
                pd.to_datetime(result[column], utc=True)
                .dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            )
    return result.to_csv(index=False).encode("utf-8-sig")


def build_direction_audit_archive(
    *, predictions: list[pd.DataFrame], decision_history: pd.DataFrame,
    trend_history: pd.DataFrame, request: ResolvedMLBacktestCreate,
    parameters: RuntimeParameters, result: dict,
    source_archives: dict | None,
) -> bytes:
    """Save every out-of-sample prediction as generated, with no reconstructed labels."""
    expected_horizons = [int(h) for h in parameters.list("forecast.horizons_hours")]
    if len(predictions) != len(expected_horizons):
        raise ValueError("Direct forecast audit requires predictions for every configured horizon")
    if any(frame.empty for frame in predictions):
        raise ValueError("Direct forecast audit cannot be empty")
    ordered = pd.concat(predictions, ignore_index=True).sort_values(
        ["forecast_horizon_hours", "timestamp"]
    )
    actual_horizons = sorted(set(ordered["forecast_horizon_hours"].astype(int)))
    if actual_horizons != sorted(expected_horizons):
        raise ValueError("Direct forecast predictions do not match configured horizons")

    counts = []
    for horizon in expected_horizons:
        frame = ordered.loc[ordered["forecast_horizon_hours"] == horizon]
        if frame["timestamp"].duplicated().any():
            raise ValueError(f"Duplicate OOS forecast for horizon {horizon}h")
        counts.append({
            "horizon_hours": horizon,
            "predictions": len(frame),
            "oos_start": pd.Timestamp(frame.iloc[0]["timestamp"]).isoformat(),
            "oos_end": pd.Timestamp(frame.iloc[-1]["timestamp"]).isoformat(),
        })

    result_horizons = result["forecast"]["horizons"]
    if [item["horizon_hours"] for item in result_horizons] != expected_horizons:
        raise ValueError("Forecast report horizons mismatch")

    manifest = {
        "schema_version": 1,
        "kind": "BTC_DIRECTION_RETURN_OOS_AUDIT",
        "experiment_type": request.experiment_type,
        "research_only": True,
        "financial_trading_enabled": False,
        "history_source": request.history_source,
        "symbol": request.symbol,
        "decision_timeframe": request.decision_timeframe,
        "trend_timeframe": request.trend_timeframe,
        "horizons": counts,
        "fold_boundaries": "folds.json",
        "market_interruptions": result["execution"].get("market_interruptions", {}),
        "market_archive_checksums": source_archives,
        "prediction_target": "P(close[t+h] > close[t]) and simple close-to-close return",
        "model_selection": "chronological train / early-stop / calibrate / purged OOS",
        "replay_original_audit_sha256": result["execution"].get(
            "replay_original_audit_sha256"
        ),
        "regression_target_scale": result.get("insights", {}).get(
            "regression_target_scale"
        ),
        "diagnostics_version": 2 if result.get("insights", {}).get(
            "model_diagnostics"
        ) else 1,
        "unsampled": True,
    }
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("manifest.json", _json(manifest))
        archive.writestr("request.json", _json(request.model_dump(mode="json")))
        archive.writestr("parameters_snapshot.json", _json(parameters.snapshot()))
        archive.writestr("forecast_metrics.json", _json(result["forecast"]["horizons"]))
        archive.writestr("folds.json", _json(result["execution"]["fold_windows"]))
        archive.writestr("history_coverage.json", _json(result["execution"]["history_coverage"]))
        archive.writestr("market_interruptions.json", _json(
            result["execution"].get("market_interruptions", {})
        ))
        archive.writestr("forecast_predictions_oos.csv", _csv(ordered))
        archive.writestr("market_history_decision.csv", _csv(decision_history))
        archive.writestr("market_history_trend.csv", _csv(trend_history))
        archive.writestr("README.txt", (
            "BTC DIRECTION / RETURN · AUDITORIA FORA DA AMOSTRA\n"
            "Este experimento NAO realiza compras ou vendas de BTC.\n"
            "forecast_predictions_oos.csv: probabilidades e retornos previstos,\n"
            "precos futuros realizados, horizontes, features e fold original.\n"
            "forecast_metrics.json: comparacoes contra frequencia do treino,\n"
            "retorno medio do treino e previsao de retorno zero.\n"
            "folds.json: fronteiras de treino, early stopping, calibracao\n"
            "e teste, com horizonte purgado entre todas as particoes.\n"
            "market_history_*.csv: candles originais da unica fonte escolhida.\n"
            "Dados de outro exchange NUNCA completam a serie historica.\n"
            "Resultados BTC/CASH e CARO pertencem a etapas posteriores.\n"
        ).encode("utf-8"))
    return buffer.getvalue()
