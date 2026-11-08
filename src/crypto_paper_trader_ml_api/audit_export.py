"""Self-contained audit exports for completed BTC Forecast / ML backtests.

Audit data is generated from the exact out-of-sample simulation before the job
finishes. Never reconstruct historical predictions using today's fitted model.
"""

from __future__ import annotations

from io import BytesIO, StringIO
import csv
import json
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from .rotation_analysis import build_rotation_timeline
from .schemas import ResolvedMLBacktestCreate
from .settings_service import RuntimeParameters


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def _csv_bytes(rows: list[dict], columns: list[str]) -> bytes:
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def build_audit_archive(
    *,
    predictions: pd.DataFrame,
    simulation: dict,
    request: ResolvedMLBacktestCreate,
    parameters: RuntimeParameters,
    result: dict,
    model_metadata: dict,
    decision_history: pd.DataFrame | None = None,
    trend_history: pd.DataFrame | None = None,
) -> bytes:
    """Export OOS evidence and, when available, raw model-training histories.

    Unlike the UI response, these files are intentionally NOT sampled or truncated.
    """
    simulated_positions = simulation["rotation_rows"]
    complete_trace = build_rotation_timeline(
        simulated_positions, max(1, len(simulated_positions))
    )
    if len(complete_trace) != len(predictions):
        raise ValueError("The audit trace must contain exactly one row per OOS prediction")

    position_rows = []
    order_rows = []
    equity_peak = float("-inf")
    for index, point in enumerate(complete_trace):
        equity = float(point["equity"])
        equity_peak = max(equity_peak, equity)
        position_rows.append({
            "timestamp_utc": point["timestamp"],
            "btc_close": point["btc_close"],
            "peak_close_so_far": point["peak_close_so_far"],
            "distance_to_price_peak": point["distance_to_peak"],
            "position_at_candle_close": point["position"],
            "equity": equity,
            "equity_peak_so_far": equity_peak,
            "equity_drawdown": equity / equity_peak - 1 if equity_peak > 0 else None,
            "buy_and_hold_equity": point["buy_hold"],
            "action_count": len(point["actions"]),
        })
        for action_index, action in enumerate(point["actions"]):
            reason = action["reason"]
            order_rows.append({
                "timestamp_utc": point["timestamp"],
                "candle_index": index,
                "action_index": action_index,
                "action": action["action"],
                "execution_price": action["execution_price"],
                "execution_timing": (
                    "CANDLE_CLOSE" if reason == "END_OF_TEST"
                    else "INTRABAR" if reason in {"PROTECTIVE_STOP", "TARGET"}
                    else "CANDLE_OPEN"
                ),
                "reason": reason,
                "position_at_candle_close": point["position"],
                "distance_to_peak_at_execution": action["distance_to_peak_at_execution"],
                "distance_to_peak_at_candle_close": point["distance_to_peak"],
            })

    manifest = {
        "schema_version": 2,
        "kind": "CRYPTO_PAPER_TRADER_ML_AUDIT",
        "model_family": "LightGBM",
        "scope": "OUT_OF_SAMPLE",
        "symbol": request.symbol,
        "market_history_source": request.history_source,
        "market_archive_checksums": model_metadata.get("source_archives"),
        "market_interruptions": model_metadata.get("market_interruptions", {}),
        "decision_timeframe": request.decision_timeframe,
        "trend_timeframe": request.trend_timeframe,
        "candles": len(complete_trace),
        "executed_actions": len(order_rows),
        "completed_trades": len(simulation["trades"]),
        "period_start": complete_trace[0]["timestamp"] if complete_trace else None,
        "period_end": complete_trace[-1]["timestamp"] if complete_trace else None,
        "position_semantics": "Effective position after every simulated candle; CASH after forced test-end sale",
        "price_peak_semantics": "Highest observed BTC closing price up to each candle, not an ex-post local peak",
        "execution_time_semantics": "Candle timestamp identifies the bar; exact intrabar execution time is unavailable",
        "unsampled": True,
        "raw_history_included": decision_history is not None,
        "history_coverage": model_metadata.get("history_coverage"),
        "fold_windows": model_metadata.get("fold_windows", []),
        "purge_candles": model_metadata.get("label_horizon_candles"),
    }
    predicted_csv = predictions.copy()
    if "timestamp" in predicted_csv.columns:
        predicted_csv["timestamp"] = pd.to_datetime(
            predicted_csv["timestamp"], utc=True
        ).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("manifest.json", _json_bytes(manifest))
        archive.writestr("request.json", _json_bytes(request.model_dump(mode="json")))
        archive.writestr("parameters_snapshot.json", _json_bytes(parameters.snapshot()))
        archive.writestr("model_metadata.json", _json_bytes(model_metadata))
        archive.writestr("report.json", _json_bytes({
            name: value for name, value in result.items()
            if name not in {"equity_curve", "rotation_timeline", "trades"}
        }))
        archive.writestr("predictions_oos.csv", predicted_csv.to_csv(index=False).encode("utf-8-sig"))
        if decision_history is not None:
            archive.writestr(
                "market_history_decision.csv",
                decision_history.to_csv(index=False).encode("utf-8-sig"),
            )
        if trend_history is not None:
            archive.writestr(
                "market_history_trend.csv",
                trend_history.to_csv(index=False).encode("utf-8-sig"),
            )
        archive.writestr("folds.json", _json_bytes(model_metadata.get("fold_windows", [])))
        archive.writestr("history_coverage.json", _json_bytes(model_metadata.get("history_coverage")))
        archive.writestr("market_interruptions.json", _json_bytes(
            model_metadata.get("market_interruptions", {})
        ))

        archive.writestr("positions_full.csv", _csv_bytes(position_rows, [
            "timestamp_utc", "btc_close", "peak_close_so_far",
            "distance_to_price_peak", "position_at_candle_close", "equity",
            "equity_peak_so_far", "equity_drawdown", "buy_and_hold_equity",
            "action_count",
        ]))
        archive.writestr("executions_full.csv", _csv_bytes(order_rows, [
            "timestamp_utc", "candle_index", "action_index", "action",
            "execution_price", "execution_timing", "reason",
            "position_at_candle_close", "distance_to_peak_at_execution",
            "distance_to_peak_at_candle_close",
        ]))
        trades = simulation["trades"]
        archive.writestr("trades_full.csv", _csv_bytes(trades, [
            "entry_time", "exit_time", "entry_price", "exit_price",
            "net_pnl", "return_pct", "reason",
        ]))
        archive.writestr("README.txt", (
            "BTC FORECAST - AUDITORIA DE BACKTEST\n"
            "Todos os CSV contêm dados do período fora da amostra (OOS).\n"
            "predictions_oos.csv: candles, indicadores, rótulos realizados, folds e probabilidades OOS.\n"
            "positions_full.csv: posição efetiva BTC/CASH, preço, patrimônio, drawdown e distância ao maior fechamento conhecido.\n"
            "executions_full.csv: todas as compras e vendas, inclusive duas no mesmo candle.\n"
            "trades_full.csv: todas as operações concluídas, sem o limite da interface.\n"
            "parameters_snapshot.json: parâmetros utilizados no momento do treinamento.\n"
            "request.json: configuração resolvida da execução.\n"
            "report.json: métricas e resultados resumidos.\n"
            "model_metadata.json: importância dos atributos e configuração de validação.\n"
            "As distâncias ao topo são análises retrospectivas; não justificam operações na abertura.\n"
            "O timestamp de stops intrabar identifica o candle, não a hora exata do evento.\n"
            "As probabilidades OOS foram geradas durante a execução original, não reestimadas.\n"
            "folds.json: fronteiras de treino, calibração e teste e intervalos de segurança.\n"
            "history_coverage.json: cobertura real dos dados, lacunas e limite da coleta.\n"
            "market_history_decision.csv e market_history_trend.csv: candles brutos utilizados (quando disponíveis).\n"
            "market_interruptions.json: interrupções oficiais documentadas, horários não negociados e fonte verificável.\n"

        ).encode("utf-8"))
    return buffer.getvalue()
