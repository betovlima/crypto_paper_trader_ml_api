"""Persisted BTC forecast replay, sourced only from immutable job audit ZIPs.

No local file paths, exchange requests, or command-line scripts are involved.
The UI selects an existing completed database job and the worker retrieves
its exact archived OHLCV candles and temporal settings.
"""
from __future__ import annotations

import csv
from hashlib import sha256
from io import BytesIO, TextIOWrapper
import json
from zipfile import BadZipFile, ZipFile

import pandas as pd

from .schemas import ResolvedMLBacktestCreate


REQUIRED_MEMBERS = frozenset({
    "manifest.json", "request.json", "parameters_snapshot.json",
    "forecast_metrics.json", "forecast_predictions_oos.csv", "folds.json",
    "market_history_decision.csv", "market_history_trend.csv",
})
GROUND_TRUTH = (
    "forecast_horizon_hours", "timestamp", "forecast_future_at_utc",
    "forecast_realized_return", "forecast_realized_up",
)
FOLD_FIELDS = (
    "fold", "horizon_hours", "purge_candles_at_each_boundary",
    "training_rows", "training_start", "training_end", "training_last_label_at",
    "early_stopping_rows", "early_stopping_start", "early_stopping_end",
    "early_stopping_last_label_at",
    "calibration_rows", "calibration_start", "calibration_end",
    "calibration_last_label_at",
    "test_rows", "test_start", "test_end", "test_last_label_at",
)
METRICS = (
    "roc_auc", "brier_skill_score", "return_mae",
    "return_mae_skill_vs_zero", "return_rmse",
)


def _json(z: ZipFile, name: str):
    return json.loads(z.read(name))


def inspect_source_archive(contents: bytes) -> tuple[ResolvedMLBacktestCreate, dict]:
    """Validate that the selected completed job contains a replayable forecast."""
    try:
        with ZipFile(BytesIO(contents)) as z:
            if not REQUIRED_MEMBERS.issubset(z.namelist()):
                raise ValueError("A execução selecionada não possui uma auditoria BTC Forecast completa")
            manifest = _json(z, "manifest.json")
            request = ResolvedMLBacktestCreate.model_validate(_json(z, "request.json"))
            snapshot = _json(z, "parameters_snapshot.json")
            if (manifest.get("kind") != "BTC_DIRECTION_RETURN_OOS_AUDIT"
                    or request.experiment_type != "BTC_DIRECTION_RETURN"
                    or request.symbol != "BTCUSDT"):
                raise ValueError("Selecione uma execução concluída de BTC Direction & Return")
            if (manifest.get("symbol") != request.symbol
                    or manifest.get("history_source") != request.history_source
                    or manifest.get("decision_timeframe") != request.decision_timeframe
                    or manifest.get("trend_timeframe") != request.trend_timeframe):
                raise ValueError("Fonte, símbolo ou timeframes divergem no ZIP original")
            if not isinstance(snapshot, dict) or not snapshot:
                raise ValueError("Snapshot de parâmetros ausente na auditoria original")
            return request, snapshot
    except (BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("A auditoria original não é um ZIP de previsão válido") from exc


def _history(blob: bytes, name: str) -> pd.DataFrame:
    history = pd.read_csv(
        BytesIO(blob), float_precision="round_trip",
        parse_dates=["timestamp"],
    )
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
    if (history.empty or history["timestamp"].isna().any()
            or history["timestamp"].duplicated().any()
            or not history["timestamp"].is_monotonic_increasing):
        raise ValueError(f"Os timestamps dos candles arquivados são inválidos: {name}")
    return history


class ArchivedMarketClient:
    def __init__(self, contents: bytes, request: ResolvedMLBacktestCreate):
        original_request, _ = inspect_source_archive(contents)
        if original_request != request:
            raise ValueError("A configuração do replay difere da execução original")
        self.replay_source_sha256 = sha256(contents).hexdigest()
        with ZipFile(BytesIO(contents)) as z:
            manifest = _json(z, "manifest.json")
            decision_blob = z.read("market_history_decision.csv")
            trend_blob = z.read("market_history_trend.csv")
        decision = _history(decision_blob, "decision")
        trend = _history(trend_blob, "trend")
        self.market_bytes = {
            "decision": decision_blob,
            "trend": trend_blob,
        }
        self.candles = {
            "decision": decision,
            "trend": trend,
        }
        self.request = request
        self.archive_manifest = manifest.get("market_archive_checksums")
        self.market_interruptions = manifest.get("market_interruptions", {})

    def fetch_history(self, *, symbol: str, timeframe: str, period_years: int,
                      progress=None, cancelled=None) -> pd.DataFrame:
        if cancelled and cancelled():
            raise InterruptedError("Replay cancelado")
        if symbol != self.request.symbol or period_years != self.request.period_years:
            raise ValueError("A solicitação diverge do histórico arquivado")
        if timeframe == self.request.decision_timeframe:
            result = self.candles["decision"]
        elif timeframe == self.request.trend_timeframe:
            result = self.candles["trend"]
        else:
            raise ValueError(f"Timeframe {timeframe} indisponível na auditoria")
        if progress:
            progress(100)
        return result.copy(deep=True)

    def close(self) -> None:
        pass


def _digest_stream(z: ZipFile, path: str) -> str:
    digest = sha256()
    with z.open(path) as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _truth_and_diversity(z: ZipFile) -> tuple[str, dict[tuple[int, int], int]]:
    digest = sha256()
    unique: dict[tuple[int, int], set[str]] = {}
    with z.open("forecast_predictions_oos.csv") as stream:
        reader = csv.DictReader(TextIOWrapper(stream, encoding="utf-8-sig"))
        if not set(GROUND_TRUTH).issubset(reader.fieldnames or []):
            raise ValueError("A auditoria não contém todos os rótulos futuros")
        for row in reader:
            digest.update(("\x1f".join(row[name] for name in GROUND_TRUTH) + "\n").encode())
            if "forecast_fold" in row and "forecast_predicted_return" in row:
                key = (int(float(row["forecast_horizon_hours"])), int(float(row["forecast_fold"])))
                unique.setdefault(key, set()).add(row["forecast_predicted_return"])
    return digest.hexdigest(), {key: len(value) for key, value in unique.items()}


def _fold_windows(z: ZipFile) -> list[dict]:
    return [
        {name: item.get(name) for name in FOLD_FIELDS}
        for item in _json(z, "folds.json")
    ]


def compare_archives(original: bytes, candidate: bytes) -> dict:
    """Compare OOS metrics only if raw candles, targets and folds are identical."""
    with ZipFile(BytesIO(original)) as v1, ZipFile(BytesIO(candidate)) as v2:
        for z in (v1, v2):
            if not REQUIRED_MEMBERS.issubset(z.namelist()):
                raise ValueError("Auditoria incompleta; comparação não permitida")
            if _json(z, "manifest.json").get("kind") != "BTC_DIRECTION_RETURN_OOS_AUDIT":
                raise ValueError("As auditorias precisam ser BTC Direction & Return")
        r1, r2 = _json(v1, "request.json"), _json(v2, "request.json")
        fields = (
            "symbol", "history_source", "period_years",
            "decision_timeframe", "trend_timeframe", "experiment_type",
        )
        if any(r1.get(name) != r2.get(name) for name in fields):
            raise ValueError("Configurações históricas diferentes; comparação recusada")
        for name in ("market_history_decision.csv", "market_history_trend.csv"):
            if _digest_stream(v1, name) != _digest_stream(v2, name):
                raise ValueError(f"Candles históricos diferentes: {name}")
        if _fold_windows(v1) != _fold_windows(v2):
            raise ValueError("Fronteiras temporais diferentes; comparação recusada")
        truth1, variety1 = _truth_and_diversity(v1)
        truth2, variety2 = _truth_and_diversity(v2)
        if truth1 != truth2:
            raise ValueError("Rótulos ou timestamps fora da amostra diferentes")
        m1 = {int(item["horizon_hours"]): item["metrics"] for item in _json(v1, "forecast_metrics.json")}
        m2 = {int(item["horizon_hours"]): item["metrics"] for item in _json(v2, "forecast_metrics.json")}
        if m1.keys() != m2.keys():
            raise ValueError("Horizontes diferentes; comparação recusada")
        horizons = []
        for horizon in sorted(m1):
            old, new = m1[horizon], m2[horizon]
            if old.get("rows") != new.get("rows"):
                raise ValueError("Número de previsões divergente")
            changes = {
                name: {
                    "v1": old.get(name), "v2": new.get(name),
                    "delta": new[name] - old[name] if old.get(name) is not None and new.get(name) is not None else None,
                }
                for name in METRICS
            }
            folds = [
                {
                    "fold": fold,
                    "v1_distinct_returns": variety1.get((horizon, fold)),
                    "v2_distinct_returns": variety2.get((horizon, fold)),
                }
                for fold in sorted({f for h, f in variety1.keys() | variety2.keys() if h == horizon})
            ]
            horizons.append({
                "horizon_hours": horizon, "rows": new.get("rows"),
                "metrics": changes, "fold_return_diversity": folds,
            })
        return {
            "status": "VERIFIED",
            "same_market_history": True, "same_oos_targets": True,
            "same_fold_windows": True, "history_source": r1["history_source"],
            "horizons": horizons,
            "warning": "Comparação preditiva, ainda sem simulação financeira BTC/CASH.",
        }
