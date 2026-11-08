"""Causal, auditable BTC/CASH paper capitalization on immutable OOS archives.

No price downloads, retraining, real brokerage, or future targets in decisions.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO, TextIOWrapper
import math
from zipfile import ZipFile

from .archive_replay import compare_archives
from .mexc import TIMEFRAME_SECONDS


def _utc(value: str) -> datetime:
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamp sem fuso horário no arquivo de auditoria")
    return result.astimezone(timezone.utc)


def _positive(value: str, field: str) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Campo inválido na auditoria: {field}") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"Valor inválido na auditoria: {field}")
    return number


def _history(archive: bytes) -> tuple[list[dict], int]:
    with ZipFile(BytesIO(archive)) as z:
        import json
        request = json.loads(z.read("request.json"))
        seconds = TIMEFRAME_SECONDS[request["decision_timeframe"]]
        with z.open("market_history_decision.csv") as stream:
            rows = list(csv.DictReader(TextIOWrapper(stream, encoding="utf-8-sig")))
    candles = [
        {
            "timestamp": _utc(row["timestamp"]),
            "open": _positive(row["open"], "open"),
            "close": _positive(row["close"], "close"),
        }
        for row in rows
    ]
    if len(candles) < 3 or any(
        candles[i]["timestamp"] >= candles[i + 1]["timestamp"]
        for i in range(len(candles) - 1)
    ):
        raise ValueError("Histórico de candles inválido ou insuficiente")
    return candles, seconds


def _signals(archive: bytes, horizon: int, candles: list[dict], seconds: int) -> dict:
    """Place each close-known prediction at the NEXT bar's open, never earlier."""
    with ZipFile(BytesIO(archive)) as z:
        with z.open("forecast_predictions_oos.csv") as stream:
            rows = list(csv.DictReader(TextIOWrapper(stream, encoding="utf-8-sig")))
    indices = {c["timestamp"]: i for i, c in enumerate(candles)}
    signals = {}
    previous = None
    for row in rows:
        if int(float(row["forecast_horizon_hours"])) != horizon:
            continue
        timestamp = _utc(row["timestamp"])
        if previous is not None and timestamp <= previous:
            raise ValueError("Previsões OOS não estão em ordem temporal")
        previous = timestamp
        index = indices.get(timestamp)
        if index is None:
            raise ValueError("Previsão sem candle correspondente no histórico")
        if index + 1 >= len(candles):
            continue
        next_candle = candles[index + 1]
        if next_candle["timestamp"] - timestamp != timedelta(seconds=seconds):
            # A missing bar is never synthesized, and stale signals are not executed.
            continue
        estimate = float(row["forecast_predicted_return"])
        p_up = float(row["forecast_p_up"])
        if not math.isfinite(estimate) or not math.isfinite(p_up) or not 0 <= p_up <= 1:
            raise ValueError("Previsão ou probabilidade inválida na auditoria")
        if index + 1 in signals:
            raise ValueError("Duas previsões para o mesmo candle de execução")
        signals[index + 1] = {
            "predicted_return": estimate, "p_up": p_up,
            "signal_at": (timestamp + timedelta(seconds=seconds)).isoformat(),
        }
    if not signals:
        raise ValueError(f"Não há sinais executáveis para {horizon}h")
    return signals


def _summary_series(
    candles: list[dict], signals: dict, *, start: int, end: int,
    capital: float, fee: float, slippage: float, edge: float,
    min_probability: float, exit_return: float, baseline: bool, label: str,
) -> dict:
    cash, quantity = capital, 0.0
    actions, points = [], []
    switches, sells, exposed = 0, 0, 0
    peak, max_drawdown = capital, 0.0
    # Required gross price appreciation to cover the entire buy/sell round trip.
    round_trip = (1 + slippage) / ((1 - fee) ** 2 * (1 - slippage)) - 1

    def trade(side: str, price: float, at: datetime, signal: dict | None, reason: str):
        nonlocal cash, quantity, switches, sells
        if side == "BUY":
            if cash <= 0:
                return
            paid = price * (1 + slippage)
            quantity = cash * (1 - fee) / paid
            cash = 0.0
        else:
            if quantity <= 0:
                return
            cash = quantity * price * (1 - slippage) * (1 - fee)
            quantity = 0.0
            sells += 1
        switches += 1
        actions.append({
            "timestamp": at.isoformat(), "action": side, "price": price,
            "reason": reason, "signal_at": signal["signal_at"] if signal else None,
            "predicted_return": signal["predicted_return"] if signal else None,
            "probability_up": signal["p_up"] if signal else None,
            "capital_after": cash + quantity * price,
        })

    for index in range(start, end + 1):
        candle = candles[index]
        signal = signals.get(index)
        if baseline and index == start:
            trade("BUY", candle["open"], candle["timestamp"], None, "BUY_AND_HOLD")
        elif not baseline and signal:
            if quantity == 0 and (
                signal["predicted_return"] > round_trip + edge
                and signal["p_up"] >= min_probability
            ):
                trade("BUY", candle["open"], candle["timestamp"], signal, "FORECAST_ENTRY")
            elif quantity > 0 and (
                signal["predicted_return"] <= exit_return
                or signal["p_up"] < min_probability
            ):
                trade("SELL", candle["open"], candle["timestamp"], signal, "FORECAST_EXIT")
        if quantity > 0:
            exposed += 1
        equity = cash + quantity * candle["close"]
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1)
        points.append({
            "timestamp": (candle["timestamp"] + timedelta(seconds=1)).isoformat(),
            "equity": equity, "position": "BTC" if quantity > 0 else "CASH",
        })

    # Closing liquidation is executed at the last known close, not a future candle.
    if quantity > 0:
        candle = candles[end]
        trade("SELL", candle["close"],
              candle["timestamp"] + timedelta(seconds=1),
              None, "END_OF_TEST")
        points[-1]["equity"] = cash
        points[-1]["position"] = "CASH"
        peak = max(peak, cash)
        max_drawdown = min(max_drawdown, cash / peak - 1)

    duration_days = max(
        (candles[end]["timestamp"] - candles[start]["timestamp"]).total_seconds() / 86400,
        1 / 48,
    )
    rate = cash / capital - 1
    cagr = (cash / capital) ** (365.25 / duration_days) - 1
    stride = max(1, math.ceil(len(points) / 500))
    sampled = [p for i, p in enumerate(points) if i % stride == 0 or i == len(points) - 1]
    return {
        "code": label,
        "initial_capital": capital,
        "capital_final": cash,
        "total_return": rate,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "exposure": exposed / len(points),
        "switch_count": switches,
        "completed_cycles": sells,
        "required_round_trip_return": round_trip,
        "timeline": sampled,
        "actions": actions[-150:],
        "action_count": len(actions),
        "actions_truncated": len(actions) > 150,
    }


def compare_capitalization(
    original: bytes, replay: bytes, *,
    horizon: int, initial_capital: float, fee_rate: float,
    slippage_rate: float, entry_edge: float,
    min_probability: float, exit_return: float,
) -> dict:
    if horizon not in {2, 6, 24}:
        raise ValueError("Horizonte precisa ser 2h, 6h ou 24h")
    if not (math.isfinite(initial_capital) and initial_capital > 0):
        raise ValueError("Capital inicial inválido")
    if any(not math.isfinite(v) for v in (
        fee_rate, slippage_rate, entry_edge, min_probability, exit_return
    )) or not (
        0 <= fee_rate <= .05 and 0 <= slippage_rate <= .05
        and 0 <= entry_edge <= .2 and 0 <= min_probability <= 1
        and -.2 <= exit_return <= .2
    ):
        raise ValueError("Parâmetros da simulação fora dos limites permitidos")
    verified = compare_archives(original, replay)
    candles, seconds = _history(original)
    source = _signals(original, horizon, candles, seconds)
    candidate = _signals(replay, horizon, candles, seconds)
    if source.keys() != candidate.keys():
        raise ValueError("Sinais executáveis V1 e V2 em timestamps diferentes")
    start, end = min(source), max(source)
    if start == end:
        raise ValueError("Período OOS insuficiente para calcular capitalização")
    common = dict(start=start, end=end, capital=initial_capital, fee=fee_rate,
                  slippage=slippage_rate, edge=entry_edge,
                  min_probability=min_probability, exit_return=exit_return)
    results = [
        _summary_series(candles, source, baseline=False, label="V1", **common),
        _summary_series(candles, candidate, baseline=False, label="V2", **common),
        _summary_series(candles, {}, baseline=True, label="BUY_AND_HOLD", **common),
    ]
    return {
        "status": "VERIFIED",
        "comparison": {k: verified[k] for k in (
            "same_market_history", "same_oos_targets", "same_fold_windows"
        )},
        "horizon_hours": horizon,
        "period_start": candles[start]["timestamp"].isoformat(),
        "period_end": candles[end]["timestamp"].isoformat(),
        "signal_count": len(source),
        "original_archive_sha256": sha256(original).hexdigest(),
        "replay_archive_sha256": sha256(replay).hexdigest(),
        "parameters": {
            "initial_capital": initial_capital, "fee_rate": fee_rate,
            "slippage_rate": slippage_rate, "entry_edge": entry_edge,
            "min_probability": min_probability, "exit_return": exit_return,
        },
        "results": results,
        "note": "Simulação histórica OOS, sem execução real nem otimização nos testes.",
    }
