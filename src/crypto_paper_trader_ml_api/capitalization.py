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
from .btc_policy import aggregate_signals, simulate_rotations, validate_policy


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


def _joint_signals(signals_by_horizon: dict[int, dict]) -> dict:
    """One causal BTC/CASH signal from 2h/6h/24h OOS predictions.

    Only indices for which ALL horizons are available are eligible. Normalize
    predicted returns to hourly rates before combining them into a six-hour
    reference return. Never use future realized targets or candle closes.
    """
    horizons = (2, 6, 24)
    if any(not signals_by_horizon.get(h) for h in horizons):
        raise ValueError("Auditoria não contém previsões executáveis nos três horizontes")
    common = set.intersection(
        *(set(signals_by_horizon[h]) for h in horizons)
    )
    if len(common) < 2:
        raise ValueError("Não há pelo menos dois candles com previsões OOS simultâneas para 2h, 6h e 24h")
    output = {}
    for index in sorted(common):
        parts = [signals_by_horizon[h][index] for h in horizons]
        timestamps = {part["signal_at"] for part in parts}
        if len(timestamps) != 1:
            raise ValueError("Horários de decisão divergentes entre os horizontes")
        hourly_returns = [
            part["predicted_return"] / float(h)
            for h, part in zip(horizons, parts, strict=True)
        ]
        votes = sum(
            part["predicted_return"] > 0 and part["p_up"] >= 0.5
            for part in parts
        )
        output[index] = {
            "predicted_return": sum(hourly_returns) / 3.0 * 6.0,
            "p_up": sum(part["p_up"] for part in parts) / 3.0,
            "signal_at": parts[0]["signal_at"],
            "positive_votes": votes,
            "horizons_hours": list(horizons),
        }
    return output



def _summary_series(
    candles: list[dict], signals: dict, *, start: int, end: int,
    capital: float, fee: float, slippage: float, edge: float,
    min_probability: float, exit_return: float, baseline: bool, label: str,
    bar_seconds: int,
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
                and signal.get("positive_votes", 3) >= 2
            ):
                trade("BUY", candle["open"], candle["timestamp"], signal, "FORECAST_ENTRY")
            elif quantity > 0 and (
                signal["predicted_return"] <= exit_return
                or signal["p_up"] < min_probability
                or signal.get("positive_votes", 3) < 2
            ):
                trade("SELL", candle["open"], candle["timestamp"], signal, "FORECAST_EXIT")
        if quantity > 0:
            exposed += 1
        equity = cash + quantity * candle["close"]
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1)
        points.append({
            "timestamp": (candle["timestamp"] + timedelta(seconds=bar_seconds)).isoformat(),
            "equity": equity, "position": "BTC" if quantity > 0 else "CASH",
        })

    # Closing liquidation is executed at the last known close, not a future candle.
    if quantity > 0:
        candle = candles[end]
        trade("SELL", candle["close"],
              candle["timestamp"] + timedelta(seconds=bar_seconds),
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
    cagr = (cash / capital) ** (365.25 / duration_days) - 1 if duration_days >= 30 else None
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
    optimized_policy: dict | None = None,
) -> dict:
    if optimized_policy is not None:
        validate_policy(optimized_policy)
        if horizon != 0:
            raise ValueError("A política otimizada usa conjuntamente 2h+6h+24h; não pode ser aplicada a horizonte isolado")
    if horizon not in {0, 2, 6, 24}:
        raise ValueError("Horizonte deve ser conjunto 2h+6h+24h ou individual 2h, 6h ou 24h")
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
    if horizon == 0:
        sources, candidates = {}, {}
        for h in (2, 6, 24):
            sources[h] = _signals(original, h, candles, seconds)
            candidates[h] = _signals(replay, h, candles, seconds)
            if sources[h].keys() != candidates[h].keys():
                raise ValueError(f"Sinais OOS originais e replay diferem para {h}h")
        source = _joint_signals(sources)
        candidate = (
            aggregate_signals(candidates, optimized_policy)
            if optimized_policy is not None else _joint_signals(candidates)
        )
        if source.keys() != candidate.keys():
            raise ValueError("O conjunto de decisões OOS não coincide entre modelos")
    else:
        source = _signals(original, horizon, candles, seconds)
        candidate = _signals(replay, horizon, candles, seconds)
        if source.keys() != candidate.keys():
            raise ValueError("Sinais executáveis V1 e V2 em timestamps diferentes")
    start, end = min(source), max(source)
    if start == end:
        raise ValueError("Período OOS insuficiente para calcular capitalização")
    common = dict(start=start, end=end, capital=initial_capital, fee=fee_rate,
                  bar_seconds=seconds,
                  slippage=slippage_rate, edge=entry_edge,
                  min_probability=min_probability, exit_return=exit_return)
    if optimized_policy is not None:
        # Same causal engine used for model/policy selection before OOS.
        # Both fee and slippage come from the same user's replay settings.
        challenger = simulate_rotations(
            candles, candidate, optimized_policy,
            initial_capital=initial_capital, fee=fee_rate,
            slippage=slippage_rate, bar_seconds=seconds,
            label="BTC_CASH_ROTATION",
        )
        buy_hold = _summary_series(
            candles, {}, baseline=True, label="BUY_AND_HOLD", **common
        )
        results = [challenger, buy_hold]
    else:
        # Historical research replays retain their previous behavior.
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
        "horizons_hours": [2, 6, 24] if horizon == 0 else [horizon],
        "policy": (
            "2 of 3 positive directional votes; equal-weight probability; "
            "6h normalized mean hourly return; next-open execution"
            if horizon == 0 else "individual forecast, next-open execution"
        ),
        "period_start": candles[start]["timestamp"].isoformat(),
        "period_end": candles[end]["timestamp"].isoformat(),
        "signal_count": len(source),
        "optimized_policy": validate_policy(optimized_policy) if optimized_policy else None,
        "portfolio_rule": "Single BTC/CASH portfolio with reinvestment, model exit, stop, trailing, take profit and cooldown" if optimized_policy else None,

        "original_archive_sha256": sha256(original).hexdigest(),
        "replay_archive_sha256": sha256(replay).hexdigest(),
        "parameters": {
            "initial_capital": initial_capital, "fee_rate": fee_rate,
            "slippage_rate": slippage_rate, "entry_edge": entry_edge,
            "min_probability": min_probability, "exit_return": exit_return,
        },
        "results": results,
        "note": (
            "Política BTC/CASH otimizada somente antes do OOS, com reinvestimento e custos. Nenhuma ordem real."
            if optimized_policy else
            "Simulação OOS de consenso fixo 2h+6h+24h, sem ajuste no período OOS."
            if horizon == 0 else "Simulação histórica individual OOS, sem negociação real."
        ),
    }
