"""Reprice exactly the SAME PRE-OOS BTC/CASH fills under higher execution costs.

Unlike running the strategy again with higher fees, this engine never applies
entry/exit thresholds, risk stops, or forecasts. Thus extra fees cannot
erase trades and make a risky strategy look harmless by staying in CASH.

Trade times, sides, source prices, and final close liquidation are frozen
from the original causal simulation. Cash/BTC quantities and Alpaca fee tiers
are recomputed independently as trading capital compounds.
"""
from __future__ import annotations

from datetime import timedelta
import math

from .alpaca_fees import ALPACA_FEE_MODEL, AlpacaFeeLedger, round_trip_cost
from .btc_policy import _timestamp, validate_policy


def replay_frozen_rotations(
    candles: list[dict], signals: dict[int, dict], baseline_actions: list[dict],
    policy: dict, *, initial_capital: float = 10000.0,
    fee: float = .001, slippage: float = .0005,
    fee_multiplier: float = 1.25, bar_seconds: int = 1800,
) -> dict:
    """Cost-stress replay. No new trades or future-derived trading decisions."""
    cfg = validate_policy(policy)
    if not (math.isfinite(initial_capital) and initial_capital > 0):
        raise ValueError("Capital inicial inválido")
    if not (math.isfinite(fee_multiplier) and 1 <= fee_multiplier <= 2):
        raise ValueError("Multiplicador de taxas inválido")
    if not (0 <= fee < .05 and 0 <= slippage < .05):
        raise ValueError("Custos de execução inválidos")
    if not signals or len(candles) < 3 or bar_seconds <= 0:
        raise ValueError("Histórico ou sinais insuficientes para replay")
    first, last = min(signals), max(signals)
    if first < 1 or last >= len(candles):
        raise ValueError("Índice de simulação inválido")
    model = cfg["policy.fee_model"]
    ledger = (
        AlpacaFeeLedger(cfg["policy.alpaca_liquidity"],
                        cfg["policy.alpaca_initial_30d_volume_usd"])
        if model == ALPACA_FEE_MODEL else None
    )
    actions_by_index: dict[int, list[dict]] = {}
    for action in baseline_actions:
        i = int(action["candle_index"])
        if not first <= i <= last:
            raise ValueError("Operação fora do intervalo validado")
        if action["action"] not in ("BUY", "SELL"):
            raise ValueError("Ações congeladas precisam ser BUY ou SELL")
        candle = candles[i]
        start = _timestamp(candle["timestamp"])
        at = _timestamp(action["timestamp"])
        final_close = action.get("reason") == "END_OF_TEST"
        if final_close:
            if i != last or action["action"] != "SELL" or at != start + timedelta(seconds=bar_seconds):
                raise ValueError("Liquidação final não corresponde ao último candle")
        elif at != start:
            raise ValueError("Horário de execução do replay não corresponde ao candle")
        px = float(candle["close"] if final_close else candle["open"])
        if not math.isclose(px, float(action["price"]), rel_tol=1e-10):
            raise ValueError("Preço da operação congelada não corresponde ao histórico")
        actions_by_index.setdefault(i, []).append(action)
    cash, btc = float(initial_capital), 0.0
    fees, slips, cycles = 0.0, 0.0, 0
    values: list[float] = []
    performed: list[dict] = []
    drawdown, max_equity = 0.0, float(initial_capital)
    previous = _timestamp(candles[first - 1]["timestamp"])
    for i in range(first, last + 1):
        candle = candles[i]
        when = _timestamp(candle["timestamp"])
        if when <= previous:
            raise ValueError("Candles fora da ordem cronológica")
        previous = when
        opening, closing = float(candle["open"]), float(candle["close"])
        if opening <= 0 or closing <= 0:
            raise ValueError("Preço de BTC inválido")
        for original in actions_by_index.get(i, ()):
            side = original["action"]
            at = _timestamp(original["timestamp"])
            px = float(original["price"])
            quote = ledger.quote(at) if ledger else {
                "rate": fee, "tier": None, "volume_30d_usd": None,
            }
            rate = float(quote["rate"]) * fee_multiplier
            if not 0 <= rate < 1:
                raise ValueError("Taxa de replay inválida")
            if side == "BUY":
                if cash <= 0 or btc != 0:
                    raise ValueError("Duas compras seguidas na agenda congelada")
                notional = cash
                charge = notional * rate
                slip_cost = notional * slippage / (1 + slippage)
                btc = (notional - charge) / (px * (1 + slippage))
                cash = 0.0
            else:
                if btc <= 0 or cash != 0:
                    raise ValueError("Venda sem posição BTC na agenda congelada")
                notional = btc * px * (1 - slippage)
                charge = notional * rate
                slip_cost = btc * px * slippage
                cash = notional - charge
                btc = 0.0
                cycles += 1
            fees += charge
            slips += slip_cost
            performed.append({
                "timestamp": at.isoformat(), "action": side,
                "candle_index": i, "reason": original.get("reason"),
                "source_price": px, "fee_rate": rate,
                "fee_tier": quote["tier"], "fee_usd_equivalent": charge,
                "slippage_usd_estimate": slip_cost,
                "notional_usd": notional,
            })
            if ledger:
                ledger.record(at, notional)
        value = cash + btc * closing
        max_equity = max(value, max_equity)
        drawdown = min(drawdown, value / max_equity - 1)
        values.append(float(value))
    if btc:
        raise ValueError("Agenda congelada não liquidou a posição final")
    returns = [
        values[i] / values[i - 1] - 1
        for i in range(1, len(values)) if values[i - 1] > 0
    ]
    mean = sum(returns) / len(returns) if returns else 0.0
    variance = sum((r - mean) ** 2 for r in returns) / len(returns) if returns else 0.0
    sharpe = (
        mean / math.sqrt(variance) * math.sqrt(365.25 * 86400 / bar_seconds)
        if variance > 1e-18 else 0.0
    )
    blocks = min(3, len(values) // 2)
    segment_returns = [
        values[(j + 1) * (len(values) - 1) // blocks]
        / values[j * (len(values) - 1) // blocks] - 1
        for j in range(blocks)
    ] if blocks else []
    schedule = [
        (a["candle_index"], a["timestamp"], a["action"], a.get("reason"))
        for a in baseline_actions
    ]
    replayed_schedule = [
        (a["candle_index"], a["timestamp"], a["action"], a.get("reason"))
        for a in performed
    ]
    if schedule != replayed_schedule or len(performed) != len(baseline_actions):
        raise ValueError("O teste de estresse alterou a agenda de operações")
    return {
        "capital_final": float(cash),
        "net_profit_usd": float(cash - initial_capital),
        "fees_paid": float(fees),
        "slippage_usd_estimate": float(slips),
        "max_drawdown": float(drawdown),
        "worst_fold_return": float(min(segment_returns, default=0)),
        "segment_returns": segment_returns,
        "sharpe": float(sharpe),
        "completed_cycles": cycles,
        "switch_count": len(performed),
        "frozen_schedule": True,
        "same_trade_schedule": True,
        "stress_method": "REPRICE_SAME_FILLS_WITH_DYNAMIC_ALPACA_TIERS",
        "fee_multiplier": fee_multiplier,
        "initial_capital": float(initial_capital),
        "fee_model": model,
        "actions": performed,
    }
