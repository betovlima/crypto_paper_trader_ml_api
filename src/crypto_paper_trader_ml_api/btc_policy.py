"""Deterministic, causal BTC/CASH capital rotation with full reinvestment.

Both pre-OOS strategy selection and final immutable OOS replay call the very
same engine. Decisions are formed at the previous candle close; fills occur at
the NEXT candle open. Stop/trailing/take-profit checks are close-confirmed,
never assumed to fill at an unknown intrabar stop price. No live orders.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

from .alpaca_fees import ALPACA_FEE_MODEL, AlpacaFeeLedger, round_trip_cost

HORIZONS = (2, 6, 24)
DEFAULT_POLICY = {
    "policy.enabled": True,
    "policy.entry_probability": 0.44,
    "policy.exit_probability": 0.32,
    "policy.entry_edge": 0.002,
    "policy.exit_return": -0.001,
    "policy.min_positive_votes": 2,
    "policy.stop_loss": 0.055,
    "policy.trailing_stop": 0.045,
    "policy.take_profit": 0.08,
    "policy.min_hold_bars": 4,
    "policy.cooldown_bars": 4,
    "policy.weight_2h": 0.4,
    "policy.weight_6h": 0.4,
    "policy.weight_24h": 0.2,
    "policy.fee_model": ALPACA_FEE_MODEL,
    "policy.alpaca_liquidity": "TAKER",
    "policy.alpaca_initial_30d_volume_usd": 0.0,
}


def validate_policy(values: dict) -> dict:
    policy = {**DEFAULT_POLICY, **{k: v for k, v in values.items() if k in DEFAULT_POLICY}}
    bounds = {
        "policy.entry_probability": (0.25, 0.85),
        "policy.exit_probability": (0.10, 0.65),
        "policy.entry_edge": (0.0, 0.06),
        "policy.exit_return": (-0.06, 0.02),
        "policy.min_positive_votes": (1, 3),
        "policy.stop_loss": (0.005, 0.35),
        "policy.trailing_stop": (0.005, 0.35),
        "policy.take_profit": (0.005, 0.5),
        "policy.min_hold_bars": (0, 96),
        "policy.cooldown_bars": (0, 96),
        "policy.weight_2h": (0.01, 1.0),
        "policy.weight_6h": (0.01, 1.0),
        "policy.weight_24h": (0.01, 1.0),
    }
    for key, (minimum, maximum) in bounds.items():
        number = float(policy[key])
        if not math.isfinite(number) or not minimum <= number <= maximum:
            raise ValueError(f"Parâmetro de decisão BTC/CASH inválido: {key}")
        if key in {"policy.min_positive_votes", "policy.min_hold_bars", "policy.cooldown_bars"}:
            if not number.is_integer():
                raise ValueError(f"Parâmetro inteiro inválido: {key}")
            policy[key] = int(number)
        else:
            policy[key] = number
    if policy["policy.exit_probability"] >= policy["policy.entry_probability"]:
        raise ValueError("A saída precisa exigir probabilidade menor que a entrada (histerese)")
    if policy["policy.fee_model"] not in {"FIXED", ALPACA_FEE_MODEL}:
        raise ValueError("Modelo de taxas inválido")
    if policy["policy.alpaca_liquidity"] not in {"MAKER", "TAKER"}:
        raise ValueError("Execução Alpaca deve ser MAKER ou TAKER")
    initial_volume = float(policy["policy.alpaca_initial_30d_volume_usd"])
    if not math.isfinite(initial_volume) or initial_volume < 0:
        raise ValueError("Volume inicial de 30 dias inválido")
    policy["policy.alpaca_initial_30d_volume_usd"] = initial_volume
    policy["policy.enabled"] = True
    return policy


def aggregate_signals(by_horizon: dict[int, dict], policy: dict) -> dict:
    """Weighted multi-horizon evidence at the exact same executable bar."""
    selected = validate_policy(policy)
    if any(not by_horizon.get(h) for h in HORIZONS):
        raise ValueError("Ausência de sinais em um dos horizontes 2h, 6h ou 24h")
    indices = set.intersection(*(set(by_horizon[h]) for h in HORIZONS))
    weights = [
        selected[f"policy.weight_{h}h"] for h in HORIZONS
    ]
    normalizer = sum(weights)
    joined: dict[int, dict] = {}
    for index in sorted(indices):
        entries = [by_horizon[h][index] for h in HORIZONS]
        if len({entry.get("signal_at") for entry in entries}) != 1:
            raise ValueError("Sinais de horizontes diferentes não coincidem no tempo")
        joint_p = sum(weight * item["p_up"] for weight, item in zip(weights, entries, strict=True)) / normalizer
        # Map returns to equivalent 6-hour scale before averaging, otherwise
        # the 24h forecast dominates a 2h decision by construction.
        joint_return = sum(
            weight * item["predicted_return"] * (6.0 / h)
            for weight, h, item in zip(weights, HORIZONS, entries, strict=True)
        ) / normalizer
        # A 50% gate previously blocked all votes when calibrated base
        # rates were below 50%, regardless of positive predicted return.
        # Votes use return signs; probabilities have a separate tuned gate.
        votes = sum(item["predicted_return"] > 0 for item in entries)
        probability_votes = sum(item["p_up"] >= .5 for item in entries)
        joined[index] = {
            "predicted_return": float(joint_return),
            "p_up": float(joint_p),
            "positive_votes": int(votes),
            "probability_votes": int(probability_votes),
            "signal_at": entries[0]["signal_at"],
        }
    return joined


def _timestamp(value) -> datetime:
    if isinstance(value, datetime):
        instant = value
    elif hasattr(value, "to_pydatetime"):
        instant = value.to_pydatetime()
    else:
        instant = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if instant.tzinfo is None:
        raise ValueError("Candles precisam ter timestamps com fuso horário")
    return instant.astimezone(timezone.utc)


def simulate_rotations(
    candles: list[dict], signals: dict[int, dict], policy: dict, *,
    initial_capital: float = 10000.0, fee: float = 0.001,
    slippage: float = 0.0005, bar_seconds: int = 1800,
    label: str = "LIGHTGBM_ROTATION",
    audit_limit: int = 200,
    timeline_limit: int | None = 500,
    fee_model: str | None = None,
    alpaca_liquidity: str | None = None,
    alpaca_initial_30d_volume_usd: float | None = None,
) -> dict:
    """Signal at prior close -> fill at next open, at most one trade per candle."""
    cfg = validate_policy(policy)
    fee_model = fee_model or cfg["policy.fee_model"]
    alpaca_liquidity = alpaca_liquidity or cfg["policy.alpaca_liquidity"]
    start_volume = (
        cfg["policy.alpaca_initial_30d_volume_usd"]
        if alpaca_initial_30d_volume_usd is None
        else alpaca_initial_30d_volume_usd
    )
    if fee_model not in {"FIXED", ALPACA_FEE_MODEL}:
        raise ValueError("Modelo de custos BTC/CASH inválido")
    ledger = AlpacaFeeLedger(alpaca_liquidity, start_volume) if fee_model == ALPACA_FEE_MODEL else None
    if not (initial_capital > 0 and math.isfinite(initial_capital)):
        raise ValueError("Capital inicial inválido")
    if not (0 <= fee <= .05 and 0 <= slippage <= .05):
        raise ValueError("Taxas de execução inválidas")
    if len(candles) < 3 or bar_seconds <= 0:
        raise ValueError("Histórico insuficiente para capitalização")
    first_time = _timestamp(candles[0]["timestamp"])
    previous = first_time
    for candle in candles:
        when = _timestamp(candle["timestamp"])
        if when < previous or (when == previous and when != first_time):
            raise ValueError("Candles fora da ordem temporal")
        previous = when
    cash, btc = float(initial_capital), 0.0
    first, last = min(signals), max(signals)
    if first < 1 or last >= len(candles):
        raise ValueError("Índice de decisão executável inválido")
    # The first signal is known only at its preceding bar's close.
    entry_price = 0.0
    peak_close = 0.0
    position_start = -1
    last_sale_index = -100000
    peak_equity, drawdown = cash, 0.0
    exposure, cycles, wins, losses, fees_paid = 0, 0, 0, 0, 0.0
    gross_wins, gross_losses = 0.0, 0.0
    actions, points, equity_values = [], [], []
    initial_fee = ledger.quote(_timestamp(candles[first]["timestamp"]))["rate"] if ledger else fee
    rt_cost = round_trip_cost(initial_fee, initial_fee, slippage)
    gross_profit_before_costs = 0.0
    slippage_paid = 0.0
    buy_reference_price = 0.0
    last_exec_index = -1
    cycle_cost = 0.0

    def perform(side: str, price: float, at: datetime, index: int, reason: str, signal: dict | None):
        nonlocal cash, btc, entry_price, peak_close, position_start, last_sale_index
        nonlocal cycles, wins, losses, gross_wins, gross_losses, fees_paid, cycle_cost
        nonlocal gross_profit_before_costs, slippage_paid, buy_reference_price
        quote = ledger.quote(at) if ledger else {
            "rate": fee, "tier": None, "volume_30d_usd": None,
            "liquidity": "FIXED", "tier_cutoff_at": None,
        }
        applied_fee = quote["rate"]
        if side == "BUY":
            gross = cash
            cost = gross * applied_fee
            btc = (gross - cost) / (price * (1 + slippage))
            fees_paid += cost
            slippage_cost = gross * slippage / (1 + slippage)
            slippage_paid += slippage_cost
            buy_reference_price = price
            filled_notional = gross
            fee_asset = "BTC"
            fee_amount_asset = gross / (price * (1 + slippage)) * applied_fee
            cash = 0.0
            entry_price, peak_close, position_start = price, price, index
            cycle_cost = gross
        else:
            gross = btc * price * (1 - slippage)
            cost = gross * applied_fee
            cash = gross - cost
            fees_paid += cost
            slippage_cost = btc * price * slippage
            slippage_paid += slippage_cost
            gross_profit_before_costs += cycle_cost * (price / buy_reference_price - 1)
            filled_notional = gross
            fee_asset = "USD"
            fee_amount_asset = cost
            btc = 0.0
            last_sale_index = index
            cycles += 1
            gain = cash - cycle_cost
            if gain > 0:
                wins += 1
                gross_wins += gain
            else:
                losses += 1
                gross_losses += -gain
            entry_price = peak_close = 0
            position_start = -1
        actions.append({
            "timestamp": at.isoformat(), "action": side, "reason": reason,
            "fee_model": fee_model, "fee_rate": applied_fee,
            "fee_tier": quote["tier"], "liquidity": quote["liquidity"],
            "fee_30d_volume_usd": quote["volume_30d_usd"],
            "fee_tier_cutoff_at": quote["tier_cutoff_at"],
            "fee_asset": fee_asset, "fee_asset_amount": fee_amount_asset,
            "fee_usd_equivalent": cost, "slippage_usd_estimate": slippage_cost,
            "executed_notional_usd": filled_notional,
            "price": price, "capital_after": cash + btc * price,
            "predicted_return": signal["predicted_return"] if signal else None,
            "probability_up": signal["p_up"] if signal else None,
            "positive_votes": signal.get("positive_votes") if signal else None,
            "signal_at": signal["signal_at"] if signal else None,
        })
        if ledger is not None:
            ledger.record(at, filled_notional)

    for index in range(first, last + 1):
        candle = candles[index]
        current_time = _timestamp(candle["timestamp"])
        price_open, price_close = float(candle["open"]), float(candle["close"])
        if price_open <= 0 or price_close <= 0:
            raise ValueError("Preço de BTC inválido")
        signal = signals.get(index)
        # If a candle is missing, do not execute a stale forecast or risk stop
        # at an unknown intrabar price. Position is still marked to market.
        consecutive = (
            index > 0
            and current_time - _timestamp(candles[index - 1]["timestamp"])
            == timedelta(seconds=bar_seconds)
        )
        if consecutive and last_exec_index != index:
            p = float(signal["p_up"]) if signal else None
            estimate = float(signal["predicted_return"]) if signal else None
            votes = int(signal.get("positive_votes", 3)) if signal else None
            if btc == 0:
                current_fee = ledger.quote(current_time)["rate"] if ledger else fee
                current_round_trip = round_trip_cost(current_fee, current_fee, slippage)
                if (
                    signal is not None
                    and index - last_sale_index > cfg["policy.cooldown_bars"]
                    and votes >= cfg["policy.min_positive_votes"]
                    and p >= cfg["policy.entry_probability"]
                    and estimate > current_round_trip + cfg["policy.entry_edge"]
                ):
                    perform("BUY", price_open, current_time, index, "MODEL_ENTRY", signal)
                    last_exec_index = index
            else:
                # Close-confirmed risk exits must work EVEN IF no new model
                # prediction is available. Fills are still next-bar OPEN.
                held = index - position_start
                close_prev = float(candles[index - 1]["close"])
                stop_loss = close_prev <= entry_price * (1 - cfg["policy.stop_loss"])
                trailing = close_prev <= peak_close * (1 - cfg["policy.trailing_stop"])
                take_profit = close_prev >= entry_price * (1 + cfg["policy.take_profit"])
                model_exit = (
                    signal is not None and (
                        p <= cfg["policy.exit_probability"]
                        or estimate <= cfg["policy.exit_return"]
                        or votes < cfg["policy.min_positive_votes"]
                    )
                )
                reason = (
                    "STOP_LOSS" if stop_loss else
                    "TRAILING_STOP" if trailing else
                    "TAKE_PROFIT" if take_profit else
                    "MODEL_EXIT" if model_exit and held >= cfg["policy.min_hold_bars"] else None
                )
                if reason:
                    perform("SELL", price_open, current_time, index, reason, signal)
                    last_exec_index = index
        if btc:
            exposure += 1
            peak_close = max(peak_close, price_close)
        value = cash + btc * price_close
        peak_equity = max(peak_equity, value)
        drawdown = min(drawdown, value / peak_equity - 1)
        timestamp = (current_time + timedelta(seconds=bar_seconds)).isoformat()
        points.append({"timestamp": timestamp, "equity": float(value), "position": "BTC" if btc else "CASH"})
        equity_values.append(float(value))

    # Liquidate at last *known* candle close; never use a future price.
    if btc:
        final_bar = candles[last]
        when = _timestamp(final_bar["timestamp"]) + timedelta(seconds=bar_seconds)
        perform("SELL", float(final_bar["close"]), when, last, "END_OF_TEST", None)
        points[-1]["equity"] = cash
        points[-1]["position"] = "CASH"
        equity_values[-1] = cash
    peak_equity = max(peak_equity, cash)
    drawdown = min(drawdown, cash / peak_equity - 1)
    values = equity_values
    returns = [values[i] / values[i-1] - 1 for i in range(1, len(values)) if values[i-1] > 0]
    mean = sum(returns) / len(returns) if returns else 0.0
    variance = sum((r - mean)**2 for r in returns) / len(returns) if returns else 0.0
    sharpe = mean / math.sqrt(variance) * math.sqrt(365.25 * 86400 / bar_seconds) if variance > 1e-18 else 0.0
    group_count = min(3, len(values) // 2)
    fold_returns = []
    for block in range(group_count):
        left = block * (len(values) - 1) // group_count
        right = (block + 1) * (len(values) - 1) // group_count
        fold_returns.append(values[right] / values[left] - 1 if values[left] > 0 else -1)
    duration = (
        _timestamp(candles[last]["timestamp"]) - _timestamp(candles[first]["timestamp"])
    ).total_seconds() / 86400
    stride = max(1, math.ceil(len(points) / timeline_limit)) if timeline_limit else 1
    sampled = [p for index, p in enumerate(points) if index % stride == 0 or index == len(points) - 1]
    return {
        "code": label,
        "initial_capital": float(initial_capital),
        "capital_final": float(cash),
        "total_return": float(cash / initial_capital - 1),
        "return_total": float(cash / initial_capital - 1),
        "cagr": (cash / initial_capital)**(365.25 / duration) - 1 if duration >= 30 else None,
        "max_drawdown": float(drawdown),
        "sharpe": float(sharpe),
        "worst_fold_return": float(min(fold_returns) if fold_returns else 0),
        "exposure": exposure / len(points),
        "switch_count": len(actions),
        "completed_cycles": cycles, "buys": cycles,
        "sells": cycles, "wins": wins, "losses": losses,
        "win_rate": wins / cycles if cycles else 0.0,
        "profit_factor": gross_wins / gross_losses if gross_losses > 0 else None,
        "fees_paid": float(fees_paid),
        "alpaca_fees_usd": float(fees_paid) if ledger else None,
        "slippage_usd_estimate": float(slippage_paid),
        "net_profit_usd": float(cash - initial_capital),
        "gross_profit_before_costs_usd": float(gross_profit_before_costs),
        "fee_model": fee_model,
        "alpaca_liquidity": alpaca_liquidity if ledger else None,
        "alpaca_initial_30d_volume_usd": start_volume if ledger else None,
        "fee_tier_history": sorted({(a["fee_tier"], a["fee_rate"]) for a in actions if a["fee_tier"] is not None}),
        "fee_rate_used_first": initial_fee,
        "timeline": sampled,
        "actions": actions[-audit_limit:],
        "action_count": len(actions),
        "actions_truncated": len(actions) > audit_limit,
        "required_round_trip_return": float(rt_cost),
        "policy": cfg,
        "execution_rule": "Prior candle close -> next candle open; close-confirmed stops; full reinvestment",
    }
