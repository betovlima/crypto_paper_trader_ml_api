"""V14 capital-first BTC trend, breakout and volatility-regime research.

Only complete UTC 4h, 1d and weekly bars are eligible. The original 30-minute
archive is the sole price source. All decisions at completed higher-timeframe
closes fill at the NEXT 30-minute OPEN, never at a retrospective extreme.
No live orders, model training, future peeking, or intra-candle stop fills.
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import timedelta
from hashlib import sha256
from io import BytesIO
from math import ceil, isfinite, log, sqrt
from zipfile import ZipFile
import json

import numpy as np
import pandas as pd

from .alpaca_fees import ALPACA_FEE_MODEL, AlpacaFeeLedger
from .btc_event_research import _build_data, _portfolio as benchmark_portfolio
from .mexc import TIMEFRAME_SECONDS

INITIAL_CAPITAL = 10000.0
POLICIES = (
    ("DAILY_SMA50_BAND", "Tendência diária SMA50 com banda de 1,2%"),
    ("DAILY_SMA20_100", "Tendência diária SMA20/100"),
    ("FOUR_HOUR_BREAKOUT", "Rompimento Donchian 4h + tendência diária + ATR"),
    ("DUAL_4H_DAILY", "Tendência 4h com confirmação diária"),
    ("DAILY_VOL_REGIME", "Regimes diários com exposição 0%, 50% ou 100%"),
    ("WEEKLY_TREND_FILTER", "Tendência diária com filtro semanal"),
)
POLICY_NAMES = {code for code, _ in POLICIES}
APPROVAL_RULES = {
    "development": "retorno líquido positivo e pelo menos uma compra e uma venda",
    "validation": (
        "retorno líquido positivo; capital final >= 101% do Buy & Hold; "
        "drawdown máximo não pior que Buy & Hold; compras e vendas registradas"
    ),
    "selection": "uma única política selecionada pelo desenvolvimento, antes da validação",
}
TARGET_ANNUAL_VOL = .30
REBALANCE_MIN_CHANGE = .40
FOUR_HOUR_SECONDS = 4 * 3600
DAY_SECONDS = 86400
WEEK_SECONDS = 7 * DAY_SECONDS


def completed_bars(frame: pd.DataFrame, *, source_seconds: int,
                   bar_seconds: int, frequency: str) -> pd.DataFrame:
    """Ignore partial/missing higher-timeframe bars, including cut-off tails."""
    if bar_seconds % source_seconds:
        raise ValueError("Granularidade fonte incompatível")
    expected = bar_seconds // source_seconds
    source = frame[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    source["end"] = pd.to_datetime(source["timestamp"], utc=True) + pd.Timedelta(seconds=source_seconds)
    records = []
    for end, group in source.groupby(pd.Grouper(key="end", freq=frequency,
                                                closed="right", label="right", origin="epoch")):
        if len(group) != expected:
            continue
        ordered = group.sort_values("timestamp")
        times = ordered["end"]
        if (times.iloc[-1] != end or
                times.iloc[0] != end - pd.Timedelta(seconds=bar_seconds-source_seconds) or
                not times.diff().iloc[1:].eq(pd.Timedelta(seconds=source_seconds)).all()):
            continue
        records.append({
            "end": pd.Timestamp(end), "open": float(ordered.iloc[0]["open"]),
            "high": float(ordered["high"].max()), "low": float(ordered["low"].min()),
            "close": float(ordered.iloc[-1]["close"]),
            "volume": float(ordered["volume"].sum()),
        })
    if not records:
        raise ValueError("Não há candles completos no intervalo solicitado")
    result = pd.DataFrame(records).sort_values("end").reset_index(drop=True)
    consecutive = []
    run = 0
    previous = None
    for stamp in result["end"]:
        run = (run + 1) if previous is not None and (stamp-previous).total_seconds() == bar_seconds else 1
        consecutive.append(run)
        previous = stamp
    result["streak"] = consecutive
    close = result["close"]
    result["sma20"] = close.rolling(20, min_periods=20).mean()
    result["sma50"] = close.rolling(50, min_periods=50).mean()
    result["sma100"] = close.rolling(100, min_periods=100).mean()
    result["ema12"] = close.ewm(span=12, adjust=False).mean()
    result["ema26"] = close.ewm(span=26, adjust=False).mean()
    result["ema36"] = close.ewm(span=36, adjust=False).mean()
    result["upper20"] = result["high"].shift(1).rolling(20, min_periods=20).max()
    result["lower10"] = result["low"].shift(1).rolling(10, min_periods=10).min()
    last = close.shift(1)
    ranges = pd.concat([
        result["high"] - result["low"],
        (result["high"] - last).abs(), (result["low"] - last).abs()
    ], axis=1).max(axis=1)
    result["atr14"] = ranges.rolling(14, min_periods=14).mean()
    returns = np.log(close / last)
    returns = returns.mask(result["end"].diff().dt.total_seconds() != bar_seconds)
    result["annual_vol30"] = returns.rolling(30, min_periods=30).std(ddof=0) * sqrt(365.25 * DAY_SECONDS / bar_seconds)
    return result


def _positive(*values) -> bool:
    return all(isfinite(float(x)) and float(x) > 0 for x in values)


def _latest_before_or_at(bars: pd.DataFrame, ends: list, at: pd.Timestamp,
                         max_age_seconds: int):
    idx = bisect_right(ends, at) - 1
    if idx < 0 or (at - ends[idx]).total_seconds() > max_age_seconds:
        return None
    return bars.iloc[idx]


def _target(policy: str, frame: str, row, daily, weekly, current: float,
            peak: float) -> tuple[float, str]:
    """Return target fraction and signal reason; retain current exposure in neutral bands."""
    close = float(row["close"])
    if policy == "DAILY_SMA50_BAND" and frame == "1d":
        if row["streak"] < 55 or not _positive(row["sma50"]):
            return current, "WARMUP"
        if close > row["sma50"] * 1.012:
            return 1.0, "SMA50_UP"
        if close < row["sma50"] * .988:
            return 0.0, "SMA50_DOWN"
        return current, "HYSTERESIS"
    if policy == "DAILY_SMA20_100" and frame == "1d":
        if row["streak"] < 105 or not _positive(row["sma20"], row["sma100"]):
            return current, "WARMUP"
        if row["sma20"] > row["sma100"] * 1.01 and close > row["sma20"]:
            return 1.0, "SMA20_ABOVE_100"
        if row["sma20"] < row["sma100"] * .99 or close < row["sma100"] * .99:
            return 0.0, "SMA20_BELOW_100"
        return current, "HYSTERESIS"
    if policy == "FOUR_HOUR_BREAKOUT" and frame == "4h":
        if (row["streak"] < 32 or daily is None or daily["streak"] < 55 or
                not _positive(row["upper20"], row["lower10"], row["atr14"], daily["sma50"])):
            return current, "WARMUP"
        if current > 0 and (close < row["lower10"] or close < peak - 3 * row["atr14"]):
            return 0.0, "DONCHIAN_OR_ATR_EXIT"
        if current == 0 and close > row["upper20"] + .35 * row["atr14"] and daily["close"] > daily["sma50"] * 1.005:
            return 1.0, "DONCHIAN_BREAKOUT"
        return current, "HOLD"
    if policy == "DUAL_4H_DAILY" and frame == "4h":
        if row["streak"] < 42 or daily is None or daily["streak"] < 55 or not _positive(daily["sma50"]):
            return current, "WARMUP"
        if row["ema12"] > row["ema36"] * 1.005 and daily["close"] > daily["sma50"] * 1.01:
            return 1.0, "DUAL_TIMEFRAME_UP"
        if row["ema12"] < row["ema36"] * .995 or daily["close"] < daily["sma50"] * .99:
            return 0.0, "DUAL_TIMEFRAME_DOWN"
        return current, "HYSTERESIS"
    if policy == "DAILY_VOL_REGIME" and frame == "1d":
        if row["streak"] < 105 or not _positive(row["sma100"], row["sma20"], row["annual_vol30"]):
            return current, "WARMUP"
        bullish = close > row["sma100"] * 1.015 and row["sma20"] > row["sma100"]
        bearish = close < row["sma100"] * .985 or row["sma20"] < row["sma100"] * .99
        if bearish:
            return 0.0, "BEARISH_CASH"
        if bullish:
            risk_weight = min(1.0, TARGET_ANNUAL_VOL / float(row["annual_vol30"]))
            return (1.0 if risk_weight >= .75 else .5), "VOLATILITY_WEIGHT"
        return current, "HYSTERESIS"
    if policy == "WEEKLY_TREND_FILTER" and frame == "1d":
        if (row["streak"] < 55 or weekly is None or weekly["streak"] < 29 or
                not _positive(row["sma50"], weekly["ema12"], weekly["ema26"])):
            return current, "WARMUP"
        if close > row["sma50"] * 1.01 and weekly["ema12"] > weekly["ema26"] * 1.005:
            return 1.0, "DAILY_AND_WEEKLY_UP"
        if close < row["sma50"] * .99 or weekly["ema12"] < weekly["ema26"] * .995:
            return 0.0, "DAILY_OR_WEEKLY_DOWN"
        return current, "HYSTERESIS"
    return current, "NOT_EVALUATED"


def build_decision_map(frame: pd.DataFrame, source_seconds: int) -> dict:
    """Only complete higher bars. Daily/weekly filters are joined backward in time."""
    if DAY_SECONDS % source_seconds or FOUR_HOUR_SECONDS % source_seconds:
        raise ValueError("Dados históricos incompatíveis com 4h/1d")
    aggregates = {
        "4h": completed_bars(frame, source_seconds=source_seconds,
                             bar_seconds=FOUR_HOUR_SECONDS, frequency="4h"),
        "1d": completed_bars(frame, source_seconds=source_seconds,
                             bar_seconds=DAY_SECONDS, frequency="1D"),
        "1w": completed_bars(frame, source_seconds=source_seconds,
                             bar_seconds=WEEK_SECONDS, frequency="7D"),
    }
    daily, weekly = aggregates["1d"], aggregates["1w"]
    day_ends = list(daily["end"])
    week_ends = list(weekly["end"])
    decisions = {}
    valid_execution_times = set(frame["timestamp"])
    for period in ("4h", "1d"):
        for row in aggregates[period].to_dict("records"):
            at = pd.Timestamp(row["end"])
            if at not in valid_execution_times:
                continue  # the next 30m open is unavailable, never backfill
            d = _latest_before_or_at(daily, day_ends, at, DAY_SECONDS)
            w = _latest_before_or_at(weekly, week_ends, at, WEEK_SECONDS)
            decisions.setdefault(at, {})[period] = (row, d, w)
    return decisions


def simulate_policy(frame: pd.DataFrame, lo: int, hi: int, policy: str, *,
                    source_seconds: int, slippage: float, decision_map: dict,
                    include_actions: bool = False) -> dict:
    if policy not in POLICY_NAMES or not (0 <= lo < hi <= len(frame)):
        raise ValueError("Política ou intervalo BTC inválido")
    if not 0 <= slippage < .05:
        raise ValueError("Slippage inválido")
    ledger = AlpacaFeeLedger("TAKER")
    cash, units = INITIAL_CAPITAL, 0.0
    fees, slip_paid = 0.0, 0.0
    buy_count, sell_count, complete_exits = 0, 0, 0
    exposure_sum, peak_equity, max_dd = 0.0, INITIAL_CAPITAL, 0.0
    last_target, peak_close = 0.0, 0.0
    last_trade_index, last_exit_index = -100000, -100000
    actions, equity = [], []
    previous_at = None
    min_hold_bars = 48 if policy not in ("WEEKLY_TREND_FILTER",) else 7 * 48

    def trade(target: float, price: float, at: pd.Timestamp, index: int,
              reason: str, signal_at: str | None) -> None:
        nonlocal cash, units, fees, slip_paid, buy_count, sell_count
        nonlocal last_target, peak_close, last_trade_index, last_exit_index, complete_exits
        current_value = cash + units * price
        current_btc = units * price
        new_btc = max(0, min(1, target)) * current_value
        delta = new_btc - current_btc
        if abs(delta) <= max(1e-8, current_value * .0001):
            last_target = target
            return
        quote = ledger.quote(at.to_pydatetime())
        fee_rate = float(quote["rate"])
        if delta > 0:
            notional = min(cash, delta)
            commission = notional * fee_rate
            slip_amount = notional * slippage / (1 + slippage)
            units += (notional - commission) / (price * (1 + slippage))
            cash -= notional
            buy_count += 1
            side = "BUY"
        else:
            quantity = min(units, -delta / price)
            notional = quantity * price * (1 - slippage)
            commission = notional * fee_rate
            slip_amount = quantity * price * slippage
            units -= quantity
            cash += notional - commission
            sell_count += 1
            side = "SELL"
            if target == 0.0:
                units = max(0.0, units)  # rounding tolerance, normally zero
                complete_exits += 1
                last_exit_index = index
        ledger.record(at.to_pydatetime(), notional)
        fees += commission
        slip_paid += slip_amount
        last_trade_index = index
        last_target = target
        if side == "BUY":
            peak_close = max(peak_close, price)
        elif target == 0:
            peak_close = 0.0
        if include_actions:
            actions.append({
                "policy": policy, "action": side,
                "timestamp": at.isoformat(), "signal_at": signal_at,
                "execution_candle_index": index, "execution_price": price,
                "target_fraction": target, "reason": reason,
                "fee_rate": fee_rate, "fee_tier": quote["tier"],
                "fee_usd": float(commission),
                "slippage_usd": float(slip_amount),
                "capital_after": float(cash + units * price),
            })

    for i in range(lo, hi):
        at = frame.at[i, "timestamp"]
        opening, close = float(frame.at[i, "open"]), float(frame.at[i, "close"])
        gap = previous_at is not None and (at - previous_at).total_seconds() != source_seconds
        previous_at = at
        if gap and units > 0:
            trade(0.0, opening, at, i, "DATA_GAP_EXIT", None)
        available = decision_map.get(at, {})
        desired = last_target
        reason = None
        decision_at = None
        if not gap:
            for period in ("4h", "1d"):
                if period not in available:
                    continue
                row, daily, weekly = available[period]
                proposed, cause = _target(policy, period, row, daily, weekly, desired, peak_close)
                if abs(proposed - desired) >= REBAlANCE_MIN_CHANGE:
                    desired, reason, decision_at = proposed, cause, at.isoformat()
            if reason is not None:
                if (desired > last_target and i - last_exit_index < 48):
                    pass  # minimum 24h after full exit
                elif (desired > 0 and last_target > 0 and abs(desired-last_target) < REBAlANCE_MIN_CHANGE):
                    pass
                elif desired < last_target and reason == "HYSTERESIS":
                    pass
                elif desired < last_target and (
                    reason not in ("DONCHIAN_OR_ATR_EXIT", "BEARISH_CASH") and
                    i - last_trade_index < min_hold_bars
                ):
                    pass
                else:
                    trade(desired, opening, at, i, reason, decision_at)
        if units > 0:
            peak_close = max(peak_close, close)
        value = cash + units * close
        exposure_sum += units * close / value if value > 0 else 0
        peak_equity = max(peak_equity, value)
        max_dd = min(max_dd, value / peak_equity - 1)
        equity.append(value)

    if units > 0:
        end = frame.iloc[hi - 1]
        stamp = end["timestamp"] + timedelta(seconds=source_seconds)
        trade(0, float(end["close"]), stamp, hi - 1, "END_OF_WINDOW_CLOSE", None)
        equity[-1] = cash
        max_dd = min(max_dd, cash / peak_equity - 1)

    values = np.asarray(equity, dtype=float)
    returns = values[1:] / values[:-1] - 1 if len(values) > 1 else np.array([])
    sigma = float(np.std(returns)) if len(returns) else 0
    sharpe = float(np.mean(returns) / sigma * sqrt(365.25 * DAY_SECONDS / source_seconds)) if sigma > 1e-12 else 0.0
    segments = [float(values[b] / values[a] - 1)
                for a, b in [((j * (len(values)-1))//3, ((j+1)*(len(values)-1))//3)
                             for j in range(3)] if values[a] > 0]
    result = {
        "initial_capital": INITIAL_CAPITAL, "capital_final": float(cash),
        "net_profit_usd": float(cash - INITIAL_CAPITAL),
        "return_total": float(cash / INITIAL_CAPITAL - 1),
        "max_drawdown": float(max_dd), "sharpe": float(sharpe),
        "segment_returns": segments, "buy_count": int(buy_count),
        "sell_count": int(sell_count), "completed_cycles": int(complete_exits),
        "operation_count": buy_count + sell_count,
        "exposure": float(exposure_sum/(hi-lo)), "fees_paid": float(fees),
        "slippage_usd_estimate": float(slip_paid),
        "start": frame.at[lo, "timestamp"].isoformat(),
        "end": frame.at[hi-1, "timestamp"].isoformat(),
        "fee_model": ALPACA_FEE_MODEL,
        "execution_rule": "Sinal somente no fechamento completo UTC 4h/1d/1w; preenchimento no próximo OPEN 30m; saída no fechamento final",
    }
    if include_actions:
        result["actions"] = actions
        result["equity_curve"] = equity
    return result


def evaluate_archived_trend_regimes(source_archive: bytes, *,
                                    include_details: bool = False) -> dict:
    from .archive_replay import inspect_source_archive

    request, _ = inspect_source_archive(source_archive)
    if request.symbol not in ("BTCUSDT", "BTCUSD"):
        raise ValueError("Pesquisa de tendência requer BTC")
    source_seconds = TIMEFRAME_SECONDS.get(request.decision_timeframe)
    if source_seconds != 1800:
        raise ValueError("Pesquisa V14 requer dados históricos BTC de 30 minutos")
    with ZipFile(BytesIO(source_archive)) as z:
        folds = json.loads(z.read("folds.json"))
        raw = pd.read_csv(BytesIO(z.read("market_history_decision.csv")),
                          parse_dates=["timestamp"], float_precision="round_trip")
    boundaries = [pd.Timestamp(f["test_start"]) for f in folds
                  if int(f["horizon_hours"]) in (2,6,24)]
    if not boundaries:
        raise ValueError("Arquivo histórico sem limite temporal OOS")
    first_oos = min(boundaries)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    historic = raw.loc[raw["timestamp"] < first_oos].reset_index(drop=True)
    if len(historic) < 1600:
        raise ValueError("Histórico anterior ao OOS insuficiente")
    frame = _build_data(historic, bar_seconds=source_seconds)
    decision_map = build_decision_map(frame, source_seconds)
    split = int(len(frame) * .60)
    embargo = max(48, ceil(DAY_SECONDS / source_seconds))
    periods = {
        "development": (max(0, int(len(frame) * .05)), split-embargo),
        "validation": (split, len(frame)),
    }
    if periods["development"][1]-periods["development"][0] < 500 or periods["validation"][1]-periods["validation"][0] < 300:
        raise ValueError("Janelas de desenvolvimento/validação insuficientes")
    baseline = {}
    for code in ("CASH", "BUY_HOLD"):
        baseline[code] = {}
        for part, (lo, hi) in periods.items():
            r = benchmark_portfolio(frame, lo, hi, code,
                                    fee_rate=request.fee_rate,
                                    slippage=request.slippage_rate,
                                    bar_seconds=source_seconds)
            r.pop("actions", None)
            baseline[code][part] = r

    results, all_actions = [], []
    for idx, (policy, label) in enumerate(POLICIES, 1):
        part_metrics = {}
        for part, (lo, hi) in periods.items():
            result = simulate_policy(
                frame, lo, hi, policy, source_seconds=source_seconds,
                slippage=float(request.slippage_rate),
                decision_map=decision_map, include_actions=include_details,
            )
            if include_details:
                all_actions.extend({"partition": part, **a}
                                   for a in result.pop("actions"))
                result.pop("equity_curve")
            part_metrics[part] = result

        dev, val = part_metrics["development"], part_metrics["validation"]
        checks = {
            "development_positive": dev["net_profit_usd"] > 0,
            "development_operated": dev["buy_count"] >= 1 and dev["sell_count"] >= 1,
            "validation_positive": val["net_profit_usd"] > 0,
            "validation_beats_buy_hold": val["capital_final"] >= 1.01 * baseline["BUY_HOLD"]["validation"]["capital_final"],
            "validation_drawdown_controlled": val["max_drawdown"] >= baseline["BUY_HOLD"]["validation"]["max_drawdown"],
            "validation_operated": val["buy_count"] >= 1 and val["sell_count"] >= 1,
        }
        results.append({
            "trial": idx, "status": "RESEARCHED",
            "parameters": {"policy": policy, "description": label},
            "metrics": {
                **val, "development": dev, "quality_checks": checks,
                "passed_internal_checks": all(checks.values()),
                "score": float(log(max(1e-12, val["capital_final"]/INITIAL_CAPITAL)) + .5 * val["max_drawdown"]),
            },
        })
    eligible = [r for r in results if r["metrics"]["quality_checks"]["development_positive"]
                and r["metrics"]["quality_checks"]["development_operated"]]
    chosen = max(eligible, key=lambda r: (
        log(max(1e-12, r["metrics"]["development"]["capital_final"]/INITIAL_CAPITAL))
        + .5*r["metrics"]["development"]["max_drawdown"]
    ), default=None)
    approved = bool(chosen and chosen["metrics"]["passed_internal_checks"])
    approved_policy = chosen["parameters"]["policy"] if approved else None
    trace_start = max(periods["validation"][0], len(frame)-ceil(35*DAY_SECONDS/source_seconds))
    stride = max(1, ceil((len(frame)-trace_start)/220))
    prices = [{"timestamp": frame.at[i, "timestamp"].isoformat(),
               "close": float(frame.at[i, "close"])}
              for i in range(trace_start, len(frame), stride)]
    if prices and prices[-1]["timestamp"] != frame.at[len(frame)-1, "timestamp"].isoformat():
        prices.append({"timestamp": frame.at[len(frame)-1, "timestamp"].isoformat(),
                       "close": float(frame.at[len(frame)-1, "close"])})
    chart_actions = []
    if approved_policy:
        replay = simulate_policy(frame, *periods["validation"], approved_policy,
                                 source_seconds=source_seconds,
                                 slippage=float(request.slippage_rate),
                                 decision_map=decision_map, include_actions=True)
        chart_actions = [a for a in replay["actions"] if a["execution_candle_index"] >= trace_start]
    summary = {
        "experiment": "BTC_TREND_REGIMES_V14",
        "result_status": "PRE_OOS_APPROVED_NEEDS_FRESH_DATA" if approved else "NO_STRATEGY_APPROVED",
        "approval_rules": APPROVAL_RULES,
        "candidate_count": len(POLICIES),
        "rejected_count": len(POLICIES)-(1 if approved else 0),
        "approved_policies": [approved_policy] if approved else [],
        "chosen_only_on_development": chosen["parameters"]["policy"] if chosen else None,
        "chosen_validation": chosen["metrics"] if chosen else None,
        "chosen_rejection_reasons": ([
            key for key, passed in chosen["metrics"]["quality_checks"].items() if not passed
        ] if chosen and not approved else []),
        "approved_validation": chosen["metrics"] if approved else None,
        "validation_kind": "PRE_OOS_CHRONOLOGICAL_HOLDOUT_NO_MODEL_TRAINING",
        "first_oos_untouched_from": first_oos.isoformat(),
        "development_period": [frame.at[periods["development"][0], "timestamp"].isoformat(),
                               frame.at[periods["development"][1]-1, "timestamp"].isoformat()],
        "validation_period": [frame.at[split, "timestamp"].isoformat(),
                              frame.at[len(frame)-1, "timestamp"].isoformat()],
        "embargo_bars": embargo, "source_symbol": request.symbol,
        "source_exchange": request.history_source,
        "source_sha256": sha256(source_archive).hexdigest(),
        "decision_timeframe": request.decision_timeframe,
        "research_timeframes": ["4h", "1d", "1w"],
        "fee_model": ALPACA_FEE_MODEL, "slippage_rate": float(request.slippage_rate),
        "initial_capital_usd_equivalent": INITIAL_CAPITAL,
        "buy_hold_baseline": baseline["BUY_HOLD"], "cash_baseline": baseline["CASH"],
        "chart": {"prices": prices, "actions": chart_actions},
        "oos_executed": False, "live_orders_enabled": False,
        "research_warning": (
            "Pesquisa PRE-OOS: nem a seleção nem a validação constituem teste independente inédito. "
            "Candles 4h/diários/semanais só entram após fechamento completo. "
            "Ordens e stops executados na abertura seguinte de 30m, ou fechamento final; "
            "não existe preenchimento intrabar garantido. BTCUSDT é proxy para Alpaca BTCUSD. "
            "Carteiras e fees usam preços históricos e slippage simulado."
        ),
    }
    return {"trials": results, "summary": summary, "trades": all_actions}
