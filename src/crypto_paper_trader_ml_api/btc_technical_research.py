"""Causal, capital-first BTC technical analysis comparison.

Every entry and discretionary stop uses information through a completed CLOSE.
Fills are at the NEXT uninterrupted candle OPEN. No intrabar/limit fill claims,
future-price pivot trades, model training, or live brokerage calls.
"""
from __future__ import annotations

from datetime import timedelta
from io import BytesIO
import json
from math import ceil, isfinite, log, sqrt
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .alpaca_fees import ALPACA_FEE_MODEL, AlpacaFeeLedger
from .btc_directional_change import find_confirmed_extrema
from .btc_event_research import _build_data, _portfolio as baseline_portfolio
from .mexc import TIMEFRAME_SECONDS

CAPITAL = 10000.0
POLICIES = (
    ("DC_2PCT", "Directional Change, confirmação de 2%"),
    ("EMA_9_SLOPE", "Inclinação da EMA9"),
    ("EMA_9_21_CROSS", "Cruzamento EMA9/EMA21"),
    ("EMA_STOCH_14_3_3", "EMA9/21 com estocástico 14,3,3"),
    ("DONCHIAN_ATR", "Donchian 20 com proteção de 2 ATR"),
    ("EMA_TRAILING_3PCT", "EMA9/21 com trailing de 3%"),
)
MIN_READY_BARS = 35


def add_indicators(frame: pd.DataFrame, seconds: int) -> pd.DataFrame:
    """Indicators use only candles available at or before their own timestamp."""
    df = frame.copy()
    close, high, low = df["close"], df["high"], df["low"]
    df["ema9"] = close.ewm(span=9, adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()
    lo14 = low.rolling(14, min_periods=14).min()
    hi14 = high.rolling(14, min_periods=14).max()
    raw_k = (100 * (close - lo14) / (hi14 - lo14).replace(0, np.nan)).clip(0, 100)
    df["stoch_k"] = raw_k.rolling(3, min_periods=3).mean()
    df["stoch_d"] = df["stoch_k"].rolling(3, min_periods=3).mean()
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14, min_periods=14).mean()
    df["upper20"] = high.shift(1).rolling(20, min_periods=20).max()
    df["lower20"] = low.shift(1).rolling(20, min_periods=20).min()
    contiguous = []
    count = 0
    for delta in df["timestamp"].diff().dt.total_seconds():
        count = count + 1 if count and delta == seconds else 1
        contiguous.append(count)
    df["continuous_bars"] = contiguous
    return df.reset_index(drop=True)


def _signal(df: pd.DataFrame, i: int, policy: str, holding: bool,
            peak_close: float) -> str | None:
    if i < 1 or int(df.at[i, "continuous_bars"]) < MIN_READY_BARS:
        return None
    cur, prev = df.iloc[i], df.iloc[i - 1]
    close = float(cur["close"])
    ema9, ema21 = float(cur["ema9"]), float(cur["ema21"])
    if policy == "EMA_9_SLOPE":
        up = ema9 > float(prev["ema9"]) and close > ema9
        down = ema9 < float(prev["ema9"]) and close < ema9
        return ("SELL" if down else None) if holding else ("BUY" if up else None)
    if policy in ("EMA_9_21_CROSS", "EMA_TRAILING_3PCT"):
        bull = ema9 > ema21
        bear = ema9 < ema21
        if policy == "EMA_TRAILING_3PCT":
            if holding:
                if close <= peak_close * .97:
                    return "SELL"
                return "SELL" if bear else None
            return "BUY" if bull and close > ema9 else None
        crossed_up = bull and float(prev["ema9"]) <= float(prev["ema21"])
        crossed_down = bear and float(prev["ema9"]) >= float(prev["ema21"])
        return ("SELL" if crossed_down else None) if holding else ("BUY" if crossed_up else None)
    if policy == "EMA_STOCH_14_3_3":
        k, d, prev_k, prev_d = (
            float(cur["stoch_k"]), float(cur["stoch_d"]),
            float(prev["stoch_k"]), float(prev["stoch_d"]),
        )
        if not all(isfinite(x) for x in (k, d, prev_k, prev_d)):
            return None
        if holding:
            return "SELL" if close < ema21 or (prev_k >= prev_d and k < d and k > 60) else None
        return "BUY" if ema9 > ema21 and prev_k <= prev_d and k > d and k < 65 else None
    if policy == "DONCHIAN_ATR":
        if holding:
            return "SELL" if (close < float(cur["lower20"]) or
                close < peak_close - 2 * float(cur["atr14"])) else None
        return "BUY" if close > float(cur["upper20"]) else None
    raise ValueError("Política técnica desconhecida: " + policy)


def simulate_policy(df: pd.DataFrame, lo: int, hi: int, policy: str, *,
                    seconds: int, slippage: float, include_actions: bool = False) -> dict:
    """Identical causal fee/next-open execution model for all candidate policies."""
    if not 0 <= slippage < .05 or not 0 <= lo < hi <= len(df):
        raise ValueError("Intervalo ou custo inválido")
    if policy not in {name for name, _ in POLICIES}:
        raise ValueError("Política técnica não reconhecida")
    dc_events = {}
    if policy == "DC_2PCT":
        dc_events = {e["confirmed_index"]: e for e in
                     find_confirmed_extrema(df, lo, hi, .02, seconds)}
    ledger = AlpacaFeeLedger("TAKER")
    cash, btc = CAPITAL, 0.0
    fees, slippage_paid = 0.0, 0.0
    cost_basis, peak_close = CAPITAL, 0.0
    peak_equity, dd = CAPITAL, 0.0
    cycles, wins, invested_bars = 0, 0, 0
    pending = None
    actions, curve = [], []
    last_at = None

    def execute(side: str, price: float, at, index: int, reason: str, signal_at=None):
        nonlocal cash, btc, cost_basis, fees, slippage_paid, cycles, wins, peak_close
        quoted = ledger.quote(at)
        rate = float(quoted["rate"])
        if side == "BUY":
            if btc or cash <= 0:
                return
            notional = cash
            commission = notional * rate
            slip = notional * slippage / (1 + slippage)
            cost_basis = notional
            btc = (notional - commission) / (price * (1 + slippage))
            cash = 0.0
            pnl = None
            peak_close = price
        else:
            if btc <= 0:
                return
            notional = btc * price * (1 - slippage)
            commission = notional * rate
            slip = btc * price * slippage
            cash = notional - commission
            btc = 0.0
            pnl = cash - cost_basis
            cycles += 1
            wins += pnl > 0
            peak_close = 0.0
        fees += commission
        slippage_paid += slip
        ledger.record(at, notional)
        if include_actions:
            actions.append({
                "policy": policy, "action": side, "timestamp": at.isoformat(),
                "signal_at": signal_at, "execution_candle_index": index,
                "execution_price": float(price), "reason": reason,
                "fee_rate": rate, "fee_tier": quoted["tier"], "fee_usd": float(commission),
                "slippage_usd": float(slip), "capital_after": float(cash + btc * price),
                "net_profit_usd": float(pnl) if pnl is not None else None,
            })

    for i in range(lo, hi):
        current = df.iloc[i]
        at = current["timestamp"]
        opening, close = float(current["open"]), float(current["close"])
        gap = last_at is not None and (at - last_at).total_seconds() != seconds
        last_at = at
        if gap:
            pending = None
            if btc:
                execute("SELL", opening, at.to_pydatetime(), i, "DATA_GAP_EXIT")
        elif pending is not None:
            side, reason, when = pending
            execute(side, opening, at.to_pydatetime(), i, reason, when)
            pending = None

        if btc:
            invested_bars += 1
            peak_close = max(peak_close, close)
        value = cash + btc * close
        peak_equity = max(peak_equity, value)
        dd = min(dd, value / peak_equity - 1)
        curve.append(value)
        if i + 1 >= hi or gap:
            continue
        if (df.at[i + 1, "timestamp"] - at).total_seconds() != seconds:
            continue
        if policy == "DC_2PCT":
            event = dc_events.get(i)
            side = ("BUY" if event["kind"] == "BOTTOM_CONFIRMED" else "SELL") if event else None
            reason = "CONFIRMED_BOTTOM" if side == "BUY" else "CONFIRMED_TOP"
        else:
            side = _signal(df, i, policy, btc > 0, peak_close)
            reason = "CLOSE_BASED_TECHNICAL_SIGNAL"
            if policy == "EMA_TRAILING_3PCT" and side == "SELL" and close <= peak_close * .97:
                reason = "TRAILING_3PCT_CLOSE"
            if policy == "DONCHIAN_ATR" and side == "SELL" and close < peak_close - 2 * float(df.at[i, "atr14"]):
                reason = "ATR_2X_CLOSE"
        if (side == "BUY" and btc == 0) or (side == "SELL" and btc > 0):
            pending = (side, reason, at.isoformat())

    if btc:
        last = df.iloc[hi - 1]
        execute("SELL", float(last["close"]),
                (last["timestamp"] + timedelta(seconds=seconds)).to_pydatetime(),
                hi - 1, "END_OF_WINDOW_CLOSE")
        curve[-1] = cash
        dd = min(dd, cash / peak_equity - 1)

    values = np.asarray(curve, dtype=float)
    returns = values[1:] / values[:-1] - 1 if len(values) > 1 else np.array([])
    sigma = float(np.std(returns)) if len(returns) else 0.0
    sharpe = (float(np.mean(returns) / sigma * sqrt(365.25 * 86400 / seconds))
              if sigma > 1e-12 else 0.0)
    segments = [float(values[b] / values[a] - 1)
                for a, b in [((x * (len(values)-1)) // 3,
                             ((x+1) * (len(values)-1)) // 3) for x in range(3)]
                if values[a] > 0]
    result = {
        "initial_capital": CAPITAL, "capital_final": float(cash),
        "net_profit_usd": float(cash - CAPITAL), "return_total": float(cash / CAPITAL - 1),
        "max_drawdown": float(dd), "sharpe": sharpe, "segment_returns": segments,
        "completed_cycles": cycles, "wins": int(wins), "losses": int(cycles - wins),
        "win_rate": wins / cycles if cycles else 0.0, "fees_paid": float(fees),
        "slippage_usd_estimate": float(slippage_paid),
        "exposure": float(invested_bars / (hi - lo)),
        "start": df.at[lo, "timestamp"].isoformat(),
        "end": df.at[hi - 1, "timestamp"].isoformat(),
        "fee_model": ALPACA_FEE_MODEL,
        "execution_rule": "CLOSE confirmado, ordem no próximo OPEN; stops por fechamento, sem intrabar",
    }
    if include_actions:
        result["actions"] = actions
        result["equity_curve"] = curve
    return result


def evaluate_archived_technical_research(source_archive: bytes, *,
                                         include_details: bool = False) -> dict:
    """Reuses archived historical source and V12 temporal embargo; OOS is not opened."""
    from .archive_replay import inspect_source_archive
    request, _ = inspect_source_archive(source_archive)
    if request.symbol not in ("BTCUSD", "BTCUSDT"):
        raise ValueError("A análise técnica requer BTC")
    seconds = TIMEFRAME_SECONDS.get(request.decision_timeframe)
    if seconds is None or seconds > 1800:
        raise ValueError("Use candles de 30 minutos ou menos")
    with ZipFile(BytesIO(source_archive)) as z:
        folds = json.loads(z.read("folds.json"))
        historical = pd.read_csv(BytesIO(z.read("market_history_decision.csv")),
                                 parse_dates=["timestamp"], float_precision="round_trip")
    first_oos = min(pd.Timestamp(f["test_start"]) for f in folds
                    if int(f["horizon_hours"]) in (2, 6, 24))
    historical["timestamp"] = pd.to_datetime(historical["timestamp"], utc=True)
    historical = historical.loc[historical["timestamp"] < first_oos].reset_index(drop=True)
    if len(historical) < 1600:
        raise ValueError("Histórico pré-OOS insuficiente")
    df = add_indicators(_build_data(historical, bar_seconds=seconds), seconds)
    split = int(len(df) * .60)
    embargo = max(48, ceil(24 * 3600 / seconds))
    partitions = {
        "development": (max(0, int(len(df) * .05)), split - embargo),
        "validation": (split, len(df)),
    }
    if partitions["development"][1] - partitions["development"][0] < 500 or len(df) - split < 300:
        raise ValueError("Janelas de treino e validação insuficientes")
    trials, all_trades, curves = [], [], {}
    for index, (policy, description) in enumerate(POLICIES, 1):
        metrics = {}
        for part, (lo, hi) in partitions.items():
            r = simulate_policy(df, lo, hi, policy, seconds=seconds,
                                slippage=float(request.slippage_rate), include_actions=True)
            curves[(policy, part)] = r.pop("equity_curve")
            actions = r.pop("actions")
            if include_details:
                all_trades.extend({"partition": part, **a} for a in actions)
            metrics[part] = r
        dev, val = metrics["development"], metrics["validation"]
        checks = {
            "development_cycles": dev["completed_cycles"] >= 3,
            "development_positive_net": dev["net_profit_usd"] > 0,
            "validation_positive_net": val["net_profit_usd"] > 0,
            "validation_positive_segments": sum(x > 0 for x in val["segment_returns"]) >= 2,
        }
        score = float(log(max(1e-12, val["capital_final"] / CAPITAL)) + .5 * val["max_drawdown"])
        trials.append({
            "trial": index, "status": "RESEARCHED",
            "parameters": {"policy": policy, "description": description},
            "metrics": {**val, "development": dev, "score": score,
                        "quality_checks": checks, "passed_internal_checks": all(checks.values())},
        })
    reference = {}
    for name in ("CASH", "BUY_HOLD"):
        reference[name] = {}
        for partition, (lo, hi) in partitions.items():
            r = baseline_portfolio(df, lo, hi, name, fee_rate=request.fee_rate,
                                   slippage=request.slippage_rate, bar_seconds=seconds)
            r.pop("actions", None)
            reference[name][partition] = r
    eligible = [t for t in trials if t["metrics"]["development"]["completed_cycles"] >= 3]
    chosen = max(eligible, key=lambda t: (
        log(max(1e-12, t["metrics"]["development"]["capital_final"] / CAPITAL))
        + .5 * t["metrics"]["development"]["max_drawdown"]), default=None)
    chosen_name = chosen["parameters"]["policy"] if chosen else None
    recent_start = max(partitions["validation"][0], len(df) - ceil(35 * 86400 / seconds))
    stride = max(1, ceil((len(df) - recent_start) / 220))
    chart_prices = [{"timestamp": df.at[i, "timestamp"].isoformat(),
                     "close": float(df.at[i, "close"])}
                    for i in range(recent_start, len(df), stride)]
    if not chart_prices or chart_prices[-1]["timestamp"] != df.at[len(df)-1, "timestamp"].isoformat():
        chart_prices.append({"timestamp": df.at[len(df)-1, "timestamp"].isoformat(),
                             "close": float(df.at[len(df)-1, "close"])})
    chart_actions = []
    if chosen_name:
        replay = simulate_policy(df, *partitions["validation"], chosen_name, seconds=seconds,
                                 slippage=float(request.slippage_rate), include_actions=True)
        chart_actions = [
            a for a in replay["actions"]
            if a["execution_candle_index"] >= recent_start
        ]
    summary = {
        "experiment": "BTC_TECHNICAL_CAPITAL_V13",
        "result_status": ("PRE_OOS_PROMISING_NEEDS_NEW_DATA"
                          if chosen and chosen["metrics"]["passed_internal_checks"]
                          else "NO_RELIABLE_EDGE_IDENTIFIED"),
        "validation_kind": "PRE_OOS_CHRONOLOGICAL_HOLDOUT",
        "first_oos_untouched_from": first_oos.isoformat(),
        "development_period": [df.at[partitions["development"][0], "timestamp"].isoformat(),
                               df.at[partitions["development"][1]-1, "timestamp"].isoformat()],
        "validation_period": [df.at[split, "timestamp"].isoformat(),
                              df.at[len(df)-1, "timestamp"].isoformat()],
        "embargo_bars": embargo,
        "source_symbol": request.symbol, "source_exchange": request.history_source,
        "decision_timeframe": request.decision_timeframe,
        "initial_capital_usd_equivalent": CAPITAL,
        "fee_model": ALPACA_FEE_MODEL,
        "slippage_rate": float(request.slippage_rate),
        "chosen_only_on_development": chosen_name,
        "chosen_validation": chosen["metrics"] if chosen else None,
        "cash_baseline": reference["CASH"], "buy_hold_baseline": reference["BUY_HOLD"],
        "chart": {"prices": chart_prices, "actions": chart_actions},
        "oos_executed": False, "live_orders_enabled": False,
        "research_warning": (
            "Resultado pré-OOS, sem prova independente futura. Dados BTCUSDT são proxy "
            "para BTCUSD Alpaca. Stops são avaliados no fechamento e executados no "
            "próximo open; não são ordens stop intrabar. Preços, spreads e fills reais "
            "podem diferir. O OOS antigo já foi examinado em outras pesquisas."
        ),
    }
    return {"trials": trials, "summary": summary, "trades": all_trades}
