"""BTC-specific retrospective extrema and causal Directional Change trading study.

Historical extrema are labeled at their actual observed CLOSE, but can be
recognized ONLY after a reversal threshold is confirmed at a later CLOSE.
Trading decisions use the confirmation, with fill at the NEXT candle OPEN.

Four fixed hypotheses (1%, 2%, 3%, volatility-scaled) are evaluated in
chronologically distinct PRE-OOS development and validation partitions.
No model training, hindsight entry, intrabar fill, OOS access, or brokerage
orders. Historical absolute highs/lows of OHLC are descriptive ONLY.
"""
from __future__ import annotations

from datetime import timedelta
from io import BytesIO
import json
from math import ceil, isfinite, sqrt
from statistics import median
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .alpaca_fees import ALPACA_FEE_MODEL, AlpacaFeeLedger
from .btc_event_research import _build_data, _portfolio as reference_portfolio
from .mexc import TIMEFRAME_SECONDS

CAPITAL = 10000.0
THRESHOLDS = (
    ("DC_FIXED_1PCT", .01, "Confirmação de 1%"),
    ("DC_FIXED_2PCT", .02, "Confirmação de 2%"),
    ("DC_FIXED_3PCT", .03, "Confirmação de 3%"),
    ("DC_VOL_24H", None, "Confirmação adaptativa à volatilidade de 24h"),
)
ADAPTIVE_VOL_MULTIPLIER = 1.2
ADAPTIVE_MIN = .01
ADAPTIVE_MAX = .04


def _threshold(df: pd.DataFrame, index: int, fixed: float | None, bar_seconds: int) -> float:
    if fixed is not None:
        return float(fixed)
    sigma = float(df.at[index, "sigma"])
    if not isfinite(sigma):
        return ADAPTIVE_MIN
    daily_bars = 86400 / bar_seconds
    return float(np.clip(ADAPTIVE_VOL_MULTIPLIER * sigma * sqrt(daily_bars),
                         ADAPTIVE_MIN, ADAPTIVE_MAX))


def find_confirmed_extrema(
    df: pd.DataFrame, lo: int, hi: int, fixed: float | None, bar_seconds: int,
) -> list[dict]:
    """Causal streaming detector; a pivot is never a tradable signal on its own.

    Start searching for a BOTTOM from CASH. The threshold is frozen at each
    new pivot using volatility observed at that pivot's confirmed close.
    Discontinuities reset the pivot state rather than fabricating crossings.
    """
    if not (0 <= lo < hi <= len(df)):
        raise ValueError("Intervalo de candles BTC inválido")
    if fixed is not None and not .005 <= fixed <= .1:
        raise ValueError("Percentual de reversão inválido")
    mode = "BOTTOM"
    pivot_idx = lo
    pivot = float(df.at[lo, "close"])
    threshold = _threshold(df, lo, fixed, bar_seconds)
    events: list[dict] = []
    for i in range(lo + 1, hi):
        at, prior = df.at[i, "timestamp"], df.at[i - 1, "timestamp"]
        price = float(df.at[i, "close"])
        if (at - prior).total_seconds() != bar_seconds:
            mode, pivot_idx, pivot = "BOTTOM", i, price
            threshold = _threshold(df, i, fixed, bar_seconds)
            continue
        if mode == "BOTTOM":
            if price < pivot:
                pivot_idx, pivot = i, price
                threshold = _threshold(df, i, fixed, bar_seconds)
            elif price >= pivot * (1 + threshold):
                events.append({
                    "kind": "BOTTOM_CONFIRMED",
                    "pivot_index": int(pivot_idx),
                    "pivot_timestamp": df.at[pivot_idx, "timestamp"].isoformat(),
                    "pivot_close": float(pivot),
                    "confirmed_index": int(i),
                    "confirmed_at": at.isoformat(),
                    "confirmed_close": float(price),
                    "threshold_fraction": threshold,
                    "lag_candles": int(i - pivot_idx),
                    "confirmation_move_fraction": float(price / pivot - 1),
                })
                mode, pivot_idx, pivot = "TOP", i, price
                threshold = _threshold(df, i, fixed, bar_seconds)
        else:
            if price > pivot:
                pivot_idx, pivot = i, price
                threshold = _threshold(df, i, fixed, bar_seconds)
            elif price <= pivot * (1 - threshold):
                events.append({
                    "kind": "TOP_CONFIRMED",
                    "pivot_index": int(pivot_idx),
                    "pivot_timestamp": df.at[pivot_idx, "timestamp"].isoformat(),
                    "pivot_close": float(pivot),
                    "confirmed_index": int(i),
                    "confirmed_at": at.isoformat(),
                    "confirmed_close": float(price),
                    "threshold_fraction": threshold,
                    "lag_candles": int(i - pivot_idx),
                    "confirmation_move_fraction": float(price / pivot - 1),
                })
                mode, pivot_idx, pivot = "BOTTOM", i, price
                threshold = _threshold(df, i, fixed, bar_seconds)
    return events


def capitalize_confirmations(
    df: pd.DataFrame, lo: int, hi: int, events: list[dict], *,
    bar_seconds: int, slippage: float, include_actions: bool = False,
) -> dict:
    """Trade BTC/CASH only on NEXT OPEN, after the extrema are confirmed.

    Uses full account reinvestment. Fees are Alpaca dynamic taker tiers;
    source BTCUSDT is treated as a USD-equivalent proxy, not real Alpaca fills.
    """
    if not 0 <= slippage < .05:
        raise ValueError("Slippage inválido")
    ledger = AlpacaFeeLedger("TAKER")
    cash, btc = CAPITAL, 0.0
    fees, slips = 0.0, 0.0
    peak, max_dd = CAPITAL, 0.0
    in_btc, cycles = 0, 0
    winners, losers = 0, 0
    cost_basis = CAPITAL
    actions: list[dict] = []
    equity: list[float] = []
    scheduled = {int(x["confirmed_index"]) + 1: x for x in events
                 if int(x["confirmed_index"]) + 1 < hi}
    last_stamp = df.at[lo, "timestamp"]

    def fill(side: str, price: float, when, signal: dict | None, idx: int, reason: str) -> None:
        nonlocal cash, btc, fees, slips, cycles, winners, losers, cost_basis
        quote = ledger.quote(when)
        rate = float(quote["rate"])
        if side == "BUY":
            if btc > 0 or cash <= 0:
                raise ValueError("Tentativa de compra com posição BTC já aberta")
            notional = cash
            commission = notional * rate
            slip_usd = notional * slippage / (1 + slippage)
            btc = (notional - commission) / (price * (1 + slippage))
            cash = 0.0
            cost_basis = notional
            cycle_net = None
        else:
            if btc <= 0:
                raise ValueError("Venda sem BTC")
            notional = btc * price * (1 - slippage)
            commission = notional * rate
            slip_usd = btc * price * slippage
            cash = notional - commission
            btc = 0.0
            cycle_net = float(cash - cost_basis)
            cycles += 1
            if cycle_net > 0:
                winners += 1
            else:
                losers += 1
        fees += commission
        slips += slip_usd
        ledger.record(when, notional)
        if include_actions:
            actions.append({
                "action": side,
                "timestamp": when.isoformat(),
                "execution_candle_index": int(idx),
                "execution_price": float(price),
                "reason": reason,
                "signal_type": signal["kind"] if signal else "END_OF_WINDOW",
                "pivot_timestamp": signal["pivot_timestamp"] if signal else None,
                "pivot_close": signal["pivot_close"] if signal else None,
                "confirmed_at": signal["confirmed_at"] if signal else None,
                "confirmed_close": signal["confirmed_close"] if signal else None,
                "confirmation_lag_candles": signal["lag_candles"] if signal else None,
                "threshold_fraction": signal["threshold_fraction"] if signal else None,
                "fee_rate": rate,
                "fee_tier": quote["tier"],
                "fee_usd": float(commission),
                "slippage_usd": float(slip_usd),
                "notional_usd": float(notional),
                "capital_after": float(cash + btc * price),
                "cycle_net_profit_usd": cycle_net,
            })

    for i in range(lo, hi):
        stamp = df.at[i, "timestamp"]
        opening = float(df.at[i, "open"])
        closing = float(df.at[i, "close"])
        gap = (i > lo and (stamp - last_stamp).total_seconds() != bar_seconds)
        last_stamp = stamp
        signal = scheduled.get(i) if not gap else None
        if gap and btc > 0:
            fill("SELL", opening, stamp.to_pydatetime(), None, i, "DATA_GAP_EXIT")
        elif signal is not None:
            if signal["kind"] == "BOTTOM_CONFIRMED" and btc == 0:
                fill("BUY", opening, stamp.to_pydatetime(), signal, i, "CONFIRMED_BOTTOM_NEXT_OPEN")
            elif signal["kind"] == "TOP_CONFIRMED" and btc > 0:
                fill("SELL", opening, stamp.to_pydatetime(), signal, i, "CONFIRMED_TOP_NEXT_OPEN")
        if btc > 0:
            in_btc += 1
        value = cash + btc * closing
        peak = max(peak, value)
        max_dd = min(max_dd, value / peak - 1)
        equity.append(float(value))

    if btc > 0:
        final = df.iloc[hi - 1]
        fill("SELL", float(final["close"]),
             final["timestamp"].to_pydatetime() + timedelta(seconds=bar_seconds),
             None, hi - 1, "END_OF_WINDOW_CLOSE_LIQUIDATION")
        equity[-1] = float(cash)
        max_dd = min(max_dd, cash / peak - 1)
    values = np.asarray(equity, dtype=float)
    returns = np.diff(values) / values[:-1] if len(values) > 1 else np.array([])
    std = float(np.std(returns)) if len(returns) else 0.0
    sharpe = (float(np.mean(returns) / std * sqrt(365.25 * 86400 / bar_seconds))
              if std > 1e-12 else 0.0)
    splits = [((j * (len(values) - 1)) // 3,
               ((j + 1) * (len(values) - 1)) // 3) for j in range(3)]
    segment_returns = [
        float(values[r] / values[l] - 1) for l, r in splits if values[l] > 0
    ]
    lags = [int(x["lag_candles"]) for x in events]
    bottom_events = [x for x in events if x["kind"] == "BOTTOM_CONFIRMED"]
    top_events = [x for x in events if x["kind"] == "TOP_CONFIRMED"]
    result = {
        "initial_capital": CAPITAL,
        "capital_final": float(cash),
        "net_profit_usd": float(cash - CAPITAL),
        "return_total": float(cash / CAPITAL - 1),
        "max_drawdown": float(max_dd),
        "sharpe": float(sharpe),
        "segment_returns": segment_returns,
        "completed_cycles": int(cycles),
        "wins": int(winners),
        "losses": int(losers),
        "win_rate": winners / cycles if cycles else 0.0,
        "fees_paid": float(fees),
        "slippage_usd_estimate": float(slips),
        "exposure": float(in_btc / (hi - lo)),
        "confirmed_bottoms": len(bottom_events),
        "confirmed_tops": len(top_events),
        "median_confirmation_lag_candles": float(median(lags)) if lags else None,
        "last_confirmations": events[-12:],
        "start": df.at[lo, "timestamp"].isoformat(),
        "end": df.at[hi - 1, "timestamp"].isoformat(),
        "execution_rule": "Pivot CLOSE retrospectivo; confirmação no fechamento; ordens no próximo OPEN",
        "fee_model": ALPACA_FEE_MODEL,
    }
    if include_actions:
        result["actions"] = actions
    return result


def _absolute_ohlc_extrema(df: pd.DataFrame, lo: int, hi: int) -> dict:
    section = df.iloc[lo:hi]
    low_i = int(section["low"].idxmin())
    high_i = int(section["high"].idxmax())
    return {
        "minimum": {"timestamp": df.at[low_i, "timestamp"].isoformat(),
                    "price": float(df.at[low_i, "low"])},
        "maximum": {"timestamp": df.at[high_i, "timestamp"].isoformat(),
                    "price": float(df.at[high_i, "high"])},
        "definition": "Extremos absolutos OHLC retrospectivos, não sinais executáveis.",
    }


def evaluate_archived_directional_change(
    source_archive: bytes, *, include_details: bool = False,
) -> dict:
    """Only PRE-OOS Bitcoin candles. Recomputable in the existing ZIP exporter."""
    from .archive_replay import inspect_source_archive

    request, _ = inspect_source_archive(source_archive)
    if request.symbol not in ("BTCUSDT", "BTCUSD"):
        raise ValueError("Directional Change requer histórico BTC")
    seconds = TIMEFRAME_SECONDS.get(request.decision_timeframe)
    if seconds is None or seconds > 1800:
        raise ValueError("Pesquisa requer candles BTC de 30 minutos ou menores")
    with ZipFile(BytesIO(source_archive)) as z:
        folds = json.loads(z.read("folds.json"))
        history = pd.read_csv(
            BytesIO(z.read("market_history_decision.csv")),
            parse_dates=["timestamp"], float_precision="round_trip",
        )
    boundary_candidates = [
        pd.Timestamp(fold["test_start"])
        for fold in folds if int(fold["horizon_hours"]) in (2, 6, 24)
    ]
    if not boundary_candidates:
        raise ValueError("Não foi encontrado o início imutável do primeiro OOS")
    first_oos = min(boundary_candidates)
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
    historic = history.loc[history["timestamp"] < first_oos].reset_index(drop=True)
    if len(historic) < 1600:
        raise ValueError("Histórico anterior ao OOS insuficiente")
    df = _build_data(historic, bar_seconds=seconds)
    split = int(len(df) * .60)
    embargo = max(48, ceil(24 * 3600 / seconds))
    development = (max(0, int(len(df) * .05)), split - embargo)
    validation = (split, len(df))
    if development[1] - development[0] < 500 or validation[1] - validation[0] < 300:
        raise ValueError("Janelas de BTC insuficientes após embargo")
    trials: list[dict] = []
    full_extrema: list[dict] = []
    full_trades: list[dict] = []
    for n, (name, fixed, description) in enumerate(THRESHOLDS, 1):
        periods = {}
        for part, (lo, hi) in (("development", development), ("validation", validation)):
            extrema = find_confirmed_extrema(df, lo, hi, fixed, seconds)
            finance = capitalize_confirmations(
                df, lo, hi, extrema, bar_seconds=seconds,
                slippage=float(request.slippage_rate), include_actions=include_details,
            )
            if include_details:
                full_extrema.extend({"policy": name, "partition": part, **e} for e in extrema)
                full_trades.extend({"policy": name, "partition": part, **a}
                                   for a in finance.pop("actions"))
            periods[part] = finance
        check = {
            "development_profit_after_fees": periods["development"]["net_profit_usd"] > 0,
            "development_minimum_6_cycles": periods["development"]["completed_cycles"] >= 6,
            "validation_profit_after_fees": periods["validation"]["net_profit_usd"] > 0,
            "validation_two_positive_subperiods": (
                sum(r > 0 for r in periods["validation"]["segment_returns"]) >= 2
            ),
        }
        trials.append({
            "trial": n,
            "status": "RESEARCHED",
            "parameters": {
                "policy": name, "description": description,
                "threshold_fraction": fixed,
                "adaptive_multiplier": ADAPTIVE_VOL_MULTIPLIER if fixed is None else None,
                "adaptive_min": ADAPTIVE_MIN if fixed is None else None,
                "adaptive_max": ADAPTIVE_MAX if fixed is None else None,
            },
            "metrics": {
                **periods["validation"],
                "development": periods["development"],
                "score": float(np.log(max(1e-12, periods["validation"]["capital_final"] / CAPITAL))
                               + .5 * periods["validation"]["max_drawdown"]),
                "quality_checks": check,
                "passed_internal_checks": all(check.values()),
            },
        })
    baselines = {
        name: {
            label: reference_portfolio(
                df, *period, name, fee_rate=request.fee_rate,
                slippage=request.slippage_rate, bar_seconds=seconds,
            )
            for label, period in (("development", development), ("validation", validation))
        }
        for name in ("CASH", "BUY_HOLD")
    }
    for item in baselines.values():
        for period in item.values():
            period.pop("actions", None)
    good_dev = [
        row for row in trials
        if row["metrics"]["quality_checks"]["development_profit_after_fees"]
        and row["metrics"]["quality_checks"]["development_minimum_6_cycles"]
    ]
    selected = max(
        good_dev, key=lambda r: (
            np.log(max(1e-12, r["metrics"]["development"]["capital_final"] / CAPITAL))
            + .5 * r["metrics"]["development"]["max_drawdown"]
        ),
        default=None,
    )
    period_gap_count = int(
        (df["timestamp"].diff().dt.total_seconds().iloc[1:] != seconds).sum()
    )
    summary = {
        "experiment": "BTC_DIRECTIONAL_CHANGE_EXTREMA_V12",
        "result_status": (
            "PRE_OOS_PROMISING_NEEDS_NEW_DATA"
            if selected and selected["metrics"]["passed_internal_checks"]
            else "NO_RELIABLE_EDGE_IDENTIFIED"
        ),
        "validation_kind": "PRE_OOS_CHRONOLOGICAL_HOLDOUT_NO_MODEL_TRAINING",
        "first_oos_untouched_from": first_oos.isoformat(),
        "development_period": [
            df.at[development[0], "timestamp"].isoformat(),
            df.at[development[1] - 1, "timestamp"].isoformat(),
        ],
        "validation_period": [
            df.at[validation[0], "timestamp"].isoformat(),
            df.at[validation[1] - 1, "timestamp"].isoformat(),
        ],
        "embargo_bars": embargo,
        "source_symbol": request.symbol,
        "source_exchange": request.history_source,
        "decision_timeframe": request.decision_timeframe,
        "ohlcv_gap_count_before_oos": period_gap_count,
        "fee_model": ALPACA_FEE_MODEL,
        "slippage_rate": float(request.slippage_rate),
        "initial_capital_usd_equivalent": CAPITAL,
        "extreme_definition": "Topos/fundos locais pelo CLOSE; confirmação após reversão. Máximas/mínimas absolutas pelas HIGH/LOW apenas descritivas.",
        "threshold_rules": "1%, 2%, 3% ou vol24h*1.2 com limites 1%-4%, congelada no pivô",
        "order_rule": "Confirma no fechamento do candle; executa no próximo OPEN contínuo",
        "absolute_ohlc_development": _absolute_ohlc_extrema(df, *development),
        "absolute_ohlc_validation": _absolute_ohlc_extrema(df, *validation),
        "chosen_only_on_development": (
            selected["parameters"]["policy"] if selected is not None else None
        ),
        "chosen_validation": (
            {k: v for k, v in selected["metrics"].items() if k not in ("development", "quality_checks", "last_confirmations")}
            if selected is not None else None
        ),
        "cash_baseline": baselines["CASH"],
        "buy_hold_baseline": baselines["BUY_HOLD"],
        "oos_executed": False,
        "live_orders_enabled": False,
        "research_warning": (
            "A identificação do pivô histórico usa preços que só confirmam depois; "
            "não é execução no fundo/topo exato. BTCUSDT é proxy aproximada "
            "de BTCUSD da Alpaca. Candles de 30m não mostram trajetória intrabar "
            "nem spread real. A amostra OOS histórica já foi examinada em versões "
            "anteriores: são necessários dados novos para teste independente."
        ),
    }
    return {
        "trials": trials,
        "summary": summary,
        "extrema": full_extrema,
        "trades": full_trades,
    }
