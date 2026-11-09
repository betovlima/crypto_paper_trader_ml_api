"""Bitcoin-specific, event-driven *research*, never a live trading strategy.

Uses archived BTC OHLCV only. CUSUM detects price events at confirmed candle
CLOSE, buys at the NEXT candle OPEN, and uses Triple Barrier to study events.
Barrier hits observed inside a bar trigger sales at the NEXT bar OPEN, not an
impossible intrabar fill at the barrier price. If both barriers hit within the
same bar, the diagnostic label counts a loss/ambiguity, never an easy win.

Four fixed, interpretable policies plus CASH and BTC buy-and-hold are compared
on chronological pre-OOS development and held-out pre-OOS validation.
No LightGBM, hindsight labels as features, derivatives, future candles,
broker calls, shell scripts, or new CSV downloads.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from io import BytesIO
from math import ceil, exp, isfinite, log, sqrt
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .alpaca_fees import ALPACA_FEE_MODEL, AlpacaFeeLedger
from .btc_policy import _timestamp
from .mexc import TIMEFRAME_SECONDS

POLICIES = (
    ("CUSUM_UP_BREAKOUT", "Continuação após salto de alta"),
    ("CUSUM_DOWN_DIP", "Reversão após queda forte, sem confirmação"),
    ("CUSUM_DOWN_CONFIRM", "Reversão após queda e candle seguinte de recuperação"),
    ("CUSUM_UP_HIGH_VOL", "Continuação de alta somente em volatilidade elevada"),
)
INITIAL_CAPITAL = 10000.0
VERTICAL_HOURS = 6
MIN_EVENT_THRESHOLD = 0.008
THRESHOLD_VOL_MULTIPLIER = 2.2
HOT_VOL_MULTIPLIER = 1.25
MIN_GAIN_BARRIER = .012
MIN_LOSS_BARRIER = .009


def _build_data(history: pd.DataFrame, *, bar_seconds: int) -> pd.DataFrame:
    needed = ("timestamp", "open", "high", "low", "close", "volume")
    if any(col not in history.columns for col in needed):
        raise ValueError("A pesquisa por eventos exige OHLCV completo")
    df = history[list(needed)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if df.empty or df["timestamp"].isna().any() or not df["timestamp"].is_monotonic_increasing or df["timestamp"].duplicated().any():
        raise ValueError("Candles de BTC inválidos ou fora de ordem")
    for col in needed[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if not np.isfinite(df[list(needed[1:])].to_numpy(dtype=float)).all():
        raise ValueError("OHLCV contém preços/volumes inválidos")
    if ((df["low"] <= 0) | (df["low"] > df["open"]) | (df["low"] > df["close"])
            | (df["high"] < df["open"]) | (df["high"] < df["close"])
            | (df["volume"] < 0)).any():
        raise ValueError("OHLCV com máxima, mínima ou volume incoerentes")
    fast_bars = max(48, ceil(24 * 3600 / bar_seconds))
    slow_bars = fast_bars * 7
    ret = np.log(df["close"] / df["close"].shift())
    df["log_return"] = ret
    df["sigma"] = ret.rolling(fast_bars, min_periods=fast_bars).std(ddof=0)
    df["sigma_slow"] = ret.rolling(slow_bars, min_periods=slow_bars).std(ddof=0)
    df["event_threshold"] = np.maximum(
        MIN_EVENT_THRESHOLD, THRESHOLD_VOL_MULTIPLIER * df["sigma"] * sqrt(3600 / bar_seconds),
    )
    df["hot"] = df["sigma"] > HOT_VOL_MULTIPLIER * df["sigma_slow"]
    df["event"] = ""
    pos, neg = 0.0, 0.0
    previous = None
    for idx in range(len(df)):
        row = df.iloc[idx]
        at = row["timestamp"]
        if previous is not None and (at - previous).total_seconds() != bar_seconds:
            pos = neg = 0.0
        previous = at
        delta, threshold = float(row["log_return"]), float(row["event_threshold"])
        if not isfinite(delta) or not isfinite(threshold):
            pos = neg = 0.0
            continue
        pos = max(0.0, pos + delta)
        neg = min(0.0, neg + delta)
        if pos >= threshold:
            df.iat[idx, df.columns.get_loc("event")] = "UP"
            pos = neg = 0.0
        elif neg <= -threshold:
            df.iat[idx, df.columns.get_loc("event")] = "DOWN"
            pos = neg = 0.0
    return df.reset_index(drop=True)


def _barrier_levels(entry: float, sigma: float, bar_seconds: int) -> tuple[float, float]:
    bars = max(1, round(VERTICAL_HOURS * 3600 / bar_seconds))
    vol = sigma * sqrt(bars)
    gain = max(MIN_GAIN_BARRIER, 2.0 * vol)
    loss = max(MIN_LOSS_BARRIER, 1.3 * vol)
    return entry * (1 + gain), entry * (1 - loss)


def _labels(df: pd.DataFrame, lo: int, hi: int, bar_seconds: int) -> dict:
    """Labels are calculated AFTER events, purely for diagnostic evaluation."""
    horizon_bars = max(1, round(VERTICAL_HOURS * 3600 / bar_seconds))
    outcome: dict[str, Counter] = {"UP": Counter(), "DOWN": Counter()}
    for event_idx in range(lo, hi - horizon_bars):
        direction = df.at[event_idx, "event"]
        if direction not in outcome or not isfinite(float(df.at[event_idx, "sigma"])):
            continue
        entry_idx = event_idx + 1
        entry = float(df.at[entry_idx, "open"])
        upper, lower = _barrier_levels(entry, float(df.at[event_idx, "sigma"]), bar_seconds)
        label = "TIMEOUT"
        prior = df.at[event_idx, "timestamp"]
        for j in range(entry_idx, entry_idx + horizon_bars):
            stamp = df.at[j, "timestamp"]
            if (stamp - prior).total_seconds() != bar_seconds:
                label = "DATA_GAP"
                break
            prior = stamp
            high, low = float(df.at[j, "high"]), float(df.at[j, "low"])
            up, down = high >= upper, low <= lower
            if up and down:
                label = "AMBIGUOUS_BOTH_CONSERVATIVE_LOSS"
                break
            if down:
                label = "LOWER_FIRST"
                break
            if up:
                label = "UPPER_FIRST"
                break
        outcome[direction][label] += 1
    return {key: dict(value) for key, value in outcome.items()}


def _eligible(df: pd.DataFrame, idx: int, name: str, seconds: int) -> tuple[bool, int]:
    """The trade at idx OPEN may only use completed bars idx-1 and older."""
    prior = idx - 1
    if prior < 1 or (df.at[idx, "timestamp"] - df.at[prior, "timestamp"]).total_seconds() != seconds:
        return False, prior
    if name == "CUSUM_UP_BREAKOUT":
        return df.at[prior, "event"] == "UP", prior
    if name == "CUSUM_DOWN_DIP":
        return df.at[prior, "event"] == "DOWN", prior
    if name == "CUSUM_UP_HIGH_VOL":
        return df.at[prior, "event"] == "UP" and bool(df.at[prior, "hot"]), prior
    if name == "CUSUM_DOWN_CONFIRM":
        first = prior - 1
        valid = (
            first >= 0 and df.at[first, "event"] == "DOWN"
            and (df.at[prior, "timestamp"] - df.at[first, "timestamp"]).total_seconds() == seconds
            and float(df.at[prior, "close"]) > float(df.at[first, "close"])
        )
        return valid, first
    raise ValueError("Política de evento desconhecida")


def _portfolio(
    df: pd.DataFrame, lo: int, hi: int, name: str, *,
    fee_rate: float, slippage: float, bar_seconds: int,
) -> dict:
    """One BTC/CASH account, full reinvestment; no intrabar imagined fills."""
    cash, btc = INITIAL_CAPITAL, 0.0
    ledger = AlpacaFeeLedger("TAKER")
    fees, slip_paid = 0.0, 0.0
    peak, maxdd = cash, 0.0
    exposure = 0
    buys, sells = 0, 0
    wins, losses = 0, 0
    gross_wins, gross_losses = 0.0, 0.0
    ambiguous, gaps = 0, 0
    position_start, target, stop, cost_basis = -1, 0.0, 0.0, 0.0
    bars = max(1, round(VERTICAL_HOURS * 3600 / bar_seconds))
    equity = []
    actions: list[dict] = []
    for i in range(lo, hi):
        row = df.iloc[i]
        at = row["timestamp"].to_pydatetime()
        opening, closing = float(row["open"]), float(row["close"])
        prev = df.iloc[i - 1] if i > lo else None
        if prev is not None:
            gap = (row["timestamp"] - prev["timestamp"]).total_seconds() != bar_seconds
        else:
            gap = False

        def execute(side: str, price: float, when: datetime, reason: str, ref_idx: int):
            nonlocal cash, btc, fees, slip_paid, buys, sells, wins, losses
            nonlocal gross_wins, gross_losses, cost_basis, position_start
            quote = ledger.quote(when)
            rate = float(quote["rate"])
            if side == "BUY":
                gross = cash
                charge = gross * rate
                slip = gross * slippage / (1 + slippage)
                btc = (gross - charge) / (price * (1 + slippage))
                cash = 0.0
                cost_basis = gross
                position_start = i
                buys += 1
            else:
                gross = btc * price * (1 - slippage)
                charge = gross * rate
                slip = btc * price * slippage
                cash = gross - charge
                pnl = cash - cost_basis
                if pnl > 0:
                    wins += 1
                    gross_wins += pnl
                else:
                    losses += 1
                    gross_losses += -pnl
                btc = 0.0
                position_start = -1
                sells += 1
            fees += charge
            slip_paid += slip
            ledger.record(when, gross)
            actions.append({
                "timestamp": when.isoformat(), "action": side, "reason": reason,
                "signal_candle_index": ref_idx, "fee_rate": rate,
                "fee_tier": quote["tier"], "fee_usd": float(charge),
                "slippage_usd": float(slip), "price": price,
                "capital_after": float(cash + btc * price),
            })

        if i == lo and name == "BUY_HOLD":
            execute("BUY", opening, at, "BUY_AND_HOLD", i - 1)
        elif i > lo:
            if btc > 0 and name != "BUY_HOLD":
                phigh, plow = float(prev["high"]), float(prev["low"])
                upper_hit, lower_hit = phigh >= target, plow <= stop
                time_end = i - position_start >= bars
                if gap or upper_hit or lower_hit or time_end:
                    if gap:
                        gaps += 1
                        reason = "DATA_GAP_EXIT"
                    elif upper_hit and lower_hit:
                        ambiguous += 1
                        reason = "AMBIGUOUS_BOTH_EXIT_NEXT_OPEN"
                    elif lower_hit:
                        reason = "LOWER_BARRIER_EXIT_NEXT_OPEN"
                    elif upper_hit:
                        reason = "UPPER_BARRIER_EXIT_NEXT_OPEN"
                    else:
                        reason = "VERTICAL_TIMEOUT_EXIT_NEXT_OPEN"
                    execute("SELL", opening, at, reason, i - 1)
            if btc == 0 and name not in ("BUY_HOLD", "CASH") and i < hi - 1 and not gap:
                eligible, event_i = _eligible(df, i, name, bar_seconds)
                sigma = float(df.at[event_i, "sigma"]) if eligible else float("nan")
                if eligible and isfinite(sigma):
                    execute("BUY", opening, at, "CUSUM_EVENT_ENTRY_NEXT_OPEN", event_i)
                    target, stop = _barrier_levels(opening, sigma, bar_seconds)
        if btc > 0:
            exposure += 1
        val = float(cash + btc * closing)
        peak = max(peak, val)
        maxdd = min(maxdd, val / peak - 1)
        equity.append(val)
    if btc > 0:
        final = df.iloc[hi - 1]
        when = final["timestamp"].to_pydatetime() + timedelta(seconds=bar_seconds)
        execute("SELL", float(final["close"]), when, "END_OF_WINDOW_CLOSE_LIQUIDATION", hi - 1)
        equity[-1] = cash
        maxdd = min(maxdd, cash / peak - 1)
    returns = [equity[i] / equity[i - 1] - 1 for i in range(1, len(equity)) if equity[i - 1]]
    sd = float(np.std(returns)) if returns else 0.0
    sharpe = (
        float(np.mean(returns) / sd * sqrt(365.25 * 86400 / bar_seconds))
        if sd > 1e-12 else 0.0
    )
    size = max(1, len(equity) - 1)
    blocks = [
        equity[(i + 1) * size // 3] / equity[i * size // 3] - 1
        for i in range(3)
    ] if len(equity) >= 6 else []
    return {
        "policy": name, "capital_final": float(cash),
        "net_profit_usd": float(cash - INITIAL_CAPITAL),
        "return_total": float(cash / INITIAL_CAPITAL - 1),
        "max_drawdown": float(maxdd),
        "sharpe": sharpe,
        "completed_cycles": sells, "buys": buys, "sells": sells,
        "wins": wins, "losses": losses,
        "win_rate": wins / sells if sells else 0.0,
        "profit_factor": gross_wins / gross_losses if gross_losses > 0 else None,
        "fees_paid": float(fees), "alpaca_fees_usd": float(fees),
        "slippage_usd_estimate": float(slip_paid),
        "exposure": exposure / (hi - lo),
        "ambiguous_barrier_exits": ambiguous,
        "gap_exits": gaps,
        "segment_returns": blocks,
        "start": df.at[lo, "timestamp"].isoformat(),
        "end": df.at[hi - 1, "timestamp"].isoformat(),
        "actions": actions[:80],
        "action_count": len(actions),
        "actions_truncated": len(actions) > 80,
        "execution_rule": "Confirmed close signal; next-bar open buy/sell; forced final close liquidation",
    }


def evaluate_archived_event_research(source_archive: bytes) -> dict:
    """Pre-OOS discovery with frozen BTC rules and a separate validation window."""
    from .archive_replay import inspect_source_archive
    request, _ = inspect_source_archive(source_archive)
    if request.symbol not in {"BTCUSDT", "BTCUSD"}:
        raise ValueError("Pesquisa por eventos aceita apenas BTC")
    if request.decision_timeframe not in TIMEFRAME_SECONDS:
        raise ValueError("Timeframe do BTC não suportado")
    seconds = TIMEFRAME_SECONDS[request.decision_timeframe]
    if seconds > 1800:
        raise ValueError("Pesquisa por eventos requer candles de 30 minutos ou mais curtos")
    if not 0 <= request.slippage_rate < .05:
        raise ValueError("Slippage inválido")
    with ZipFile(BytesIO(source_archive)) as z:
        folds = __import__("json").loads(z.read("folds.json"))
        history = pd.read_csv(
            BytesIO(z.read("market_history_decision.csv")),
            parse_dates=["timestamp"], float_precision="round_trip",
        )
    first_oos = min(
        pd.Timestamp(fold["test_start"]) for fold in folds
        if int(fold["horizon_hours"]) in (2, 6, 24)
    )
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
    before = history.loc[history["timestamp"] < first_oos].copy().reset_index(drop=True)
    if len(before) < 1600:
        raise ValueError("Poucos candles antes da primeira janela OOS")
    df = _build_data(before, bar_seconds=seconds)
    bar_count = max(1, round(VERTICAL_HOURS * 3600 / seconds))
    split = int(len(df) * .60)
    embargo = max(24, bar_count)
    development = (max(0, int(len(df) * .05)), split - embargo)
    holdout = (split, len(df))
    if development[1] - development[0] < 500 or holdout[1] - holdout[0] < 300:
        raise ValueError("Janelas pré-OOS insuficientes para experimento")
    labels = {
        "development": _labels(df, *development, seconds),
        "validation": _labels(df, *holdout, seconds),
    }
    trials = []
    for i, (name, description) in enumerate(POLICIES, 1):
        train = _portfolio(
            df, *development, name,
            fee_rate=request.fee_rate, slippage=request.slippage_rate, bar_seconds=seconds,
        )
        validation = _portfolio(
            df, *holdout, name,
            fee_rate=request.fee_rate, slippage=request.slippage_rate, bar_seconds=seconds,
        )
        selected_in_development = train["net_profit_usd"] > 0 and train["completed_cycles"] >= 6
        quality = {
            "training_profitable_and_sufficient_cycles": selected_in_development,
            "validation_positive_after_alpaca": validation["net_profit_usd"] > 0,
            "validation_sufficient_cycles": validation["completed_cycles"] >= 6,
            "validation_two_positive_segments": sum(r > 0 for r in validation["segment_returns"]) >= 2,
        }
        trials.append({
            "trial": i, "status": "RESEARCHED",
            "parameters": {"policy": name, "description": description},
            "metrics": {
                **{k: v for k, v in validation.items() if k != "actions"},
                "development": {k: v for k, v in train.items() if k != "actions"},
                "passed_quality_checks": all(quality.values()),
                "quality_checks": quality,
                "score": (
                    float(np.log(max(1e-12, validation["capital_final"] / INITIAL_CAPITAL))
                    + 0.5 * validation["max_drawdown"])
                ),
            },
        })
    baselines = {
        name: {
            "development": _portfolio(
                df, *development, name,
                fee_rate=request.fee_rate, slippage=request.slippage_rate, bar_seconds=seconds,
            ),
            "validation": _portfolio(
                df, *holdout, name,
                fee_rate=request.fee_rate, slippage=request.slippage_rate, bar_seconds=seconds,
            ),
        }
        for name in ("CASH", "BUY_HOLD")
    }
    candidates = [
        trial for trial in trials
        if trial["metrics"]["development"]["net_profit_usd"] > 0
        and trial["metrics"]["development"]["completed_cycles"] >= 6
    ]
    best_on_development = max(
        candidates, key=lambda row: (
            np.log(max(1e-12, row["metrics"]["development"]["capital_final"] / INITIAL_CAPITAL))
            + .5 * row["metrics"]["development"]["max_drawdown"]
        ),
        default=None,
    )
    chosen_validation = best_on_development["metrics"] if best_on_development else None
    evidence = (
        "PRE_OOS_PROMISING_NEEDS_NEW_DATA"
        if chosen_validation and chosen_validation["passed_quality_checks"]
        else "NO_RELIABLE_EDGE_IDENTIFIED"
    )
    gap_count = int((df["timestamp"].diff().dt.total_seconds().iloc[1:] != seconds).sum())
    return {
        "trials": trials,
        "summary": {
            "experiment": "BTC_CUSUM_TRIPLE_BARRIER_FIXED_BASELINES_V11",
            "result_status": evidence,
            "validation_kind": "PRE_OOS_CHRONOLOGICAL_HOLDOUT_NO_MODEL_FIT",
            "first_oos_untouched_from": first_oos.isoformat(),
            "development_period": [df.at[development[0], "timestamp"].isoformat(),
                                   df.at[development[1] - 1, "timestamp"].isoformat()],
            "validation_period": [df.at[holdout[0], "timestamp"].isoformat(),
                                  df.at[holdout[1] - 1, "timestamp"].isoformat()],
            "embargo_bars": embargo,
            "decision_timeframe": request.decision_timeframe,
            "source_symbol": request.symbol,
            "source_exchange": request.history_source,
            "fee_model": ALPACA_FEE_MODEL,
            "liquidity": "TAKER",
            "slippage_rate": request.slippage_rate,
            "initial_capital_usd_equivalent": INITIAL_CAPITAL,
            "event_threshold_rule": "CUSUM of close-to-close log returns; max(0.8%, 2.2 * rolling_24h_std * sqrt(3600/bar_seconds))",
            "triple_barrier_rule": "Upper=max(1.2%, 2*sigma*sqrt(6h)); lower=max(0.9%, 1.3*sigma*sqrt(6h)); vertical=6h",
            "barrier_fill_rule": "High/low check only at completed candle; exit at next OPEN, never barrier-level intrabar fills",
            "same_bar_both_hit_rule": "Conservative LOSS label, ambiguous recorded separately; simulated sale at next open",
            "labels": labels,
            "ohlcv_gap_count_before_oos": gap_count,
            "best_policy_selected_by_development_only": (
                best_on_development["parameters"]["policy"] if best_on_development else None
            ),
            "chosen_development_score": (
                best_on_development["metrics"]["development"]["net_profit_usd"]
                if best_on_development else None
            ),
            "chosen_validation_result": (
                {k: v for k, v in chosen_validation.items() if k not in ("development",)}
                if chosen_validation else None
            ),
            "cash_baseline": {
                "development": {k: v for k, v in baselines["CASH"]["development"].items() if k != "actions"},
                "validation": {k: v for k, v in baselines["CASH"]["validation"].items() if k != "actions"},
            },
            "buy_hold_baseline": {
                "development": {k: v for k, v in baselines["BUY_HOLD"]["development"].items() if k != "actions"},
                "validation": {k: v for k, v in baselines["BUY_HOLD"]["validation"].items() if k != "actions"},
            },
            "oos_executed": False,
            "live_orders_enabled": False,
            "research_warning": (
                "Candle OHLCV 30m cannot identify intrabar touch order and cannot reproduce 1m "
                "microstructure, real Alpaca BTC/USD spreads, order book, or actual executions. "
                "Results are hypothetical BTC/USDT USD-equivalent historical observations. "
                "The known historical OOS was not read for signal scoring and is not a fresh "
                "independent test across prior versions. No LightGBM was trained."
            ),
            "source_archive_sha256": __import__("hashlib").sha256(source_archive).hexdigest(),
        },
    }
