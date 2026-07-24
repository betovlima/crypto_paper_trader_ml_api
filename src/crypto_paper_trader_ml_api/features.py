from __future__ import annotations

import numpy as np
import pandas as pd

from .settings_service import RuntimeParameters


def feature_names(parameters: RuntimeParameters) -> list[str]:
    ema_periods = [int(value) for value in parameters.list("features.ema_periods")]
    if len(ema_periods) < 3:
        raise ValueError("features.ema_periods must contain at least three periods")
    fast, medium, slow = ema_periods[:3]
    rsi_period = parameters.integer("features.rsi_period")
    atr_period = parameters.integer("features.atr_period")
    adx_period = parameters.integer("features.adx_period")
    return_periods = [int(value) for value in parameters.list("features.return_periods")]
    volatility_period = parameters.integer("features.volatility_period")
    bollinger_period = parameters.integer("features.bollinger_period")
    stochastic_period = parameters.integer("features.stochastic_period")
    range_period = parameters.integer("features.range_period")
    efficiency_period = parameters.integer("features.range_efficiency_period")
    fib_lookback = parameters.integer("features.fibonacci_lookback")
    fibonacci_levels = [float(value) for value in parameters.list("features.fibonacci_levels")]

    names = [
        f"ema_gap_{fast}_{medium}",
        f"ema_gap_{medium}_{slow}",
        f"price_gap_ema_{fast}",
        f"price_gap_ema_{medium}",
        f"price_gap_ema_{slow}",
        f"rsi_{rsi_period}",
        f"atr_pct_{atr_period}",
        f"adx_{adx_period}",
        "relative_volume",
        *[f"return_{period}" for period in return_periods],
        f"volatility_{volatility_period}",
        f"bollinger_zscore_{bollinger_period}",
        f"bollinger_width_{bollinger_period}",
        f"stochastic_k_{stochastic_period}",
        f"range_position_{range_period}",
        f"range_efficiency_{efficiency_period}",
        "candle_body_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        f"fib_position_{fib_lookback}",
        *[f"distance_fib_{int(round(level * 1000)):03d}" for level in fibonacci_levels],
        "trend_ema_gap",
        "trend_ema_slope",
    ]
    return names


def atr_column(parameters: RuntimeParameters) -> str:
    return f"atr_{parameters.integer('features.atr_period')}"


def signal_ema_columns(parameters: RuntimeParameters) -> tuple[str, str, str]:
    ema_periods = [int(value) for value in parameters.list("features.ema_periods")]
    if len(ema_periods) < 3:
        raise ValueError("features.ema_periods must contain at least three periods")
    fast, medium, slow = ema_periods[:3]
    return f"ema_{fast}", f"ema_{medium}", f"ema_{slow}"


def _true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    return pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def build_features(
    decision: pd.DataFrame,
    parameters: RuntimeParameters,
    trend: pd.DataFrame | None = None,
) -> pd.DataFrame:
    data = decision.copy().sort_values("timestamp").reset_index(drop=True)
    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    open_ = data["open"].astype(float)
    volume = data["volume"].astype(float)
    safe_close = close.replace(0, np.nan)

    ema_periods = [int(value) for value in parameters.list("features.ema_periods")]
    if len(ema_periods) < 3:
        raise ValueError("features.ema_periods must contain at least three periods")
    fast_ema, medium_ema, slow_ema = ema_periods[:3]
    for period in ema_periods:
        data[f"ema_{period}"] = close.ewm(span=period, adjust=False, min_periods=period).mean()

    rsi_period = parameters.integer("features.rsi_period")
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    data[f"rsi_{rsi_period}"] = (100 - 100 / (1 + rs)).fillna(50.0)

    atr_period = parameters.integer("features.atr_period")
    tr = _true_range(data)
    data[f"atr_{atr_period}"] = tr.ewm(alpha=1 / atr_period, adjust=False, min_periods=atr_period).mean()
    data[f"atr_pct_{atr_period}"] = data[f"atr_{atr_period}"] / safe_close

    adx_period = parameters.integer("features.adx_period")
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=data.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=data.index)
    smoothed_tr = tr.ewm(alpha=1 / adx_period, adjust=False, min_periods=adx_period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / adx_period, adjust=False, min_periods=adx_period).mean() / smoothed_tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / adx_period, adjust=False, min_periods=adx_period).mean() / smoothed_tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    data[f"adx_{adx_period}"] = dx.ewm(alpha=1 / adx_period, adjust=False, min_periods=adx_period).mean().fillna(0)

    volume_period = parameters.integer("features.volume_period")
    data["average_volume"] = volume.rolling(volume_period, min_periods=volume_period).mean()
    data["relative_volume"] = volume / data["average_volume"].replace(0, np.nan)

    return_periods = [int(value) for value in parameters.list("features.return_periods")]
    for period in return_periods:
        data[f"return_{period}"] = close.pct_change(period)
    base_return_column = f"return_{return_periods[0]}"

    volatility_period = parameters.integer("features.volatility_period")
    data[f"volatility_{volatility_period}"] = data[base_return_column].rolling(
        volatility_period, min_periods=volatility_period
    ).std()

    bollinger_period = parameters.integer("features.bollinger_period")
    bb_mean = close.rolling(bollinger_period, min_periods=bollinger_period).mean()
    bb_std = close.rolling(bollinger_period, min_periods=bollinger_period).std().replace(0, np.nan)
    data[f"bollinger_zscore_{bollinger_period}"] = (close - bb_mean) / bb_std
    width_multiplier = parameters.floating("features.bollinger_width_multiplier")
    data[f"bollinger_width_{bollinger_period}"] = width_multiplier * bb_std / safe_close

    stochastic_period = parameters.integer("features.stochastic_period")
    rolling_low = low.rolling(stochastic_period, min_periods=stochastic_period).min()
    rolling_high = high.rolling(stochastic_period, min_periods=stochastic_period).max()
    data[f"stochastic_k_{stochastic_period}"] = (
        100 * (close - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)
    )

    range_period = parameters.integer("features.range_period")
    support = low.rolling(range_period, min_periods=range_period).min()
    resistance = high.rolling(range_period, min_periods=range_period).max()
    data[f"range_position_{range_period}"] = (
        (close - support) / (resistance - support).replace(0, np.nan)
    )

    efficiency_period = parameters.integer("features.range_efficiency_period")
    path = close.diff().abs().rolling(efficiency_period, min_periods=efficiency_period).sum().replace(0, np.nan)
    data[f"range_efficiency_{efficiency_period}"] = close.diff(efficiency_period).abs() / path

    candle_top = pd.concat([open_, close], axis=1).max(axis=1)
    candle_bottom = pd.concat([open_, close], axis=1).min(axis=1)
    data["candle_body_pct"] = (close - open_) / safe_close
    data["upper_wick_pct"] = (high - candle_top) / safe_close
    data["lower_wick_pct"] = (candle_bottom - low) / safe_close

    fib_lookback = parameters.integer("features.fibonacci_lookback")
    swing_low = low.rolling(fib_lookback, min_periods=fib_lookback).min()
    swing_high = high.rolling(fib_lookback, min_periods=fib_lookback).max()
    impulse = (swing_high - swing_low).replace(0, np.nan)
    data[f"fib_position_{fib_lookback}"] = (close - swing_low) / impulse
    for level in [float(value) for value in parameters.list("features.fibonacci_levels")]:
        fib_price = swing_high - level * impulse
        suffix = int(round(level * 1000))
        data[f"distance_fib_{suffix:03d}"] = (close - fib_price) / safe_close

    data[f"ema_gap_{fast_ema}_{medium_ema}"] = (
        data[f"ema_{fast_ema}"] - data[f"ema_{medium_ema}"]
    ) / safe_close
    data[f"ema_gap_{medium_ema}_{slow_ema}"] = (
        data[f"ema_{medium_ema}"] - data[f"ema_{slow_ema}"]
    ) / safe_close
    for period in (fast_ema, medium_ema, slow_ema):
        data[f"price_gap_ema_{period}"] = (close - data[f"ema_{period}"]) / safe_close

    trend_fast = parameters.integer("features.trend_ema_fast")
    trend_slow = parameters.integer("features.trend_ema_slow")
    trend_slope_period = parameters.integer("features.trend_slope_period")
    if trend is not None and not trend.empty:
        trend_frame = trend.copy().sort_values("timestamp")
        trend_close = trend_frame["close"].astype(float)
        trend_frame["trend_ema_fast"] = trend_close.ewm(
            span=trend_fast, adjust=False, min_periods=trend_fast
        ).mean()
        trend_frame["trend_ema_slow"] = trend_close.ewm(
            span=trend_slow, adjust=False, min_periods=trend_slow
        ).mean()
        trend_frame["trend_ema_gap"] = (
            trend_frame["trend_ema_fast"] - trend_frame["trend_ema_slow"]
        ) / trend_close.replace(0, np.nan)
        trend_frame["trend_ema_slope"] = trend_frame["trend_ema_fast"].pct_change(trend_slope_period)
        data = pd.merge_asof(
            data.sort_values("timestamp"),
            trend_frame[["timestamp", "trend_ema_gap", "trend_ema_slope"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
    else:
        data["trend_ema_gap"] = data[f"ema_gap_{medium_ema}_{slow_ema}"]
        fallback_period = parameters.integer("features.fallback_trend_slope_period")
        data["trend_ema_slope"] = data[f"ema_{medium_ema}"].pct_change(fallback_period)

    columns = feature_names(parameters)
    data[columns] = data[columns].replace([np.inf, -np.inf], np.nan)
    return data


def create_labels(frame: pd.DataFrame, parameters: RuntimeParameters) -> pd.DataFrame:
    data = frame.copy()
    closes = data["close"].to_numpy(float)
    highs = data["high"].to_numpy(float)
    lows = data["low"].to_numpy(float)
    atr = data[atr_column(parameters)].to_numpy(float)
    entry_labels = np.full(len(data), np.nan)
    exit_labels = np.full(len(data), np.nan)

    horizon = parameters.integer("labels.horizon_candles")
    entry_target_atr = parameters.floating("labels.entry_target_atr")
    entry_stop_atr = parameters.floating("labels.entry_stop_atr")
    exit_target_atr = parameters.floating("labels.exit_target_atr")
    exit_invalidation_atr = parameters.floating("labels.exit_invalidation_atr")

    for index in range(len(data) - horizon - 1):
        if not np.isfinite(atr[index]) or atr[index] <= 0:
            continue
        entry = closes[index]
        entry_target = entry + entry_target_atr * atr[index]
        entry_stop = entry - entry_stop_atr * atr[index]
        exit_target = entry - exit_target_atr * atr[index]
        exit_invalidation = entry + exit_invalidation_atr * atr[index]

        entry_outcome = 0.0
        exit_outcome = 0.0
        for future in range(index + 1, index + horizon + 1):
            if lows[future] <= entry_stop:
                entry_outcome = 0.0
                break
            if highs[future] >= entry_target:
                entry_outcome = 1.0
                break
        for future in range(index + 1, index + horizon + 1):
            if highs[future] >= exit_invalidation:
                exit_outcome = 0.0
                break
            if lows[future] <= exit_target:
                exit_outcome = 1.0
                break
        entry_labels[index] = entry_outcome
        exit_labels[index] = exit_outcome

    data["entry_label"] = entry_labels
    data["exit_label"] = exit_labels
    return data
