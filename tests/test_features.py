from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_paper_trader_ml_api.features import build_features, create_labels, feature_names

from .helpers import default_parameters


def synthetic_candles(size: int = 1600) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=size, freq="30min", tz="UTC")
    base = 100 + np.linspace(0, 25, size) + 2 * np.sin(np.arange(size) / 20)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": base - 0.15,
            "high": base + 0.8,
            "low": base - 0.8,
            "close": base,
            "volume": 1000 + 100 * np.sin(np.arange(size) / 7),
            "close_time_ms": (timestamps.astype("int64") // 1_000_000) + 1_799_999,
            "quote_volume": base * 1000,
        }
    )


def test_feature_engineering_produces_expected_columns() -> None:
    parameters = default_parameters()
    frame = build_features(synthetic_candles(), parameters)
    assert set(feature_names(parameters)).issubset(frame.columns)
    assert frame.dropna(subset=feature_names(parameters)).shape[0] > 1000


def test_labels_are_binary_for_usable_rows() -> None:
    parameters = default_parameters()
    labelled = create_labels(build_features(synthetic_candles(), parameters), parameters)
    values = set(labelled["entry_label"].dropna().unique())
    assert values.issubset({0.0, 1.0})


def test_four_hour_indicators_are_not_visible_until_their_candle_closes():
    parameters = default_parameters(**{
        "features.trend_ema_fast": 2,
        "features.trend_ema_slow": 3,
        "features.trend_slope_period": 1,
    })
    decision = synthetic_candles(96)
    trend = decision.iloc[::8].copy().reset_index(drop=True)
    original = build_features(
        decision, parameters, trend,
        decision_timeframe="30min", trend_timeframe="4hour",
    )
    changed = trend.copy()
    # 08:00 candle closes at 12:00, so changing it cannot alter
    # any 30-minute indicator whose observation time precedes 12:00.
    changed.loc[2, "close"] = changed.loc[2, "close"] * 5
    updated = build_features(
        decision, parameters, changed,
        decision_timeframe="30min", trend_timeframe="4hour",
    )
    before_close = original["decision_available_at"] < pd.Timestamp(
        "2024-01-01T12:00:00Z"
    )
    pd.testing.assert_series_equal(
        original.loc[before_close, "trend_ema_gap"],
        updated.loc[before_close, "trend_ema_gap"],
    )
    pd.testing.assert_series_equal(
        original.loc[before_close, "trend_ema_slope"],
        updated.loc[before_close, "trend_ema_slope"],
    )
    eligible = original.loc[
        original["timestamp"] == pd.Timestamp("2024-01-01T11:30:00Z")
    ].iloc[0]
    assert eligible["trend_source_open_at"] == pd.Timestamp("2024-01-01T08:00:00Z")
    assert eligible["trend_available_at"] <= eligible["decision_available_at"]
    assert updated.loc[~before_close, "trend_ema_gap"].ne(
        original.loc[~before_close, "trend_ema_gap"]
    ).any()


def test_future_label_availability_is_recorded_for_fold_purging():
    params = default_parameters(**{"labels.horizon_candles": 4})
    featured = build_features(
        synthetic_candles(40), params,
        decision_timeframe="30min",
    )
    labelled = create_labels(featured, params)
    assert labelled["label_available_at"].iloc[0] == pd.Timestamp(
        "2024-01-01T02:30:00Z"
    )
    assert pd.isna(labelled["label_available_at"].iloc[-1])
    assert pd.isna(labelled["entry_label"].iloc[-1])
    assert pd.notna(labelled["entry_label"].iloc[-5])


def test_market_halt_resets_indicators_and_disallows_labels_crossing_the_gap():
    params = default_parameters(**{"labels.horizon_candles": 12})
    raw = synthetic_candles(520)
    # 30-minute bars 200 and 201 are genuinely absent. Never create them.
    interrupted = raw.drop(index=[200, 201]).reset_index(drop=True)
    featured = build_features(interrupted, params, decision_timeframe="30min")
    assert len(featured) == 518
    assert featured["market_segment"].nunique() == 2
    assert featured.loc[199, "market_segment"] == 0
    assert featured.loc[200, "market_segment"] == 1
    assert pd.isna(featured.loc[200, "return_1"])
    assert pd.isna(featured.loc[200, "atr_14"])
    assert pd.isna(featured.loc[200, "ema_21"])
    assert pd.notna(featured.loc[250, "return_1"])
    assert pd.notna(featured.loc[250, "atr_14"])
    labelled = create_labels(featured, params)
    assert pd.isna(labelled.loc[198, "entry_label"])
    assert pd.isna(labelled.loc[198, "exit_label"])
    assert pd.isna(labelled.loc[198, "label_available_at"])
    assert pd.notna(labelled.loc[187, "entry_label"])
    assert pd.notna(labelled.loc[250, "entry_label"])
    assert raw.loc[200, "timestamp"] not in set(labelled["timestamp"])
    assert raw.loc[201, "timestamp"] not in set(labelled["timestamp"])
