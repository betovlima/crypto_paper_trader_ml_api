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
