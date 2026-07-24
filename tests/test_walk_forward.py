from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_paper_trader_ml_api.engine import walk_forward_predict
from crypto_paper_trader_ml_api.features import build_features, create_labels

from .helpers import default_parameters


def test_walk_forward_generates_out_of_sample_probabilities() -> None:
    rng = np.random.default_rng(7)
    size = 2200
    timestamps = pd.date_range("2023-01-01", periods=size, freq="30min", tz="UTC")
    returns = rng.normal(0.00005, 0.006, size) + 0.001 * np.sin(np.arange(size) / 30)
    close = 100 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = close * 0.004
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": np.maximum(open_, close) + spread * rng.random(size),
            "low": np.minimum(open_, close) - spread * rng.random(size),
            "close": close,
            "volume": rng.lognormal(7, 0.5, size),
            "close_time_ms": 0,
            "quote_volume": 0,
        }
    )
    parameters = default_parameters(**{"xgboost.n_jobs": 1})
    trend = frame.iloc[::8].reset_index(drop=True)
    labelled = create_labels(build_features(frame, parameters, trend), parameters)
    predicted, metadata = walk_forward_predict(
        labelled,
        parameters,
        lambda _value, _stage: None,
        lambda: False,
    )

    assert len(predicted) >= 700
    assert predicted["entry_probability"].between(0, 1).all()
    assert predicted["exit_probability"].between(0, 1).all()
    assert metadata["fold_count"] >= 3
    assert metadata["feature_importance"]
