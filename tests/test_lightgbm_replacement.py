"""Model-family replacement guardrails: no XGBoost dependency in new ML jobs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier, LGBMRegressor
from crypto_paper_trader_ml_api.engine import _fit_model
from crypto_paper_trader_ml_api.direct_forecast import _regressor
from crypto_paper_trader_ml_api.forecast_diagnostics import tree_diagnostics
from crypto_paper_trader_ml_api.settings_service import list_parameter_rows

from .helpers import default_parameters


def test_classifier_uses_lightgbm_without_changing_oos_calibration_contract():
    t = np.arange(240, dtype=float)
    x = pd.DataFrame({"momentum": np.sin(t / 8), "trend": np.cos(t / 15)})
    y = (np.sin(t / 8) + np.cos(t / 15) > 0).astype(int)
    settings = default_parameters(**{
        "lightgbm.n_estimators": 30,
        "lightgbm.early_stopping_rounds": 5,
        "lightgbm.min_child_samples": 5,
        "lightgbm.n_jobs": 1,
    })
    model = _fit_model(x.iloc[:180], y[:180], x.iloc[180:], y[180:], settings)
    assert isinstance(model, LGBMClassifier)
    probabilities = model.predict_proba(x.iloc[180:])[:, 1]
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    diagnosis = tree_diagnostics(model, probabilities)
    assert diagnosis["trained"] is True
    assert diagnosis["used_trees"] >= 1


def test_regressor_uses_lightgbm_and_preserves_simple_return_units():
    values = np.sin(np.arange(200) / 8)
    data = pd.DataFrame({
        "signal": values,
        "forecast_realized_return": 0.004 * values,
    })
    parameters = default_parameters(**{
        "forecast.regression_n_estimators": 35,
        "forecast.regression_early_stopping_rounds": 5,
        "forecast.regression_num_leaves": 7,
        "forecast.regression_min_child_samples": 5,
        "lightgbm.n_jobs": 1,
    })
    model = _regressor(data.iloc[:150], data.iloc[150:], ["signal"], parameters)
    assert isinstance(model, LGBMRegressor)
    pred = model.predict(data.iloc[150:][["signal"]]) / parameters.floating(
        "forecast.regression_target_scale"
    )
    assert np.isfinite(pred).all()
    assert np.max(np.abs(pred)) < 0.1
    assert tree_diagnostics(model, pred)["used_trees"] >= 1


def test_runtime_catalog_shows_lightgbm_and_hides_obsolete_xgboost_controls():
    rows = list_parameter_rows(public_only=True)
    keys = {item["key"] for item in rows}
    assert "lightgbm.num_leaves" in keys
    assert "lightgbm.min_child_samples" in keys
    assert "forecast.regression_num_leaves" in keys
    assert "forecast.regression_min_child_samples" in keys
    assert not any(key.startswith("xgboost.") for key in keys)
