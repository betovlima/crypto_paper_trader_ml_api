from __future__ import annotations

import pytest

from crypto_paper_trader_ml_api.settings_service import (
    ParameterValidationError,
    get_runtime_parameters,
    update_parameter_rows,
)


def test_runtime_parameters_are_loaded_from_database() -> None:
    parameters = get_runtime_parameters()
    assert parameters.string("backtest.default_symbol")
    assert parameters.integer("xgboost.n_estimators") > 0


def test_cross_parameter_validation_rejects_invalid_thresholds() -> None:
    parameters = get_runtime_parameters()
    original = parameters.floating("backtest.minimum_probability_threshold")
    try:
        with pytest.raises(ParameterValidationError):
            update_parameter_rows(
                [
                    {"key": "backtest.minimum_probability_threshold", "value": 0.96},
                    {"key": "backtest.maximum_probability_threshold", "value": 0.95},
                ]
            )
    finally:
        update_parameter_rows(
            [{"key": "backtest.minimum_probability_threshold", "value": original}]
        )


def test_timeframe_options_are_database_parameters() -> None:
    parameters = get_runtime_parameters()
    assert parameters.string("backtest.default_decision_timeframe") in parameters.list(
        "backtest.allowed_decision_timeframes"
    )
    assert parameters.string("backtest.default_trend_timeframe") in parameters.list(
        "backtest.allowed_trend_timeframes"
    )
    assert parameters.integer("frontend.job_poll_interval_ms") > 0
