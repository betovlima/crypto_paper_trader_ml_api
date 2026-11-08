"""LightGBM model builders for the Crypto ML API.

Feature vectors, chronological validation/calibration partitions and OOS evidence
remain in their existing modules; this layer contains only the learners.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation

from .settings_service import RuntimeParameters


def _shared(parameters: RuntimeParameters) -> dict:
    return {
        "max_depth": parameters.integer("lightgbm.max_depth"),
        "num_leaves": parameters.integer("lightgbm.num_leaves"),
        "min_child_samples": parameters.integer("lightgbm.min_child_samples"),
        "min_child_weight": parameters.floating("lightgbm.min_child_weight"),
        "min_split_gain": parameters.floating("lightgbm.min_split_gain"),
        "subsample": parameters.floating("lightgbm.subsample"),
        "subsample_freq": parameters.integer("lightgbm.subsample_freq"),
        "colsample_bytree": parameters.floating("lightgbm.colsample_bytree"),
        "reg_alpha": parameters.floating("lightgbm.reg_alpha"),
        "reg_lambda": parameters.floating("lightgbm.reg_lambda"),
        "max_bin": parameters.integer("lightgbm.max_bin"),
        "random_state": parameters.integer("lightgbm.random_state"),
        "n_jobs": parameters.integer("lightgbm.n_jobs"),
        "importance_type": "gain",
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }


def fit_classifier(
    x_train: pd.DataFrame, y_train: np.ndarray,
    x_validation: pd.DataFrame, y_validation: np.ndarray,
    parameters: RuntimeParameters,
) -> LGBMClassifier | None:
    """Binary classification, with early stopping only on chronological validation."""
    if np.unique(y_train).size < 2:
        return None
    positive = max(1, int(np.count_nonzero(y_train)))
    negative = max(1, int(len(y_train) - positive))
    model = LGBMClassifier(
        objective="binary",
        n_estimators=parameters.integer("lightgbm.n_estimators"),
        learning_rate=parameters.floating("lightgbm.learning_rate"),
        scale_pos_weight=negative / positive,
        **_shared(parameters),
    )
    model.fit(
        x_train, y_train,
        eval_set=[(x_validation, y_validation)],
        eval_metric="binary_logloss",
        callbacks=[
            early_stopping(
                stopping_rounds=parameters.integer("lightgbm.early_stopping_rounds"),
                verbose=False,
            ),
            log_evaluation(period=0),
        ],
    )
    return model


def fit_regressor(
    training: pd.DataFrame, validation: pd.DataFrame,
    columns: list[str], parameters: RuntimeParameters,
) -> LGBMRegressor | None:
    """Predict numerical returns; target scale is reversed before any reporting."""
    scale = parameters.floating("forecast.regression_target_scale")
    outcomes = training["forecast_realized_return"].to_numpy(dtype=float) * scale
    if np.ptp(outcomes) < 1e-8:
        return None
    shared = _shared(parameters)
    # A distinct regression complexity budget is configurable in the Front.
    shared.update({
        "max_depth": parameters.integer("forecast.regression_max_depth"),
        "num_leaves": parameters.integer("forecast.regression_num_leaves"),
        "min_child_samples": parameters.integer("forecast.regression_min_child_samples"),
        "min_child_weight": parameters.floating("forecast.regression_min_child_weight"),
        "min_split_gain": parameters.floating("forecast.regression_gamma"),
        "reg_alpha": parameters.floating("forecast.regression_reg_alpha"),
        "reg_lambda": parameters.floating("forecast.regression_reg_lambda"),
    })
    model = LGBMRegressor(
        objective="regression",
        n_estimators=parameters.integer("forecast.regression_n_estimators"),
        learning_rate=parameters.floating("forecast.regression_learning_rate"),
        **shared,
    )
    model.fit(
        training[columns], outcomes,
        eval_set=[(
            validation[columns],
            validation["forecast_realized_return"].to_numpy(dtype=float) * scale,
        )],
        eval_metric="rmse",
        callbacks=[
            early_stopping(
                stopping_rounds=parameters.integer("forecast.regression_early_stopping_rounds"),
                verbose=False,
            ),
            log_evaluation(period=0),
        ],
    )
    return model
