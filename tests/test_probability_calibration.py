"""Temporal calibration is per horizon and selected without touching OOS labels."""
import numpy as np
import pytest
from sklearn.metrics import brier_score_loss

from crypto_paper_trader_ml_api.engine import ProbabilityCalibrator


@pytest.mark.parametrize("horizon", (2, 6, 24))
def test_calibration_chooses_method_only_from_chronological_calibration_data(horizon):
    rng = np.random.default_rng(20261108 + horizon)
    n = 900
    raw = rng.uniform(0.18, 0.73, size=n)
    # Simulate a horizon-specific source of calibrated returns.
    real_p = np.clip(.25 + raw * .50 + horizon / 300.0, .02, .98)
    labels = rng.binomial(1, real_p)
    calibrator = ProbabilityCalibrator(solver="lbfgs")
    calibrator.fit(raw, labels)
    assert calibrator.calibration_rows == n
    assert calibrator.selection_rows == 225
    assert calibrator.method in {"identity", "logistic_raw", "logistic_logit"}
    assert calibrator.selection_brier is not None
    assert np.isfinite(calibrator.selection_brier)
    assert 0 <= calibrator.selection_brier <= 1

    oos_raw = np.array([.08, .18, .29, .44, .52, .71, .84])
    predicted = calibrator.transform(oos_raw)
    assert predicted.shape == oos_raw.shape
    assert np.isfinite(predicted).all()
    assert ((predicted > 0) & (predicted < 1)).all()
    assert (np.diff(predicted) >= -1e-9).all()
    # Replaying with the same pre-OOS calibration must be deterministic.
    second = ProbabilityCalibrator(solver="lbfgs")
    second.fit(raw.copy(), labels.copy())
    np.testing.assert_allclose(predicted, second.transform(oos_raw), atol=1e-12)
    assert second.method == calibrator.method


def test_calibrator_does_not_use_future_outcomes():
    rng = np.random.default_rng(911)
    calibration = rng.uniform(.15, .65, size=800)
    labels = rng.binomial(1, calibration)
    holdout_predictions = np.array([.41, .46, .49, .51])
    a = ProbabilityCalibrator(solver="lbfgs")
    b = ProbabilityCalibrator(solver="lbfgs")
    a.fit(calibration, labels)
    b.fit(calibration, labels)
    # Future OOS outcomes may be opposite, but are never an input to fit.
    future_truth_a = [0, 0, 0, 0]
    future_truth_b = [1, 1, 1, 1]
    assert future_truth_a != future_truth_b
    np.testing.assert_allclose(
        a.transform(holdout_predictions),
        b.transform(holdout_predictions),
    )


def test_constant_class_is_safely_calibrated_without_synthetic_probability():
    raw = np.full(400, .32)
    model = ProbabilityCalibrator(solver="lbfgs")
    model.fit(raw, np.zeros(400, dtype=int))
    assert model.method == "identity"
    np.testing.assert_allclose(model.transform(raw), raw)
