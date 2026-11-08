"""Return magnitude diagnostics use strictly pre-OOS predictions and labels."""
import numpy as np
import pytest

from crypto_paper_trader_ml_api.forecast_quality import describe_return_predictions


@pytest.mark.parametrize("horizon", [2, 6, 24])
def test_repeated_return_diagnostics_are_independent_for_each_horizon(horizon):
    rng = np.random.default_rng(12 + horizon)
    observed = rng.normal(.0002, .01 * (horizon / 6) ** .5, 250)
    forecast = observed * .4 + rng.normal(0, .002, 250)
    result = describe_return_predictions(
        forecast, observed, training_baseline_return=.0002,
        horizon_hours=horizon,
    )
    assert result["horizon_hours"] == horizon
    assert result["rows"] == 250
    assert result["model_mae"] < result["baseline_mae"]
    assert result["mae_skill_vs_training_mean"] > 0
    assert result["prediction_correlation"] > 0
    assert result["prediction_amplitude_ratio"] > 0
    assert sum(x["count"] for x in result["prediction_quantiles"]) == 250
    assert result["prediction_quantiles"][-1]["predicted_mean"] > result["prediction_quantiles"][0]["predicted_mean"]


def test_constant_forecaster_detected_as_no_signal_and_scorable_against_training_mean():
    # A constant return prediction close to zero cannot explain large moves.
    observed = np.array([-.015, -.01, 0, .008, .018, -.012, .004, .017])
    estimated = np.full(8, .0003)
    result = describe_return_predictions(
        estimated, observed, training_baseline_return=.0003,
        horizon_hours=6,
    )
    assert result["prediction_correlation"] is None
    assert result["prediction_amplitude_ratio"] == pytest.approx(0)
    assert result["mae_skill_vs_training_mean"] == pytest.approx(0)
    assert abs(result["mean_predicted_return"]) < .001
    assert result["realized_return_std"] > .01


def test_worse_than_constant_baseline_has_negative_predictive_skill():
    actual = np.array([.01, -.01, .03, -.02, .04, -.03])
    predictions = -actual * 2
    result = describe_return_predictions(
        predictions, actual, training_baseline_return=0,
        horizon_hours=24,
    )
    assert result["prediction_correlation"] == pytest.approx(-1)
    assert result["mae_skill_vs_training_mean"] < 0
    assert result["rmse_skill_vs_training_mean"] < 0
    assert result["prediction_quantiles"][0]["realized_mean"] < result["prediction_quantiles"][-1]["realized_mean"] or result["prediction_correlation"] < 0


def test_rejects_nonfinite_data_and_mismatched_shapes():
    with pytest.raises(ValueError):
        describe_return_predictions(np.array([.1, .2]), np.array([.1]),
                                    training_baseline_return=.0, horizon_hours=6)
    with pytest.raises(ValueError):
        describe_return_predictions(np.array([np.nan]), np.array([.1]),
                                    training_baseline_return=.0, horizon_hours=2)
