"""Causal BTC/CASH decision-policy tests: no future-price fills or invented stops."""
from datetime import datetime, timedelta, timezone

import pytest

from crypto_paper_trader_ml_api.btc_policy import (
    DEFAULT_POLICY, aggregate_signals, simulate_rotations, validate_policy,
)


def _candles(values, *, starts_at=None):
    begin = starts_at or datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        {"timestamp": begin + timedelta(minutes=30 * index),
         "open": float(op), "close": float(cl)}
        for index, (op, cl) in enumerate(values)
    ]


def _signal(index, *, value=.05, probability=.85, votes=3):
    return {
        "signal_at": (datetime(2024, 1, 1, tzinfo=timezone.utc)
                      + timedelta(minutes=30 * index)).isoformat(),
        "predicted_return": value, "p_up": probability,
        "positive_votes": votes,
    }


def test_policy_entry_full_reinvestment_and_close_confirmed_take_profit():
    candles = _candles([(100, 100), (100, 100), (105, 112),
                        (111, 111), (111, 111), (111, 111)])
    signals = {i: _signal(i) for i in range(1, len(candles))}
    policy = {**DEFAULT_POLICY, "policy.min_hold_bars": 0,
              "policy.take_profit": .05, "policy.cooldown_bars": 10,
              "policy.trailing_stop": .20}
    result = simulate_rotations(
        candles, signals, policy, initial_capital=10000, fee=0, slippage=0,
    )
    assert result["completed_cycles"] == 1
    assert result["wins"] == 1
    assert [x["action"] for x in result["actions"]] == ["BUY", "SELL"]
    assert result["actions"][0]["timestamp"].startswith("2024-01-01T00:30")
    assert result["actions"][1]["timestamp"].startswith("2024-01-01T01:30")
    assert result["actions"][1]["reason"] == "TAKE_PROFIT"
    assert result["capital_final"] == pytest.approx(11100)
    assert result["timeline"][-1]["position"] == "CASH"


def test_stop_fills_next_open_not_unavailable_intrabar_price():
    candles = _candles([(100, 100), (100, 100), (97, 90),
                        (86, 86), (87, 87)])
    signals = {i: _signal(i) for i in range(1, len(candles))}
    policy = {**DEFAULT_POLICY, "policy.stop_loss": .05,
              "policy.min_hold_bars": 40, "policy.take_profit": .3,
              "policy.trailing_stop": .30, "policy.cooldown_bars": 20}
    result = simulate_rotations(candles, signals, policy, fee=0, slippage=0)
    assert result["completed_cycles"] == 1
    assert result["actions"][1]["reason"] == "STOP_LOSS"
    assert result["actions"][1]["price"] == 86
    assert result["capital_final"] == pytest.approx(8600)



def test_protective_stop_works_without_new_model_predictions():
    candles = _candles([(100, 100), (100, 100), (98, 90),
                        (83, 83), (83, 83)])
    policy = {**DEFAULT_POLICY, "policy.stop_loss": .05,
              "policy.trailing_stop": .3, "policy.take_profit": .3,
              "policy.cooldown_bars": 10}
    result = simulate_rotations(
        candles, {1: _signal(1), 4: _signal(4, probability=.3)},
        policy, fee=0, slippage=0,
    )
    assert result["completed_cycles"] == 1
    assert result["actions"][1]["reason"] == "STOP_LOSS"
    assert result["actions"][1]["signal_at"] is None
    assert result["actions"][1]["price"] == 83
    assert result["capital_final"] == pytest.approx(8300)


def test_full_audit_keeps_every_equity_candle():
    candles = _candles([(100, 100)] * 1003)
    signals = {i: _signal(i, value=-.01, probability=.3)
               for i in range(1, len(candles))}
    full = simulate_rotations(
        candles, signals, DEFAULT_POLICY, timeline_limit=None,
    )
    short = simulate_rotations(
        candles, signals, DEFAULT_POLICY, timeline_limit=500,
    )
    assert len(full["timeline"]) == 1002
    assert len(short["timeline"]) <= 501
    assert full["capital_final"] == short["capital_final"] == 10000



def test_reentry_cooldown_and_single_operation_per_candle():
    candles = _candles([(100, 100), (100, 100), (110, 110),
                        (111, 111), (111, 111), (111, 111), (112, 112)])
    signals = {i: _signal(i) for i in range(1, len(candles))}
    policy = {**DEFAULT_POLICY, "policy.take_profit": .05,
              "policy.cooldown_bars": 15, "policy.min_hold_bars": 0}
    result = simulate_rotations(candles, signals, policy, fee=0, slippage=0)
    assert [a["action"] for a in result["actions"]] == ["BUY", "SELL"]
    assert result["switch_count"] == 2
    assert result["exposure"] < 1


def test_joint_forecast_weights_and_vote_threshold_validation():
    timestamp = "2024-01-01T00:30:00+00:00"
    inputs = {
        2: {1: {"signal_at": timestamp, "predicted_return": .02, "p_up": .8}},
        6: {1: {"signal_at": timestamp, "predicted_return": .03, "p_up": .75}},
        24: {1: {"signal_at": timestamp, "predicted_return": -.04, "p_up": .35}},
    }
    policy = {**DEFAULT_POLICY, "policy.weight_2h": 1.0,
              "policy.weight_6h": .5, "policy.weight_24h": .2}
    joint = aggregate_signals(inputs, policy)
    assert joint[1]["positive_votes"] == 2
    assert 0.5 < joint[1]["p_up"] < .8
    assert joint[1]["predicted_return"] > 0
    with pytest.raises(ValueError, match="histerese"):
        validate_policy({**DEFAULT_POLICY,
                         "policy.entry_probability": .55,
                         "policy.exit_probability": .6})


def test_prohibit_unsupported_fee_and_unavailable_next_bar():
    candles = _candles([(100, 100), (100, 100), (100, 100)])
    with pytest.raises(ValueError, match="Taxas"):
        simulate_rotations(candles, {1: _signal(1)}, DEFAULT_POLICY, fee=.2)
    # A 30-minute missing bar blocks the signal instead of executing it stale.
    candles[1]["timestamp"] += timedelta(minutes=30)
    candles[2]["timestamp"] += timedelta(minutes=30)
    out = simulate_rotations(candles, {1: _signal(1), 2: _signal(2, probability=.3)}, DEFAULT_POLICY)
    assert out["switch_count"] == 0
