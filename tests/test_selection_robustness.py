"""V9: reject single-trade and unstable pre-OOS winners; never use OOS."""
import copy
import pytest

from crypto_paper_trader_ml_api.selection_robustness import (
    assess_robustness, MIN_COMPLETED_CYCLES,
    STRESS_FEE_MULTIPLIER, STRESS_SLIPPAGE_MULTIPLIER,
)
from crypto_paper_trader_ml_api.alpaca_fees import ALPACA_FEE_MODEL
from crypto_paper_trader_ml_api.btc_policy import DEFAULT_POLICY, simulate_rotations
from crypto_paper_trader_ml_api.frozen_cost_replay import replay_frozen_rotations
from .test_btc_policy import _candles, _signal


def _good_candidate():
    base = {
        "completed_cycles": 8,
        "segment_cycle_counts": [2, 3, 3],
        "segment_returns": [.08, -.02, .07],
        "net_profit_usd": 900,
        "capital_final": 10900,
        "fees_paid": 180,
        "slippage_usd_estimate": 50,
        "average_win_loss_ratio": 1.5,
        "largest_win_share_of_gross_wins": .35,
    }
    stressed = {
        **base, "net_profit_usd": 450, "capital_final": 10450,
        "fees_paid": 310, "slippage_usd_estimate": 120,
        "frozen_schedule": True, "same_trade_schedule": True,
        "stress_method": "REPRICE_SAME_FILLS_WITH_DYNAMIC_ALPACA_TIERS",
    }
    return base, stressed


def test_diversified_compounding_candidate_passes_all_fixed_gates():
    base, stressed = _good_candidate()
    outcome = assess_robustness(base, stressed)
    assert outcome["passed"]
    assert outcome["reason_codes"] == []
    assert outcome["active_segments"] == 3
    assert outcome["profitable_segments"] == 2
    assert outcome["stress_net_profit_usd"] == 450
    assert outcome["thresholds"]["minimum_completed_cycles"] == MIN_COMPLETED_CYCLES


@pytest.mark.parametrize("changes,reason", [
    ({"completed_cycles": 1, "segment_cycle_counts": [0, 1, 0]}, "MIN_CYCLES"),
    ({"segment_cycle_counts": [0, 1, 7]}, "DISTRIBUTED_CYCLES"),
    ({"segment_returns": [-.02, -.01, .15]}, "POSITIVE_SEGMENTS"),
    ({"net_profit_usd": -20}, "POSITIVE_NET_PROFIT"),
    ({"average_win_loss_ratio": .5}, "AVERAGE_GAIN_VS_LOSS"),
    ({"largest_win_share_of_gross_wins": .9}, "AVOID_SINGLE_WINNING_TRADE"),
])
def test_rejects_single_trade_unstable_or_low_quality_candidates(changes, reason):
    base, stress = _good_candidate()
    base.update(changes)
    outcome = assess_robustness(base, stress)
    assert not outcome["passed"]
    assert reason in outcome["reason_codes"]


def test_rejects_strategy_when_alpaca_stress_wipes_out_profit():
    base, stress = _good_candidate()
    stress["net_profit_usd"], stress["capital_final"] = -5, 9995
    result = assess_robustness(base, stress)
    assert not result["passed"]
    assert "STRESS_POSITIVE_NET_PROFIT" in result["reason_codes"]


def test_stress_uses_same_market_candles_with_cost_multiplier_not_future_prices():
    candles = _candles([(100, 100), (100, 100), (105, 120),
                        (120, 120), (120, 120)])
    decisions = {i: _signal(i) for i in range(1, len(candles))}
    policy = {
        **DEFAULT_POLICY,
        "policy.fee_model": ALPACA_FEE_MODEL,
        "policy.min_hold_bars": 0,
        "policy.take_profit": .08,
        "policy.trailing_stop": .3,
        "policy.cooldown_bars": 20,
    }
    frozen_candles, frozen_signals = copy.deepcopy(candles), copy.deepcopy(decisions)
    baseline = simulate_rotations(candles, decisions, policy, slippage=.0005)
    stressed = replay_frozen_rotations(
        candles, decisions, baseline["actions"], policy,
        slippage=.0005 * STRESS_SLIPPAGE_MULTIPLIER,
        fee_multiplier=STRESS_FEE_MULTIPLIER,
    )
    assert baseline["completed_cycles"] == stressed["completed_cycles"] == 1
    assert baseline["capital_final"] > stressed["capital_final"]
    assert stressed["fees_paid"] > baseline["fees_paid"]
    assert stressed["slippage_usd_estimate"] > baseline["slippage_usd_estimate"]
    assert baseline["segment_cycle_counts"] == stressed["segment_cycle_counts"]
    assert baseline["segment_returns"] != stressed["segment_returns"]
    assert stressed["frozen_schedule"] and stressed["same_trade_schedule"]
    assert all(a["candle_index"] >= 1 for a in baseline["actions"])
    assert candles == frozen_candles and decisions == frozen_signals
    assert baseline["average_win_loss_ratio"] is not None
    assert baseline["largest_win_share_of_gross_wins"] == 1.0
    assert stressed["stress_method"] == "REPRICE_SAME_FILLS_WITH_DYNAMIC_ALPACA_TIERS"

def test_six_hour_forecast_diagnostic_does_not_look_beyond_selection_window():
    from crypto_paper_trader_ml_api.strategy_research import _forecast_realization_diagnostics
    candles = _candles([(100 + i, 100 + i) for i in range(30)])
    signals = {
        i: {
            **_signal(i, value=.008, probability=.65, votes=2),
            "predicted_return": .008 + i * .0001,
        }
        for i in (1, 4, 8, 17, 18, 25)
    }
    policy = {
        **DEFAULT_POLICY, "policy.entry_probability": .5,
        "policy.min_positive_votes": 1, "policy.entry_edge": 0,
    }
    frozen = copy.deepcopy(signals)
    result = _forecast_realization_diagnostics(
        candles, signals, policy, fee=.001, slippage=.0005, bar_seconds=1800,
    )
    # Last start at i=25 has no full 6h path inside scoring; no future OOS.
    assert result["rows"] == 5
    assert result["horizon_hours"] == 6
    assert result["entry_eligible_rows"] == 5
    assert result["mean_realized_return"] > 0
    assert result["return_correlation"] is not None
    assert signals == frozen


def test_repricing_at_original_fees_exactly_reproduces_base_capital_and_fee_ledger():
    candles = _candles([(100, 100), (100, 100), (105, 120),
                        (120, 120), (120, 120)])
    decisions = {i: _signal(i) for i in range(1, len(candles))}
    policy = {**DEFAULT_POLICY,
              "policy.fee_model": ALPACA_FEE_MODEL,
              "policy.min_hold_bars": 0,
              "policy.take_profit": .08,
              "policy.trailing_stop": .3,
              "policy.cooldown_bars": 20}
    base = simulate_rotations(candles, decisions, policy, slippage=.0005)
    replay = replay_frozen_rotations(
        candles, decisions, base["actions"], policy,
        slippage=.0005, fee_multiplier=1.0,
    )
    for key in ("capital_final", "net_profit_usd", "fees_paid",
                "slippage_usd_estimate", "max_drawdown",
                "worst_fold_return", "sharpe"):
        assert replay[key] == pytest.approx(base[key], rel=1e-9), key
    assert replay["completed_cycles"] == base["completed_cycles"]
    assert replay["switch_count"] == base["switch_count"]


def test_stress_does_not_make_entry_disappear_when_expected_return_below_new_cost():
    candles = _candles([(100,100), (100,100), (104,112),
                        (113,113), (113,113)])
    decisions = {i: _signal(i, value=.0075) for i in range(1, len(candles))}
    policy = {
        **DEFAULT_POLICY,
        "policy.fee_model": ALPACA_FEE_MODEL,
        "policy.entry_edge": 0,
        "policy.min_hold_bars": 0, "policy.take_profit": .08,
        "policy.trailing_stop": .3, "policy.cooldown_bars": 20,
    }
    base = simulate_rotations(candles, decisions, policy, slippage=.0005)
    rerun = simulate_rotations(
        candles, decisions, policy, slippage=.001,
        fee_multiplier=1.25,
    )
    replay = replay_frozen_rotations(
        candles, decisions, base["actions"], policy,
        slippage=.001, fee_multiplier=1.25,
    )
    assert base["completed_cycles"] == 1
    assert rerun["completed_cycles"] == 0
    assert rerun["capital_final"] == 10000
    assert replay["completed_cycles"] == 1
    assert [x["action"] for x in replay["actions"]] == ["BUY", "SELL"]
    assert replay["capital_final"] < base["capital_final"]
    assert replay["fees_paid"] > base["fees_paid"]
    assert replay["same_trade_schedule"]


def test_bad_or_missing_frozen_actions_are_rejected_not_silently_rebuilt():
    candles = _candles([(100,100), (100,100), (105,120),
                        (120,120), (120,120)])
    signals = {i: _signal(i) for i in range(1, len(candles))}
    policy = {**DEFAULT_POLICY, "policy.min_hold_bars": 0,
              "policy.take_profit": .08, "policy.trailing_stop": .3}
    base = simulate_rotations(candles, signals, policy)
    corrupted = copy.deepcopy(base["actions"])
    corrupted[0]["price"] = 999
    with pytest.raises(ValueError, match="Preço"):
        replay_frozen_rotations(candles, signals, corrupted, policy)
    with pytest.raises(ValueError, match="posição final"):
        replay_frozen_rotations(candles, signals, base["actions"][:1], policy)


def test_stress_without_frozen_schedule_cannot_pass_robustness_gate():
    base, stressed = _good_candidate()
    stressed["frozen_schedule"] = False
    outcome = assess_robustness(base, stressed)
    assert not outcome["passed"]
    assert "SAME_TRADES_UNDER_STRESS" in outcome["reason_codes"]



def test_full_frozen_audit_replays_more_than_default_200_trades():
    candles = _candles([(100, 100)] * 570)
    signals = {
        i: _signal(i, value=.025 if i % 2 else -.02,
                   probability=.8 if i % 2 else .1)
        for i in range(1, len(candles))
    }
    policy = {
        **DEFAULT_POLICY, "policy.fee_model": ALPACA_FEE_MODEL,
        "policy.entry_edge": 0, "policy.min_hold_bars": 0,
        "policy.cooldown_bars": 0, "policy.take_profit": .5,
        "policy.stop_loss": .3, "policy.trailing_stop": .3,
    }
    baseline = simulate_rotations(
        candles, signals, policy, slippage=.0005,
        audit_limit=2 * len(candles),
    )
    assert baseline["switch_count"] > 200
    assert not baseline["actions_truncated"]
    exact_reprice = replay_frozen_rotations(
        candles, signals, baseline["actions"], policy,
        slippage=.0005, fee_multiplier=1,
    )
    stressed = replay_frozen_rotations(
        candles, signals, baseline["actions"], policy,
        slippage=.001, fee_multiplier=1.25,
    )
    assert exact_reprice["capital_final"] == pytest.approx(
        baseline["capital_final"], rel=1e-9
    )
    assert exact_reprice["completed_cycles"] == baseline["completed_cycles"]
    assert stressed["switch_count"] == baseline["switch_count"]
    assert stressed["completed_cycles"] == baseline["completed_cycles"]
    assert stressed["capital_final"] < baseline["capital_final"]
    assert stressed["same_trade_schedule"]


