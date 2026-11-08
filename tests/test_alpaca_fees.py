"""Official Alpaca Crypto spot fee costs in causal BTC/CASH portfolios."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from crypto_paper_trader_ml_api.alpaca_fees import (
    ALPACA_FEE_MODEL, ALPACA_TIERS, AlpacaFeeLedger,
    crypto_fee_tier, round_trip_cost,
)
from crypto_paper_trader_ml_api.btc_policy import DEFAULT_POLICY, simulate_rotations
from crypto_paper_trader_ml_api.capitalization import compare_capitalization

from .test_btc_policy import _candles, _signal
from .test_capitalization import _all_horizon_forecasts, _options


@pytest.mark.parametrize("volume,maker,taker,tier", [
    (0, .0015, .0025, 1),
    (99_999.99, .0015, .0025, 1),
    (100_000, .0012, .0022, 2),
    (500_000, .001, .002, 3),
    (1_000_000, .0008, .0018, 4),
    (10_000_000, .0005, .0015, 5),
    (25_000_000, .0002, .0013, 6),
    (50_000_000, .0002, .0012, 7),
    (100_000_000, 0, .001, 8),
])
def test_all_official_alpaca_crypto_volume_tiers(volume, maker, taker, tier):
    assert crypto_fee_tier(volume, "MAKER")["rate"] == maker
    assert crypto_fee_tier(volume, "TAKER")["rate"] == taker
    assert crypto_fee_tier(volume, "TAKER")["tier"] == tier


def test_tiers_update_next_day_at_3am_new_york_not_during_same_session():
    ledger = AlpacaFeeLedger("TAKER")
    ny = ZoneInfo("America/New_York")
    # 2026-01-07 04:00 NY: day started at 03:00, tier 1.
    before = datetime(2026, 1, 7, 4, 0, tzinfo=ny)
    assert ledger.quote(before)["tier"] == 1
    ledger.record(before, 60_000)
    ledger.record(before + timedelta(minutes=30), 45_000)
    # Same day stays tier 1 despite exceeding $100k already.
    assert ledger.quote(before + timedelta(hours=12))["tier"] == 1
    # At next daily reset, volume from both fills counts.
    after = datetime(2026, 1, 8, 3, 1, tzinfo=ny)
    assert ledger.quote(after)["tier"] == 2
    assert ledger.quote(after)["volume_30d_usd"] == 105_000
    # Previous fills drop out of the 30-day window.
    future = datetime(2026, 2, 10, 4, 0, tzinfo=ny)
    assert ledger.quote(future)["tier"] == 1


def test_alpaca_taker_buy_fee_in_btc_and_sell_fee_in_usd_are_subtracted():
    candles = _candles([(100, 100), (100, 100), (105, 120),
                        (120, 120), (120, 120)])
    decisions = {i: _signal(i) for i in range(1, len(candles))}
    policy = {
        **DEFAULT_POLICY,
        "policy.fee_model": ALPACA_FEE_MODEL,
        "policy.alpaca_liquidity": "TAKER",
        "policy.min_hold_bars": 0,
        "policy.cooldown_bars": 20,
        "policy.take_profit": .08,
        "policy.trailing_stop": .3,
    }
    result = simulate_rotations(
        candles, decisions, policy, initial_capital=10000,
        fee=.001, slippage=0,
    )
    buy, sell = result["actions"]
    assert buy["action"] == "BUY"
    assert sell["action"] == "SELL"
    assert buy["fee_rate"] == sell["fee_rate"] == .0025
    assert buy["fee_asset"] == "BTC"
    assert sell["fee_asset"] == "USD"
    assert buy["fee_asset_amount"] == pytest.approx(10000 / 100 * .0025)
    assert buy["fee_usd_equivalent"] == pytest.approx(25)
    assert buy["fee_tier"] == 1
    assert result["completed_cycles"] == 1
    expected = 10000 * (1 - .0025) * (120 / 100) * (1 - .0025)
    assert result["capital_final"] == pytest.approx(expected)
    assert result["net_profit_usd"] == pytest.approx(expected - 10000)
    assert result["fees_paid"] == pytest.approx(25 + 10000 * (1 - .0025) * 1.2 * .0025)
    assert result["alpaca_fees_usd"] == pytest.approx(result["fees_paid"])
    assert result["slippage_usd_estimate"] == 0


def test_slippage_is_accounted_separately_from_alpaca_fee():
    candles = _candles([(100,100), (100,100), (105,120),
                        (120,120), (120,120)])
    decisions = {i: _signal(i) for i in range(1, len(candles))}
    policy = {
        **DEFAULT_POLICY, "policy.fee_model": ALPACA_FEE_MODEL,
        "policy.take_profit": .08, "policy.cooldown_bars": 20,
        "policy.min_hold_bars": 0, "policy.trailing_stop": .3,
    }
    report = simulate_rotations(candles, decisions, policy, fee=.001, slippage=.0005)
    assert report["fees_paid"] > 0
    assert report["slippage_usd_estimate"] > 0
    assert report["net_profit_usd"] == pytest.approx(report["capital_final"] - 10000)
    assert all(action["slippage_usd_estimate"] >= 0 for action in report["actions"])
    assert report["fee_model"] == ALPACA_FEE_MODEL
    assert report["required_round_trip_return"] == pytest.approx(round_trip_cost(.0025, .0025, .0005))


def test_legacy_policy_has_fixed_unchanged_rate_and_tierless_actions():
    candles = _candles([(100,100), (100,100), (105,120),
                        (120,120), (120,120)])
    decisions = {i: _signal(i) for i in range(1, len(candles))}
    policy = {
        **DEFAULT_POLICY, "policy.take_profit": .08,
        "policy.min_hold_bars": 0, "policy.cooldown_bars": 20,
        "policy.trailing_stop": .3,
    }
    old = simulate_rotations(candles, decisions, policy, fee=.001, slippage=0)
    assert old["fee_model"] == "FIXED"
    assert all(action["fee_rate"] == .001 for action in old["actions"])
    assert all(action["fee_tier"] is None for action in old["actions"])


def test_oos_alpaca_policy_applies_same_schedule_to_buy_and_hold():
    policy = {
        **DEFAULT_POLICY,
        "policy.fee_model": ALPACA_FEE_MODEL,
        "policy.alpaca_liquidity": "TAKER",
        "policy.entry_probability": .52,
        "policy.min_hold_bars": 0,
        "policy.cooldown_bars": 1,
    }
    report = compare_capitalization(
        _all_horizon_forecasts(.0001),
        _all_horizon_forecasts(.015),
        **{**_options(), "horizon": 0}, optimized_policy=policy,
    )
    assert report["fee_model"] == ALPACA_FEE_MODEL
    assert report["fee_liquidity"] == "TAKER"
    assert report["alpaca_fee_schedule"][0]["taker"] == .0025
    assert [v["code"] for v in report["results"]] == ["BTC_CASH_ROTATION", "BUY_AND_HOLD"]
    assert all(v["fee_model"] == ALPACA_FEE_MODEL for v in report["results"])
    assert all(v["alpaca_fees_usd"] > 0 for v in report["results"])
    assert all(v["actions"][0]["fee_asset"] == "BTC" for v in report["results"])
