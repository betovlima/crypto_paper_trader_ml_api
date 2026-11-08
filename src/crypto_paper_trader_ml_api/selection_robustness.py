"""Hard selection gates applied exclusively to PRE-OOS capital simulations.

No OOS data, future predictions, target labels or exchange information is
accepted here. More trades are not a financial objective: they provide enough
independent observations to reject one-trade 'winners'. Thresholds are fixed
before evaluation and are not tuned against the untouched final OOS.
"""
from __future__ import annotations

from typing import Any
import math

MIN_COMPLETED_CYCLES = 6
MIN_ACTIVE_SEGMENTS = 2
MIN_PROFITABLE_SEGMENTS = 2
MIN_CYCLES_PER_ACTIVE_SEGMENT = 2
MIN_GAIN_LOSS_RATIO = 0.75
MAX_SINGLE_WIN_SHARE = 0.70
MIN_STRESS_NET_PROFIT_USD = 0.0
# Assumed execution uncertainty, in addition to actual Alpaca maker/taker
# schedule; never mislabel a hypothetical fee uplift as an Alpaca charge.
STRESS_FEE_MULTIPLIER = 1.25
STRESS_SLIPPAGE_MULTIPLIER = 2.0


def assess_robustness(base: dict[str, Any], stressed: dict[str, Any]) -> dict[str, Any]:
    """Classify whether a candidate has *sufficient evidence*, not whether it
    will be profitable in live trading.

    Cycle counts and net gains are after fees. Segment returns are the
    contiguous, chronological mark-to-market periods of ONE compound account.
    """
    cycles = int(base["completed_cycles"])
    segments = [int(v) for v in base["segment_cycle_counts"]]
    returns = [float(v) for v in base["segment_returns"]]
    if len(segments) != 3 or len(returns) != 3:
        raise ValueError("A validação econômica requer três segmentos temporais")
    active = sum(v >= MIN_CYCLES_PER_ACTIVE_SEGMENT for v in segments)
    positive = sum(v > 0 for v in returns)
    payoff = base.get("average_win_loss_ratio")
    concentration = float(base.get("largest_win_share_of_gross_wins") or 0)
    net = float(base["net_profit_usd"])
    stressed_net = float(stressed["net_profit_usd"])
    checks = {
        "min_cycles": cycles >= MIN_COMPLETED_CYCLES,
        "distributed_cycles": active >= MIN_ACTIVE_SEGMENTS,
        "positive_segments": positive >= MIN_PROFITABLE_SEGMENTS,
        "positive_net_profit": net > 0,
        "stress_positive_net_profit": stressed_net > MIN_STRESS_NET_PROFIT_USD,
        "average_gain_vs_loss": (payoff is not None and float(payoff) >= MIN_GAIN_LOSS_RATIO),
        "avoid_single_winning_trade": concentration <= MAX_SINGLE_WIN_SHARE,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "reason_codes": [name.upper() for name, passed in checks.items() if not passed],
        "completed_cycles": cycles,
        "segment_cycle_counts": segments,
        "segment_returns": returns,
        "active_segments": active,
        "profitable_segments": positive,
        "avg_gain_loss_ratio": payoff,
        "largest_win_share_of_gross_wins": concentration,
        "base_net_profit_usd": net,
        "stress_net_profit_usd": stressed_net,
        "stress_final_capital": float(stressed["capital_final"]),
        "stress_fees_paid_usd": float(stressed["fees_paid"]),
        "stress_slippage_usd_estimate": float(stressed["slippage_usd_estimate"]),
        "stress_fee_multiplier": STRESS_FEE_MULTIPLIER,
        "stress_slippage_multiplier": STRESS_SLIPPAGE_MULTIPLIER,
        "thresholds": {
            "minimum_completed_cycles": MIN_COMPLETED_CYCLES,
            "minimum_active_segments": MIN_ACTIVE_SEGMENTS,
            "minimum_profitable_segments": MIN_PROFITABLE_SEGMENTS,
            "minimum_cycles_per_active_segment": MIN_CYCLES_PER_ACTIVE_SEGMENT,
            "minimum_gain_loss_ratio": MIN_GAIN_LOSS_RATIO,
            "maximum_single_winning_trade_share": MAX_SINGLE_WIN_SHARE,
            "minimum_stressed_net_profit_usd": MIN_STRESS_NET_PROFIT_USD,
        },
    }
