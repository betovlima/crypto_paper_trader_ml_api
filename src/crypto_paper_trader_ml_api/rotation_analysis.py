"""Retrospective BTC/CASH rotation analysis, using effective simulated positions.

A peak uses closing prices up through the displayed candle only. It is
not a trading signal at that candle's opening.
"""

from __future__ import annotations

import math


def build_rotation_timeline(rows: list[dict], maximum_points: int) -> list[dict]:
    """Sample the actual candle history while preserving every executed switch."""
    if not rows:
        return []
    if maximum_points < 1:
        raise ValueError("maximum_points must be positive")

    stride = max(1, math.ceil(len(rows) / maximum_points))
    important = {0, len(rows) - 1}
    for index, row in enumerate(rows):
        if row.get("actions") or (index > 0 and row["position"] != rows[index - 1]["position"]):
            important.add(index)

    high_water_close = float("-inf")
    timeline: list[dict] = []
    for index, row in enumerate(rows):
        close = float(row["btc_close"])
        if not math.isfinite(close) or close <= 0:
            raise ValueError("BTC close must be positive and finite")
        previous_peak = high_water_close if high_water_close > 0 else None
        high_water_close = max(high_water_close, close)
        if index % stride and index not in important:
            continue
        timeline.append(
            {
                "timestamp": row["timestamp"],
                "btc_close": close,
                "peak_close_so_far": high_water_close,
                "distance_to_peak": close / high_water_close - 1.0,
                "position": row["position"],
                "equity": float(row["equity"]),
                "buy_hold": float(row["buy_hold"]),
                "actions": [
                    {
                        **action,
                        "distance_to_previous_peak": (
                            float(action["execution_price"])
                            / (high_water_close if action.get("reason") == "END_OF_TEST" else previous_peak)
                            - 1.0
                        )
                        if (high_water_close if action.get("reason") == "END_OF_TEST" else previous_peak)
                        else None,
                    }
                    for action in row.get("actions", [])
                ],
            }
        )
    return timeline
