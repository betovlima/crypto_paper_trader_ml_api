from __future__ import annotations

from datetime import datetime, timezone

import pytest

from crypto_paper_trader_ml_api.mexc import MEXCDataError, MEXCHistoryClient

from .helpers import default_parameters


class FakeResponse:
    def __init__(self, payload):  # type: ignore[no-untyped-def]
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):  # type: ignore[no-untyped-def]
        return self._payload


def aligned_closed_payload(*, interval_ms: int, count: int) -> list[list]:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    current_open_ms = (now_ms // interval_ms) * interval_ms
    first_open_ms = current_open_ms - count * interval_ms
    rows = []
    for index in range(count):
        open_ms = first_open_ms + index * interval_ms
        close_ms = open_ms + interval_ms - 1
        rows.append([open_ms, "100", "101", "99", "100.5", "10", close_ms, "1005"])
    return rows


def test_fetch_history_uses_explicit_non_overlapping_windows_when_server_caps_at_500() -> None:
    parameters = default_parameters(
        **{
            "market.minimum_raw_candles": 500,
            "market.history_warmup_candles": 0,
            "market.history_fetch_margin_candles": 0,
            "market.max_candles": 1200,
            "market.page_limit": 1000,
            "market.page_window_candles": 500,
            "market.request_delay_seconds": 0.0,
        }
    )
    client = MEXCHistoryClient(parameters, "test")
    interval_ms = 30 * 60 * 1000
    history = aligned_closed_payload(interval_ms=interval_ms, count=1200)
    calls: list[dict] = []

    def fake_get(_path, params):  # type: ignore[no-untyped-def]
        calls.append(dict(params))
        start_ms = int(params["startTime"])
        end_ms = int(params["endTime"])
        matching = [row for row in history if start_ms <= int(row[0]) <= end_ms]
        return FakeResponse(matching[:500])

    client.client.get = fake_get  # type: ignore[method-assign]
    try:
        result = client.fetch_history(symbol="BTCUSDT", timeframe="30min", period_years=1)
    finally:
        client.close()

    assert len(result) == 1200
    assert len(calls) == 3
    assert all("startTime" in call and "endTime" in call for call in calls)
    assert calls[0]["endTime"] < calls[1]["startTime"]
    assert calls[1]["endTime"] < calls[2]["startTime"]


def test_fetch_history_continues_across_empty_windows_before_listing() -> None:
    parameters = default_parameters(
        **{
            "market.minimum_raw_candles": 500,
            "market.history_warmup_candles": 0,
            "market.history_fetch_margin_candles": 0,
            "market.max_candles": 1200,
            "market.page_limit": 1000,
            "market.page_window_candles": 500,
            "market.request_delay_seconds": 0.0,
        }
    )
    client = MEXCHistoryClient(parameters, "test")
    interval_ms = 30 * 60 * 1000
    history = aligned_closed_payload(interval_ms=interval_ms, count=600)
    calls: list[dict] = []

    def fake_get(_path, params):  # type: ignore[no-untyped-def]
        calls.append(dict(params))
        start_ms = int(params["startTime"])
        end_ms = int(params["endTime"])
        matching = [row for row in history if start_ms <= int(row[0]) <= end_ms]
        return FakeResponse(matching[:500])

    client.client.get = fake_get  # type: ignore[method-assign]
    try:
        result = client.fetch_history(symbol="NEWUSDT", timeframe="30min", period_years=1)
    finally:
        client.close()

    assert len(calls) == 3
    assert len(result) == 600


def test_fetch_history_reports_diagnostics_when_minimum_is_not_reached() -> None:
    parameters = default_parameters(
        **{
            "market.minimum_raw_candles": 500,
            "market.history_warmup_candles": 0,
            "market.history_fetch_margin_candles": 0,
            "market.max_candles": 1200,
            "market.page_limit": 1000,
            "market.page_window_candles": 500,
            "market.request_delay_seconds": 0.0,
        }
    )
    client = MEXCHistoryClient(parameters, "test")
    interval_ms = 30 * 60 * 1000
    history = aligned_closed_payload(interval_ms=interval_ms, count=499)

    def fake_get(_path, params):  # type: ignore[no-untyped-def]
        start_ms = int(params["startTime"])
        end_ms = int(params["endTime"])
        matching = [row for row in history if start_ms <= int(row[0]) <= end_ms]
        return FakeResponse(matching[:500])

    client.client.get = fake_get  # type: ignore[method-assign]
    try:
        with pytest.raises(MEXCDataError) as exc_info:
            client.fetch_history(symbol="SHORTUSDT", timeframe="30min", period_years=1)
    finally:
        client.close()

    message = str(exc_info.value)
    assert "499 unique closed candles" in message
    assert "explicit startTime/endTime requests" in message
