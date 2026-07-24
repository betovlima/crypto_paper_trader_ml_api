from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Callable

import httpx
import pandas as pd

from .settings_service import RuntimeParameters


logger = logging.getLogger(__name__)

TIMEFRAME_SECONDS = {
    "15min": 900,
    "30min": 1800,
    "1hour": 3600,
    "4hour": 14400,
    "1day": 86400,
    "1week": 604800,
}

MEXC_INTERVALS = {
    "15min": "15m",
    "30min": "30m",
    "1hour": "60m",
    "4hour": "4h",
    "1day": "1d",
    "1week": "1W",
}


class MEXCDataError(RuntimeError):
    pass


class MEXCHistoryClient:
    def __init__(self, parameters: RuntimeParameters, app_version: str) -> None:
        self.parameters = parameters
        self.client = httpx.Client(
            base_url=parameters.string("market.mexc_base_url").rstrip("/"),
            timeout=parameters.floating("market.http_timeout_seconds"),
            headers={"User-Agent": f"crypto-paper-trader-ml/{app_version}"},
        )

    def close(self) -> None:
        self.client.close()

    @staticmethod
    def _closed_history_bounds(*, now_ms: int, interval_ms: int, candle_count: int) -> tuple[int, int]:
        """Return a UTC-aligned range containing only fully closed candles."""
        current_open_ms = (now_ms // interval_ms) * interval_ms
        latest_closed_open_ms = current_open_ms - interval_ms
        latest_closed_end_ms = current_open_ms - 1
        earliest_open_ms = max(0, latest_closed_open_ms - max(0, candle_count - 1) * interval_ms)
        return earliest_open_ms, latest_closed_end_ms

    @staticmethod
    def _parse_payload(
        payload: object,
        *,
        now_ms: int,
        window_start_ms: int,
        window_end_ms: int,
    ) -> pd.DataFrame:
        if not isinstance(payload, list) or not payload:
            return pd.DataFrame()

        rows: list[dict[str, object]] = []
        for item in payload:
            if not isinstance(item, list) or len(item) < 8:
                continue
            timestamp_ms = int(item[0])
            if timestamp_ms < 10_000_000_000:
                timestamp_ms *= 1000
            close_time_ms = int(item[6])
            if close_time_ms < 10_000_000_000:
                close_time_ms *= 1000

            if timestamp_ms < window_start_ms or timestamp_ms > window_end_ms:
                continue
            if close_time_ms >= now_ms:
                continue

            rows.append(
                {
                    "timestamp": pd.to_datetime(timestamp_ms, unit="ms", utc=True),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                    "close_time_ms": close_time_ms,
                    "quote_volume": float(item[7]),
                }
            )

        if not rows:
            return pd.DataFrame()
        return (
            pd.DataFrame(rows)
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    def fetch_history(
        self,
        *,
        symbol: str,
        timeframe: str,
        period_years: int,
        progress: Callable[[int], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> pd.DataFrame:
        if timeframe not in MEXC_INTERVALS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        seconds = TIMEFRAME_SECONDS[timeframe]
        interval_ms = seconds * 1000
        requested_period = math.ceil(period_years * 365.25 * 86400 / seconds)
        warmup = self.parameters.integer("market.history_warmup_candles")
        margin = self.parameters.integer("market.history_fetch_margin_candles")
        maximum = self.parameters.integer("market.max_candles")
        minimum = self.parameters.integer("market.minimum_raw_candles")
        requested = min(max(minimum, requested_period + warmup), maximum)
        fetch_target = min(maximum, requested + margin)
        page_limit = self.parameters.integer("market.page_limit")
        window_candles = self.parameters.integer("market.page_window_candles")
        delay = self.parameters.floating("market.request_delay_seconds")

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        range_start_ms, range_end_ms = self._closed_history_bounds(
            now_ms=now_ms,
            interval_ms=interval_ms,
            candle_count=fetch_target,
        )
        window_size_ms = window_candles * interval_ms
        cursor_start_ms = range_start_ms
        frames: list[pd.DataFrame] = []
        request_count = 0
        raw_row_count = 0

        # MEXC can return only 500 rows even when limit=1000, and endTime-only
        # pagination is not reliable across deployments. Explicit, non-overlapping
        # startTime/endTime windows make the history traversal deterministic.
        while cursor_start_ms <= range_end_ms:
            if cancelled and cancelled():
                raise InterruptedError("Backtest cancellation requested")

            window_end_ms = min(range_end_ms, cursor_start_ms + window_size_ms - 1)
            params: dict[str, object] = {
                "symbol": symbol,
                "interval": MEXC_INTERVALS[timeframe],
                "startTime": cursor_start_ms,
                "endTime": window_end_ms,
                "limit": page_limit,
            }

            response = self.client.get("/api/v3/klines", params=params)
            response.raise_for_status()
            request_count += 1
            payload = response.json()
            if isinstance(payload, list):
                raw_row_count += len(payload)

            batch = self._parse_payload(
                payload,
                now_ms=now_ms,
                window_start_ms=cursor_start_ms,
                window_end_ms=window_end_ms,
            )
            if not batch.empty:
                frames.append(batch)

            cursor_start_ms = window_end_ms + 1
            if progress:
                covered = max(0, min(range_end_ms, cursor_start_ms) - range_start_ms)
                total = max(1, range_end_ms - range_start_ms)
                progress(min(100, int(100 * covered / total)))
            if delay > 0:
                time.sleep(delay)

        if not frames:
            raise MEXCDataError(
                f"No closed candles were returned for {symbol} ({timeframe}) after "
                f"{request_count} explicit history-window requests."
            )

        all_rows = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        result = all_rows.tail(requested).reset_index(drop=True)
        logger.info(
            "Loaded %s unique closed candles for %s (%s) from %s raw rows across %s requests.",
            len(result),
            symbol,
            timeframe,
            raw_row_count,
            request_count,
        )
        if len(result) < minimum:
            first_timestamp = (
                pd.Timestamp(all_rows.iloc[0]["timestamp"]).isoformat() if not all_rows.empty else "n/a"
            )
            last_timestamp = (
                pd.Timestamp(all_rows.iloc[-1]["timestamp"]).isoformat() if not all_rows.empty else "n/a"
            )
            raise MEXCDataError(
                f"Only {len(result)} unique closed candles are available for {symbol} ({timeframe}); "
                f"the database parameter market.minimum_raw_candles requires {minimum}. "
                f"The ML API made {request_count} explicit startTime/endTime requests and received "
                f"{raw_row_count} raw rows. Available range: {first_timestamp} to {last_timestamp}."
            )
        return result
