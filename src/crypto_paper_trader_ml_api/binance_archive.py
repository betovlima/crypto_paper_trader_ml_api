"""Official Binance SPOT public-data archives for independent BTC research.

Never mix these candles with MEXC candles in one backtest. Binance archives
represent another exchange, not missing MEXC history. A run must explicitly
request this source and its provenance is attached to the results.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO, TextIOWrapper
import math
import re
import time
from typing import Callable
from zipfile import BadZipFile, ZipFile

import httpx
import pandas as pd

from .mexc import TIMEFRAME_SECONDS
from .settings_service import RuntimeParameters


SOURCE = "BINANCE_SPOT_ARCHIVE"
BASE_URL = "https://data.binance.vision"
INTERVALS = {
    "15min": "15m",
    "30min": "30m",
    "1hour": "1h",
    "4hour": "4h",
    "1day": "1d",
    "1week": "1w",
}
KLINE_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time_ms", "quote_volume",
]
MAX_COMPRESSED_BYTES = 16 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 96 * 1024 * 1024
MAX_DAILY_REPAIR_SLOTS = 12

# Binance confirmed a Spot-wide trading interruption on 2023-03-24,
# resuming at 14:00 UTC. These two 30-minute candle slots were absent
# from BOTH the official monthly and daily BTCUSDT Spot archives.
# This allowlist is deliberately narrow; all other gaps remain fatal.
DOCUMENTED_BTCUSDT_SPOT_30M_HALT = (
    "2023-03-24T13:00:00+00:00",
    "2023-03-24T13:30:00+00:00",
)
BINANCE_HALT_20230324_SOURCE = (
    "https://www.binance.com/en/support/announcement/detail/"
    "813a31506e9f478ea8c1058b425df87a"
)


def _known_spot_halt(
    *, symbol: str, timeframe: str,
    missing: pd.DatetimeIndex, unexpected: pd.DatetimeIndex,
) -> list[dict]:
    """Accept only the exact independently documented market closure."""
    expected_halt = pd.DatetimeIndex(
        [pd.Timestamp(ts) for ts in DOCUMENTED_BTCUSDT_SPOT_30M_HALT]
    )
    if (
        symbol == "BTCUSDT"
        and timeframe == "30min"
        and len(unexpected) == 0
        and len(missing) == len(expected_halt)
        and missing.equals(expected_halt)
    ):
        return [{
            "event": "BINANCE_SPOT_TRADING_HALT",
            "date": "2023-03-24",
            "missing_candle_openings_utc": [
                stamp.isoformat() for stamp in missing
            ],
            "missing_candles": len(missing),
            "source_url": BINANCE_HALT_20230324_SOURCE,
            "market_status": "SPOT_TRADING_SUSPENDED",
            "trading_resumed_at_utc": "2023-03-24T14:00:00+00:00",
        }]
    return []



class BinanceArchiveError(ValueError):
    """Missing, corrupted, inconsistent, or incomplete historical source."""


def _month_starts(start: datetime, end: datetime):
    cursor = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cursor < end:
        yield cursor
        if cursor.month == 12:
            cursor = cursor.replace(year=cursor.year + 1, month=1)
        else:
            cursor = cursor.replace(month=cursor.month + 1)


def _archive_paths(symbol: str, interval: str, start: datetime, end: datetime) -> list[str]:
    """Monthly archives for past months, daily archives for the last month."""
    result: list[str] = []
    last_month = end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    for month in _month_starts(start, last_month):
        stamp = f"{month.year:04d}-{month.month:02d}"
        result.append(
            f"/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{stamp}.zip"
        )
    day = max(start, last_month).replace(hour=0, minute=0, second=0, microsecond=0)
    while day < end:
        stamp = day.strftime("%Y-%m-%d")
        result.append(
            f"/data/spot/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{stamp}.zip"
        )
        day += timedelta(days=1)
    return result


def _parse_zip(payload: bytes, expected_name: str, *, cutoff_ms: int) -> list[dict]:
    if len(payload) > MAX_COMPRESSED_BYTES:
        raise BinanceArchiveError(f"Archive exceeds compressed-size limit: {expected_name}")
    try:
        with ZipFile(BytesIO(payload)) as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].filename != expected_name:
                raise BinanceArchiveError(f"Unexpected files in archive {expected_name}")
            if members[0].file_size > MAX_UNCOMPRESSED_BYTES:
                raise BinanceArchiveError(f"Archive exceeds decompressed-size limit: {expected_name}")
            with archive.open(members[0]) as raw_stream:
                reader = csv.reader(TextIOWrapper(raw_stream, encoding="utf-8-sig"))
                output: list[dict] = []
                for record in reader:
                    if not record or record[0].strip().lower() in {"open_time", "open time"}:
                        continue
                    if len(record) < 8:
                        raise BinanceArchiveError(f"Incomplete kline in {expected_name}")
                    try:
                        raw_open = int(record[0])
                        raw_close = int(record[6])
                        # SPOT archives changed from milliseconds to microseconds
                        # on 2025-01-01; the field magnitude identifies the unit.
                        opening_ms = raw_open // 1000 if raw_open >= 10**15 else raw_open
                        closing_ms = raw_close // 1000 if raw_close >= 10**15 else raw_close
                        if closing_ms >= cutoff_ms:
                            continue
                        values = [float(record[i]) for i in (1, 2, 3, 4, 5, 7)]
                    except (ValueError, OverflowError, TypeError) as exc:
                        raise BinanceArchiveError(f"Invalid candle in {expected_name}") from exc
                    if not all(math.isfinite(value) for value in values):
                        raise BinanceArchiveError(f"Non-finite candle in {expected_name}")
                    if min(values[:4]) <= 0 or values[4] < 0 or values[5] < 0:
                        raise BinanceArchiveError(f"Invalid OHLCV in {expected_name}")
                    opening, high, low, closing, volume, quote_volume = values
                    if not low <= min(opening, closing) <= max(opening, closing) <= high:
                        raise BinanceArchiveError(f"Inconsistent OHLC in {expected_name}")
                    output.append({
                        "timestamp": pd.to_datetime(opening_ms, unit="ms", utc=True),
                        "open": opening,
                        "high": high,
                        "low": low,
                        "close": closing,
                        "volume": volume,
                        "close_time_ms": closing_ms,
                        "quote_volume": quote_volume,
                    })
                return output
    except (BadZipFile, EOFError) as exc:
        raise BinanceArchiveError(f"Corrupted ZIP: {expected_name}") from exc



def _filter_closed_window(
    frame: pd.DataFrame,
    *,
    start_ms: int,
    cutoff_ms: int,
    maximum_rows: int,
) -> pd.DataFrame:
    """Keep closed UTC candles without assuming pandas timestamp precision.

    pandas can store datetime64[ns] or datetime64[us]. Converting its
    timestamps to int64 and dividing by 1_000_000 is only correct for ns;
    direct timezone-aware comparison works for both.
    """
    start_at = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    return frame.loc[
        (frame["timestamp"] >= start_at)
        & (frame["close_time_ms"] < cutoff_ms)
    ].tail(maximum_rows).reset_index(drop=True)


class BinancePublicArchiveClient:
    """Read checksum-verified monthly/daily spot archives with SHA256 checks."""

    def __init__(
        self,
        parameters: RuntimeParameters,
        app_version: str,
        *,
        now: datetime | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.parameters = parameters
        self.now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        self.client = client or httpx.Client(
            base_url=BASE_URL,
            timeout=parameters.floating("market.http_timeout_seconds"),
            headers={"User-Agent": f"crypto-paper-trader-ml/{app_version}"},
        )
        self.archive_manifest: dict[str, list[dict]] = {}
        self.market_interruptions: dict[str, list[dict]] = {}

    def close(self) -> None:
        self.client.close()

    def _read_verified_archive(
        self, path: str, *, cutoff_ms: int
    ) -> tuple[list[dict], dict]:
        """Use the same SHA256 verification for monthly and daily ZIPs."""
        try:
            response = self.client.get(path)
            response.raise_for_status()
            checksum_response = self.client.get(path + ".CHECKSUM")
            checksum_response.raise_for_status()
        except httpx.HTTPError as exc:
            raise BinanceArchiveError(
                f"Archive or SHA256 checksum unavailable: {BASE_URL}{path}"
            ) from exc

        checksum_text = checksum_response.text.strip().split()
        if not checksum_text or not re.fullmatch(r"[0-9a-fA-F]{64}", checksum_text[0]):
            raise BinanceArchiveError(f"Malformed checksum for {BASE_URL}{path}")
        expected_hash = checksum_text[0].lower()
        downloaded = response.content
        actual_hash = sha256(downloaded).hexdigest()
        if actual_hash != expected_hash:
            raise BinanceArchiveError(f"SHA256 mismatch for {BASE_URL}{path}")

        rows = _parse_zip(
            downloaded, path.rsplit("/", 1)[-1].removesuffix(".zip") + ".csv",
            cutoff_ms=cutoff_ms,
        )
        return rows, {
            "url": BASE_URL + path,
            "sha256": actual_hash,
            "rows": len(rows),
        }

    def _repair_small_gaps_from_daily(
        self,
        *,
        symbol: str,
        interval: str,
        frame: pd.DataFrame,
        expected: pd.DatetimeIndex,
        cutoff_ms: int,
        cancelled: Callable[[], bool] | None,
    ) -> tuple[pd.DataFrame, list[dict], list[str]]:
        """Repair only documented missing timestamps from verified Binance daily files.

        Never create candles, borrow MEXC prices, or treat a missing file as
        evidence that trading occurred. If verified daily history also lacks
        a candle, return the gap so the backtest can fail explicitly.
        """
        present = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
        missing = expected.difference(present)
        if missing.empty:
            return frame, [], []
        if len(missing) > MAX_DAILY_REPAIR_SLOTS:
            return frame, [], [
                f"Skipped daily recovery: {len(missing)} absent timestamps "
                f"exceed safety limit {MAX_DAILY_REPAIR_SLOTS}"
            ]

        verified: list[dict] = []
        attempts: list[str] = []
        missing_days = sorted({timestamp.strftime("%Y-%m-%d") for timestamp in missing})
        retrieved: list[dict] = []
        missing_set = set(missing)
        for date in missing_days:
            if cancelled and cancelled():
                raise InterruptedError("Backtest cancellation requested")
            path = (
                f"/data/spot/daily/klines/{symbol}/{interval}/"
                f"{symbol}-{interval}-{date}.zip"
            )
            try:
                rows, metadata = self._read_verified_archive(
                    path, cutoff_ms=cutoff_ms
                )
            except BinanceArchiveError as exc:
                attempts.append(str(exc))
                continue
            selected = [
                row for row in rows
                if row["timestamp"] in missing_set
            ]
            retrieved.extend(selected)
            verified.append({
                **metadata,
                "role": "VERIFIED_DAILY_GAP_REPAIR",
                "recovered_candles": len(selected),
            })
            attempts.append(
                f"{date}: {len(selected)} missing candles found in verified daily archive"
            )

        if retrieved:
            frame = (
                pd.concat([frame, pd.DataFrame.from_records(retrieved)], ignore_index=True)
                .drop_duplicates(subset=["timestamp"], keep="first")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
        return frame, verified, attempts

    def fetch_history(
        self, *, symbol: str, timeframe: str, period_years: int,
        progress: Callable[[int], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> pd.DataFrame:
        if symbol != "BTCUSDT":
            raise BinanceArchiveError("Binance archive research is enabled only for BTCUSDT")
        if timeframe not in INTERVALS:
            raise BinanceArchiveError(f"Unsupported Binance archive timeframe: {timeframe}")
        if period_years < 1:
            raise BinanceArchiveError("period_years must be positive")

        interval_seconds = TIMEFRAME_SECONDS[timeframe]
        interval_ms = interval_seconds * 1000
        requested = math.ceil(period_years * 365.25 * 86400 / interval_seconds)
        required_with_warmup = requested + self.parameters.integer("market.history_warmup_candles")
        max_candles = self.parameters.integer("market.max_candles")
        if required_with_warmup > max_candles:
            raise BinanceArchiveError(
                f"Requested {required_with_warmup} {timeframe} candles exceeds "
                f"the database limit market.max_candles={max_candles}"
            )

        # Archive daily bundles may lag publication. Research closes two UTC
        # days before the current day; never combine recent MEXC candles.
        midnight_today = self.now.replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff = midnight_today - timedelta(days=1)
        start = cutoff - timedelta(milliseconds=required_with_warmup * interval_ms)
        paths = _archive_paths(symbol, INTERVALS[timeframe], start, cutoff)
        cutoff_ms = int(cutoff.timestamp() * 1000)
        start_ms = int(start.timestamp() * 1000)
        records: list[dict] = []
        provenance: list[dict] = []

        for index, path in enumerate(paths):
            if cancelled and cancelled():
                raise InterruptedError("Backtest cancellation requested")
            rows, metadata = self._read_verified_archive(
                path, cutoff_ms=cutoff_ms
            )
            records.extend(rows)
            provenance.append(metadata)
            if progress:
                progress(int((index + 1) / max(1, len(paths)) * 100))
            delay = self.parameters.floating("market.request_delay_seconds")
            if delay > 0:
                time.sleep(delay)

        if not records:
            raise BinanceArchiveError("No Binance archive candles found for the requested period")
        frame = (
            pd.DataFrame.from_records(records, columns=KLINE_COLUMNS)
            .drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        frame = _filter_closed_window(
            frame,
            start_ms=start_ms,
            cutoff_ms=cutoff_ms,
            maximum_rows=required_with_warmup,
        )
        expected = pd.date_range(
            start=pd.Timestamp(start_ms, unit="ms", tz="UTC"),
            periods=required_with_warmup,
            freq=pd.Timedelta(milliseconds=interval_ms),
        )
        frame, supplemental, repair_attempts = self._repair_small_gaps_from_daily(
            symbol=symbol,
            interval=INTERVALS[timeframe],
            frame=frame,
            expected=expected,
            cutoff_ms=cutoff_ms,
            cancelled=cancelled,
        )
        provenance.extend(supplemental)
        # Do not silently shorten the research period or overlook gaps.
        present = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
        missing = expected.difference(present)
        unexpected = present.difference(expected)
        allowed_halts = _known_spot_halt(
            symbol=symbol, timeframe=timeframe,
            missing=missing, unexpected=unexpected,
        )
        # Do not waive a gap when the daily file could not be checked.
        # The official event alone is not evidence that THIS downloaded
        # dataset was complete outside the suspension.
        if allowed_halts and not any(
            attempt.startswith("2023-03-24: 0 missing candles found in verified daily archive")
            for attempt in repair_attempts
        ):
            allowed_halts = []
        self.market_interruptions[timeframe] = allowed_halts
        if (
            (len(missing) or len(unexpected) or len(frame) != required_with_warmup)
            and not (
                allowed_halts
                and len(frame) == required_with_warmup - len(missing)
            )
        ):
            missing_details = ", ".join(
                timestamp.isoformat() for timestamp in missing[:12]
            ) or "none"
            unexpected_details = ", ".join(
                timestamp.isoformat() for timestamp in unexpected[:5]
            ) or "none"
            repair_details = "; ".join(repair_attempts[:12]) or "no daily recovery needed"
            raise BinanceArchiveError(
                f"Binance archive has {len(missing)} missing or irregular "
                f"{timeframe} candle intervals; requested {required_with_warmup}, "
                f"recovered {len(frame)}. Missing UTC openings: {missing_details}. "
                f"Unexpected UTC openings: {unexpected_details}. "
                f"Verified daily recovery: {repair_details}. "
                "No synthetic candles or cross-exchange substitution were used."
            )
        diffs = frame["timestamp"].diff().dropna().dt.total_seconds()
        if not diffs.eq(interval_seconds).all() and not allowed_halts:
            bad = int((diffs != interval_seconds).sum())
            raise BinanceArchiveError(
                f"Binance archive has {bad} missing or irregular {timeframe} candle intervals; "
                "refusing to interpolate research data"
            )
        self.archive_manifest[timeframe] = provenance
        return frame
