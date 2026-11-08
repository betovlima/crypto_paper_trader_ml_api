"""Official Binance archive parsing and historical integrity checks.

All HTTP is mocked: the regression suite never accesses real exchanges.
"""

from __future__ import annotations

import calendar
import csv
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import PurePosixPath
from zipfile import ZipFile, ZIP_DEFLATED

import httpx
import pandas as pd
import pytest

from crypto_paper_trader_ml_api.binance_archive import (
    BASE_URL,
    BinanceArchiveError,
    BinancePublicArchiveClient,
    _archive_paths,
    _parse_zip,
)
from crypto_paper_trader_ml_api.schemas import MLBacktestCreate

from .helpers import default_parameters


def _make_archive(filename, rows):
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as target:
        target.writestr(filename, "\n".join(",".join(map(str, row)) for row in rows) + "\n")
    return buffer.getvalue()


def _candle(start, *, microseconds=False, price=100.0):
    opened_ms = int(start.timestamp() * 1000)
    closed_ms = opened_ms + 24 * 3600 * 1000 - 1
    scale = 1000 if microseconds else 1
    return [
        opened_ms * scale, str(price), str(price + 1), str(price - 1), str(price),
        "13.5", closed_ms * scale, "1300.5", 20, "6", "600", "0",
    ]


def _server(*, gap_day=None, corrupt=False, missing=False):
    hits = []

    def handle(req):
        path = req.url.path
        hits.append(path)
        if missing and path.endswith("2025-10.zip"):
            return httpx.Response(404)
        target = path.removesuffix(".CHECKSUM")
        name = PurePosixPath(target).name.removesuffix(".zip")
        parts = name.split("-")
        is_daily = "/daily/" in path
        if is_daily:
            year, month, day = map(int, parts[-3:])
            start = datetime(year, month, day, tzinfo=timezone.utc)
            dates = [start]
        else:
            year, month = map(int, parts[-2:])
            dates = [
                datetime(year, month, day, tzinfo=timezone.utc)
                for day in range(1, calendar.monthrange(year, month)[1] + 1)
            ]
        if gap_day is not None:
            dates = [d for d in dates if d.date().isoformat() != gap_day]
        rows = [_candle(d, microseconds=d.year >= 2025) for d in dates]
        binary = _make_archive(name + ".csv", rows)
        digest = sha256(binary).hexdigest()
        if path.endswith(".CHECKSUM"):
            return httpx.Response(200, text=digest + "  " + PurePosixPath(target).name + "\n")
        if corrupt:
            binary = b"not a zip file"
        return httpx.Response(200, content=binary)

    return httpx.MockTransport(handle), hits


def _params(**overrides):
    return default_parameters(**{
        "market.history_warmup_candles": 0,
        "market.request_delay_seconds": 0.0,
        **overrides,
    })


def _client(transport, now=datetime(2026, 11, 8, 12, tzinfo=timezone.utc)):
    return BinancePublicArchiveClient(
        _params(), "0.test", now=now,
        client=httpx.Client(transport=transport, base_url=BASE_URL),
    )


def test_archive_parses_spot_milliseconds_and_microseconds_without_timestamp_shift():
    before = datetime(2024, 12, 31, tzinfo=timezone.utc)
    after = datetime(2025, 1, 1, tzinfo=timezone.utc)
    payload = _make_archive(
        "BTCUSDT-1d-2025-01-01.csv",
        [_candle(before), _candle(after, microseconds=True)],
    )
    values = _parse_zip(
        payload, "BTCUSDT-1d-2025-01-01.csv",
        cutoff_ms=int(datetime(2025, 1, 3, tzinfo=timezone.utc).timestamp() * 1000),
    )
    assert len(values) == 2
    assert values[0]["timestamp"] == pd.Timestamp("2024-12-31T00:00:00Z")
    assert values[1]["timestamp"] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert values[1]["close_time_ms"] == int(after.timestamp() * 1000) + 86400000 - 1


def test_four_year_paths_use_monthly_archives_then_completed_daily_files():
    start = datetime(2022, 11, 6, tzinfo=timezone.utc)
    cutoff = datetime(2026, 11, 7, tzinfo=timezone.utc)
    urls = _archive_paths("BTCUSDT", "30m", start, cutoff)
    assert urls[0].endswith("/BTCUSDT-30m-2022-11.zip")
    assert "/BTCUSDT-30m-2026-10.zip" in urls[-7]
    assert urls[-1].endswith("/BTCUSDT-30m-2026-11-06.zip")
    assert not any("2026-11-07" in item for item in urls)
    assert len(urls) == 48 + 6


def test_full_year_of_daily_candles_uses_only_verified_binance_data():
    transport, hits = _server()
    client = _client(transport)
    try:
        frame = client.fetch_history(
            symbol="BTCUSDT", timeframe="1day", period_years=1,
            progress=lambda value: None,
        )
    finally:
        client.close()
    assert len(frame) == 366
    assert frame["timestamp"].min() == pd.Timestamp("2025-11-06T00:00:00Z")
    assert frame["timestamp"].max() == pd.Timestamp("2026-11-06T00:00:00Z")
    assert (frame["timestamp"].diff().dropna() == pd.Timedelta(days=1)).all()
    assert len(client.archive_manifest["1day"]) > 12
    assert all(len(row["sha256"]) == 64 for row in client.archive_manifest["1day"])
    assert any(path.endswith(".CHECKSUM") for path in hits)
    assert any("2026-11-06" in path for path in hits)
    assert not any("mexc" in path.lower() for path in hits)


def test_missing_archive_fails_closed_without_cross_exchange_fallback():
    transport, _ = _server(missing=True)
    client = _client(transport, now=datetime(2025, 11, 8, tzinfo=timezone.utc))
    try:
        with pytest.raises(BinanceArchiveError, match="Archive or SHA256 checksum unavailable"):
            client.fetch_history(symbol="BTCUSDT", timeframe="1day", period_years=1)
    finally:
        client.close()


def test_checksum_mismatch_or_corrupted_archive_is_rejected():
    transport, _ = _server(corrupt=True)
    client = _client(transport)
    try:
        with pytest.raises(BinanceArchiveError, match="SHA256 mismatch"):
            client.fetch_history(symbol="BTCUSDT", timeframe="1day", period_years=1)
    finally:
        client.close()


def test_missing_interior_candle_is_not_interpolated():
    transport, _ = _server(gap_day="2026-06-14")
    client = _client(transport)
    try:
        with pytest.raises(BinanceArchiveError, match="missing or irregular"):
            client.fetch_history(symbol="BTCUSDT", timeframe="1day", period_years=1)
    finally:
        client.close()


def test_request_rejects_unsupported_symbol_or_candle_limit():
    transport, _ = _server()
    client = _client(transport)
    try:
        with pytest.raises(BinanceArchiveError, match="only for BTCUSDT"):
            client.fetch_history(symbol="ETHUSDT", timeframe="1day", period_years=1)
    finally:
        client.close()
    constrained = BinancePublicArchiveClient(
        _params(**{"market.max_candles": 200}), "0.test",
        now=datetime(2026, 11, 8, tzinfo=timezone.utc),
        client=httpx.Client(transport=transport, base_url=BASE_URL),
    )
    try:
        with pytest.raises(BinanceArchiveError, match="market.max_candles"):
            constrained.fetch_history(symbol="BTCUSDT", timeframe="1day", period_years=1)
    finally:
        constrained.close()


def test_source_is_explicit_and_rejects_unknown_values():
    assert MLBacktestCreate(history_source="MEXC").history_source == "MEXC"
    assert MLBacktestCreate(history_source="BINANCE_SPOT_ARCHIVE").history_source == "BINANCE_SPOT_ARCHIVE"
    with pytest.raises(ValueError):
        MLBacktestCreate(history_source="MEXC_PLUS_BINANCE")


def test_explicit_history_source_is_persisted_in_resolved_job_configuration():
    from crypto_paper_trader_ml_api.service import _resolve_request
    chosen = _resolve_request(
        MLBacktestCreate(symbol="BTCUSDT", history_source="BINANCE_SPOT_ARCHIVE"),
        default_parameters(),
    )
    assert chosen.history_source == "BINANCE_SPOT_ARCHIVE"
    assert chosen.model_dump()["history_source"] == "BINANCE_SPOT_ARCHIVE"
    legacy = _resolve_request(MLBacktestCreate(symbol="BTCUSDT"), default_parameters())
    assert legacy.history_source == "MEXC"
