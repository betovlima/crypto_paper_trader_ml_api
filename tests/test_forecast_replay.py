"""The archived replay uses database-audit bytes, not local ZIP file paths."""
from __future__ import annotations

import pytest

from crypto_paper_trader_ml_api.archive_replay import ArchivedMarketClient, inspect_source_archive
from .test_archive_replay_ui import _make_archive


def test_db_audit_replay_uses_saved_30m_and_4h_candles():
    audit = _make_archive()
    request, _ = inspect_source_archive(audit)
    provider = ArchivedMarketClient(audit, request)
    assert len(provider.fetch_history(
        symbol=request.symbol, timeframe="30min", period_years=request.period_years
    )) == 70
    assert len(provider.fetch_history(
        symbol=request.symbol, timeframe="4hour", period_years=request.period_years
    )) == 9
    with pytest.raises(ValueError, match="histórico arquivado"):
        provider.fetch_history(
            symbol="ETHUSDT", timeframe="30min", period_years=request.period_years
        )


def test_zip_without_required_provenance_is_rejected():
    with pytest.raises(ValueError, match="não é um ZIP"):
        inspect_source_archive(b"not a zip archive")
