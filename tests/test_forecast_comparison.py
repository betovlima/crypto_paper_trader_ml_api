"""Guard against false V1/V2 gains when the archived market changes."""
from __future__ import annotations

import pytest

from crypto_paper_trader_ml_api.archive_replay import compare_archives
from .test_archive_replay_ui import _make_archive


def test_matching_oos_archive_produces_comparison():
    report = compare_archives(_make_archive(predicted=0.005), _make_archive(predicted=0.008))
    assert report["status"] == "VERIFIED"
    assert report["horizons"][0]["rows"] == 1


def test_different_source_candles_reject_comparison():
    with pytest.raises(ValueError, match="Candles históricos diferentes"):
        compare_archives(_make_archive(), _make_archive(market_suffix="changed"))
