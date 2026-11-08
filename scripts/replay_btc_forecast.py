"""Replay BTC_DIRECTION_RETURN V2 against the exact candles of a V1 audit.

Run from the ML API repository, with the environment used for normal backtests:

    poetry run python scripts/replay_btc_forecast.py previous-audit.zip

Nothing is downloaded from the exchange. The original experiment's labels,
features, fold parameters and raw candles are preserved; only new V2
regression hyperparameters and bootstrap configuration come from the current
runtime database. The generated audit records its original ZIP SHA-256.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crypto_paper_trader_ml_api.database import initialize_database  # noqa: E402
from crypto_paper_trader_ml_api.direct_forecast import run_direction_backtest  # noqa: E402
from crypto_paper_trader_ml_api.schemas import ResolvedMLBacktestCreate  # noqa: E402
from crypto_paper_trader_ml_api.settings_service import (  # noqa: E402
    RuntimeParameters, get_runtime_parameters,
)

from compare_forecast_audits import compare_audits  # noqa: E402


def _raw(z: ZipFile, filename: str) -> pd.DataFrame:
    dataframe = pd.read_csv(
        z.open(filename), float_precision="round_trip", parse_dates=["timestamp"]
    )
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], utc=True)
    if dataframe.empty or dataframe["timestamp"].duplicated().any():
        raise ValueError(f"Missing or duplicate source timestamps in {filename}")
    return dataframe.sort_values("timestamp").reset_index(drop=True)


class VerifiedAuditMarket:
    """Serve only the user's archived data; never substitute another exchange."""

    def __init__(
        self, *, source_sha256: str, request: ResolvedMLBacktestCreate,
        decision: pd.DataFrame, trend: pd.DataFrame, manifest: dict,
    ) -> None:
        self.replay_source_sha256 = source_sha256
        self.request = request
        self.candles = {
            request.decision_timeframe: decision,
            request.trend_timeframe: trend,
        }
        self.archive_manifest = manifest.get("market_archive_checksums")
        self.market_interruptions = manifest.get("market_interruptions", {})

    def fetch_history(
        self, *, symbol: str, timeframe: str, period_years: int,
        progress=None, cancelled=None,
    ) -> pd.DataFrame:
        if cancelled and cancelled():
            raise InterruptedError("Replay cancellation requested")
        if symbol != self.request.symbol or period_years != self.request.period_years:
            raise ValueError("Replay request differs from the archived source")
        if timeframe not in self.candles:
            raise ValueError(f"Replay source does not contain {timeframe}")
        if progress is not None:
            progress(100)
        return self.candles[timeframe].copy(deep=True)

    def close(self) -> None:
        pass


def replay_audit(original: Path, output: Path) -> dict:
    initialize_database()
    with ZipFile(original) as z:
        manifest = json.loads(z.read("manifest.json"))
        if manifest.get("kind") != "BTC_DIRECTION_RETURN_OOS_AUDIT":
            raise ValueError("Only complete BTC_DIRECTION_RETURN audit ZIPs can be replayed")
        request = ResolvedMLBacktestCreate.model_validate_json(z.read("request.json"))
        if request.experiment_type != "BTC_DIRECTION_RETURN":
            raise ValueError("Audit is not a direct BTC forecast")
        if manifest.get("history_source") != request.history_source:
            raise ValueError("History source does not match audit request")
        original_params = json.loads(z.read("parameters_snapshot.json"))
        decision = _raw(z, "market_history_decision.csv")
        trend = _raw(z, "market_history_trend.csv")

    current_params = get_runtime_parameters().snapshot()
    overrides = {
        key: value
        for key, value in current_params.items()
        if key.startswith("forecast.regression_") or key == "forecast.bootstrap_samples"
    }
    parameters = RuntimeParameters({**original_params, **overrides})
    client = VerifiedAuditMarket(
        source_sha256=sha256(original.read_bytes()).hexdigest(),
        request=request, decision=decision, trend=trend,
        manifest=manifest,
    )
    written: list[bytes] = []
    result = run_direction_backtest(
        request, parameters,
        progress=lambda *_: None,
        cancelled=lambda: False,
        audit_writer=written.append,
        historical_client=client,
    )
    if len(written) != 1 or not written[0]:
        raise RuntimeError("Replay generated no complete audit ZIP")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(written[0])
    comparison = compare_audits(original, output)
    return {
        "output": str(output),
        "original_sha256": client.replay_source_sha256,
        "horizons": [row["horizon_hours"] for row in result["forecast"]["horizons"]],
        "controlled_comparison": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path, help="Original V1 Binance audit ZIP")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    original = args.original.resolve()
    output = args.output or original.with_name(original.stem + "-v2-replay.zip")
    summary = replay_audit(original, output.resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
