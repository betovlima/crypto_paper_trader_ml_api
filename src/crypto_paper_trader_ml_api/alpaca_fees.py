"""Alpaca Crypto spot maker/taker fee schedule and causal 30-day volume tiers.

Official schedule: https://docs.alpaca.markets/us/docs/crypto-fees
Tier recalculated daily at 03:00 America/New_York for fills from the
PREVIOUS 30-day window. Only simulated activity in this portfolio is known;
other account activity and any exchange-specific promotional fees are absent.
Orders are assumed taker unless explicitly configured maker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
import math

ALPACA_FEE_SOURCE = "https://docs.alpaca.markets/us/docs/crypto-fees"
ALPACA_FEE_MODEL = "ALPACA_CRYPTO_SPOT_2026"
ALPACA_TIERS = (
    (0, .0015, .0025),
    (100_000, .0012, .0022),
    (500_000, .0010, .0020),
    (1_000_000, .0008, .0018),
    (10_000_000, .0005, .0015),
    (25_000_000, .0002, .0013),
    (50_000_000, .0002, .0012),
    (100_000_000, .0000, .0010),
)
_ALPACA_TZ = ZoneInfo("America/New_York")


def crypto_fee_tier(volume_30d_usd: float, liquidity: str = "TAKER") -> dict:
    """Quote fees using USD trading volume; buy AND sell fills count."""
    if not math.isfinite(volume_30d_usd) or volume_30d_usd < 0:
        raise ValueError("Volume negociado nos últimos 30 dias inválido")
    if liquidity not in {"MAKER", "TAKER"}:
        raise ValueError("Tipo de ordem Alpaca inválido: use MAKER ou TAKER")
    index = 0
    for i, (minimum, _maker, _taker) in enumerate(ALPACA_TIERS):
        if volume_30d_usd >= minimum:
            index = i
    minimum, maker, taker = ALPACA_TIERS[index]
    return {
        "tier": index + 1,
        "volume_30d_usd": float(volume_30d_usd),
        "tier_minimum_usd": minimum,
        "liquidity": liquidity,
        "rate": maker if liquidity == "MAKER" else taker,
    }


def _ensure_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("Data da operação deve ter timezone")
    return value.astimezone(timezone.utc)


def _tier_cutoff(at: datetime) -> datetime:
    """Last completed daily Alpaca tier recalculation at 03:00 New York."""
    local = _ensure_utc(at).astimezone(_ALPACA_TZ)
    day = local.date()
    if local.time() < time(3):
        day -= timedelta(days=1)
    return datetime.combine(day, time(3), tzinfo=_ALPACA_TZ).astimezone(timezone.utc)


@dataclass
class AlpacaFeeLedger:
    liquidity: str = "TAKER"
    initial_30d_volume_usd: float = 0.0
    # Filled amounts only, never projected future fills.
    fills: list[tuple[datetime, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        crypto_fee_tier(self.initial_30d_volume_usd, self.liquidity)

    def quote(self, at: datetime) -> dict:
        stamp = _ensure_utc(at)
        cutoff = _tier_cutoff(stamp)
        window_start = cutoff - timedelta(days=30)
        past_volume = sum(
            amount for when, amount in self.fills
            if window_start <= when < cutoff
        )
        # If the account had external activity, it must be included in the
        # seed volume supplied by the user; the local simulation cannot know it.
        tier = crypto_fee_tier(self.initial_30d_volume_usd + past_volume, self.liquidity)
        return {**tier, "tier_cutoff_at": cutoff.isoformat()}

    def record(self, at: datetime, notional_usd: float) -> None:
        at = _ensure_utc(at)
        if not math.isfinite(notional_usd) or notional_usd > 1e12 or notional_usd < 0:
            raise ValueError("Valor negociado na Alpaca inválido")
        if self.fills and at < self.fills[-1][0]:
            raise ValueError("Operações da Alpaca devem estar em ordem temporal")
        self.fills.append((at, float(notional_usd)))


def round_trip_cost(buy_fee: float, sell_fee: float, slippage: float) -> float:
    """Gross price increase required to break even, incl. two sides and slip."""
    if any(not (0 <= v < 1) for v in (buy_fee, sell_fee, slippage)):
        raise ValueError("Custos de execução inválidos")
    return (1 + slippage) / ((1 - buy_fee) * (1 - sell_fee) * (1 - slippage)) - 1
