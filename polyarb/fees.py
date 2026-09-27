"""Taker fee models for both venues.

Both venues charge a fee proportional to p * (1 - p): largest at 50c, vanishing near 0 and 1.

Kalshi:     fee = ceil_to_cent(base * multiplier * C * p * (1 - p))
            base = 0.07; `fee_multiplier` comes from the market's series (e.g. 0.5 for some sports/index series).
            Rounded up to the next cent per order.

Polymarket: fee = C * rate * (p * (1 - p)) ** exponent, rounded to 5 decimals (USDC).
            `rate` comes from the market's feeSchedule (0 for geopolitics and some sports lines,
            0.03-0.07 elsewhere). Taker-only.

A "fill" is (price, contracts). Fees are computed on the whole order (all fills on one venue) because
Kalshi rounds per order, which makes small orders proportionally more expensive.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

Fill = tuple[float, float]


def kalshi_fee(fills: Iterable[Fill], multiplier: float = 1.0, base: float = 0.07) -> float:
    raw = sum(base * multiplier * qty * p * (1 - p) for p, qty in fills)
    if raw <= 0:
        return 0.0
    # Subtract a hair before ceiling so float noise like 0.0700000001 doesn't round up a full cent.
    return math.ceil(round(raw * 100, 9) - 1e-9) / 100


def polymarket_fee(fills: Iterable[Fill], rate: float, exponent: float = 1.0) -> float:
    if rate <= 0:
        return 0.0
    raw = sum(qty * rate * (p * (1 - p)) ** exponent for p, qty in fills)
    return round(raw, 5)


def kalshi_fee_per_contract(p: float, multiplier: float = 1.0, base: float = 0.07) -> float:
    """Unrounded marginal fee for one contract; used for top-of-book screening."""
    return base * multiplier * p * (1 - p)


def polymarket_fee_per_contract(p: float, rate: float, exponent: float = 1.0) -> float:
    return rate * (p * (1 - p)) ** exponent if rate > 0 else 0.0
