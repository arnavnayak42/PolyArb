"""Cross-venue arbitrage math.

A matched pair (same proposition on both venues) is an arbitrage when buying YES on one venue and NO on
the other costs less than the guaranteed $1 payout, after fees. Exactly one leg pays $1 at resolution, so
the payoff is locked in regardless of outcome (assuming both venues resolve the question identically,
which is what the matcher's resolution-confidence score is about).

We walk both ask ladders simultaneously: the k-th contract pair costs (poly ask + kalshi ask) at whatever
depth the k-th contract reaches on each book. Profit(size) is concave-ish and piecewise linear between
book breakpoints (Kalshi's per-order cent rounding adds small sawtooth), so the profit-maximizing size is
found by evaluating every breakpoint in the gross-positive region.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from polyarb.fees import (
    Fill,
    kalshi_fee,
    kalshi_fee_per_contract,
    polymarket_fee,
    polymarket_fee_per_contract,
)
from polyarb.models import BinaryBook, Level

MAX_CURVE_POINTS = 60


@dataclass(frozen=True)
class FeeModel:
    poly_rate: float = 0.0
    poly_exponent: float = 1.0
    kalshi_multiplier: float = 1.0
    kalshi_base: float = 0.07

    def poly(self, fills: list[Fill]) -> float:
        return polymarket_fee(fills, self.poly_rate, self.poly_exponent)

    def kalshi(self, fills: list[Fill]) -> float:
        return kalshi_fee(fills, self.kalshi_multiplier, self.kalshi_base)

    def per_contract(self, poly_price: float, kalshi_price: float) -> float:
        return polymarket_fee_per_contract(poly_price, self.poly_rate, self.poly_exponent) + kalshi_fee_per_contract(
            kalshi_price, self.kalshi_multiplier, self.kalshi_base
        )


@dataclass
class ArbResult:
    poly_side: str
    kalshi_side: str
    poly_price: float | None
    kalshi_price: float | None
    gross_cost_top: float | None = None
    gross_return_top: float | None = None
    net_return_top: float | None = None
    book_walked: bool = False
    size: int = 0
    capital: float = 0.0
    profit: float = 0.0
    net_return: float | None = None
    days_to_resolution: float | None = None
    annualized_return: float | None = None
    excess_annualized_return: float | None = None
    curve: list[dict] = field(default_factory=list)

    @property
    def direction(self) -> str:
        return f"PM {self.poly_side} + K {self.kalshi_side}"

    @property
    def is_gross_arb(self) -> bool:
        return self.gross_cost_top is not None and self.gross_cost_top < 1

    @property
    def is_net_arb(self) -> bool:
        return self.size > 0 and self.profit > 0


def directions(inverted: bool) -> list[tuple[str, str]]:
    """(poly side, kalshi side) combos that together pay exactly $1.

    Normally YES on one venue is hedged by NO on the other. If the two venues phrase the question with
    opposite polarity (Polymarket "above X" vs Kalshi "below X"), YES+YES is the complementary pair.
    """
    if inverted:
        return [("YES", "YES"), ("NO", "NO")]
    return [("YES", "NO"), ("NO", "YES")]


def fills_for(asks: list[Level], qty: float) -> tuple[list[Fill], float] | None:
    """Fills and total cost for buying `qty` contracts off a best-first ask ladder; None if too thin."""
    fills: list[Fill] = []
    remaining = qty
    cost = 0.0
    for lv in asks:
        if remaining <= 1e-12:
            break
        take = min(lv.size, remaining)
        fills.append((lv.price, take))
        cost += take * lv.price
        remaining -= take
    if remaining > 1e-9:
        return None
    return fills, cost


def gross_positive_breakpoints(a: list[Level], b: list[Level]) -> list[float]:
    """Cumulative sizes at which either ladder moves to a new level, while price_a + price_b < 1."""
    i = j = 0
    rem_a = a[0].size if a else 0.0
    rem_b = b[0].size if b else 0.0
    total = 0.0
    points: list[float] = []
    while i < len(a) and j < len(b) and a[i].price + b[j].price < 1:
        step = min(rem_a, rem_b)
        total += step
        points.append(total)
        rem_a -= step
        rem_b -= step
        if rem_a <= 1e-12:
            i += 1
            rem_a = a[i].size if i < len(a) else 0.0
        if rem_b <= 1e-12:
            j += 1
            rem_b = b[j].size if j < len(b) else 0.0
    return points


def evaluate_size(poly_asks: list[Level], kalshi_asks: list[Level], qty: int, fees: FeeModel) -> dict | None:
    poly = fills_for(poly_asks, qty)
    kal = fills_for(kalshi_asks, qty)
    if poly is None or kal is None:
        return None
    poly_fills, poly_cost = poly
    kal_fills, kal_cost = kal
    poly_fee = fees.poly(poly_fills)
    kal_fee = fees.kalshi(kal_fills)
    capital = poly_cost + kal_cost + poly_fee + kal_fee
    profit = qty - capital
    return {
        "size": qty,
        "capital": round(capital, 6),
        "profit": round(profit, 6),
        "return": profit / capital if capital > 0 else None,
        "fees": round(poly_fee + kal_fee, 6),
    }


def _downsample(curve: list[dict], keep: dict | None) -> list[dict]:
    if len(curve) <= MAX_CURVE_POINTS:
        return curve
    step = len(curve) / MAX_CURVE_POINTS
    sampled = [curve[int(k * step)] for k in range(MAX_CURVE_POINTS)]
    if keep is not None and keep not in sampled:
        sampled.append(keep)
    sampled.append(curve[-1])
    seen: set[int] = set()
    out = []
    for pt in sorted(sampled, key=lambda p: p["size"]):
        if pt["size"] not in seen:
            seen.add(pt["size"])
            out.append(pt)
    return out


def walk_books(
    poly_asks: list[Level], kalshi_asks: list[Level], fees: FeeModel, min_size: float = 1.0
) -> tuple[dict | None, list[dict]]:
    """Profit-maximizing size and the profit curve over the gross-positive region.

    Contracts are whole numbers (Kalshi requires it). Sizes below Polymarket's minimum order are skipped.
    """
    points = gross_positive_breakpoints(poly_asks, kalshi_asks)
    if not points:
        return None, []
    depth = points[-1]
    floor_size = max(1, math.ceil(min_size))
    candidates = {int(math.floor(p)) for p in points}
    candidates.add(floor_size)
    sizes = sorted(q for q in candidates if floor_size <= q <= depth)

    curve = [pt for q in sizes if (pt := evaluate_size(poly_asks, kalshi_asks, q, fees)) is not None]
    if not curve:
        return None, []
    best = max(curve, key=lambda pt: (pt["profit"], -pt["size"]))
    return best, _downsample(curve, best)


def days_until(when: datetime | None, now: datetime) -> float | None:
    if when is None:
        return None
    return (when - now).total_seconds() / 86400


def annualize(net_return: float, days: float, min_days: float) -> float:
    """Simple (non-compounded) annualization. Compounding a 2%-in-one-day arb gives nonsense like 1000x."""
    return net_return * 365.0 / max(days, min_days)


def evaluate_direction(
    poly_asks: list[Level],
    kalshi_asks: list[Level],
    poly_side: str,
    kalshi_side: str,
    fees: FeeModel,
    days: float | None,
    *,
    poly_min_size: float = 1.0,
    min_days: float = 1.0,
    risk_free_rate: float = 0.0,
    walk: bool = True,
) -> ArbResult:
    res = ArbResult(
        poly_side=poly_side,
        kalshi_side=kalshi_side,
        poly_price=poly_asks[0].price if poly_asks else None,
        kalshi_price=kalshi_asks[0].price if kalshi_asks else None,
        days_to_resolution=days,
    )
    if res.poly_price is None or res.kalshi_price is None:
        return res

    cost = res.poly_price + res.kalshi_price
    res.gross_cost_top = cost
    res.gross_return_top = (1 - cost) / cost
    net_cost = cost + fees.per_contract(res.poly_price, res.kalshi_price)
    res.net_return_top = (1 - net_cost) / net_cost

    if cost >= 1 or not walk:
        return res

    best, curve = walk_books(poly_asks, kalshi_asks, fees, poly_min_size)
    res.book_walked = True
    res.curve = curve
    if best is None or best["profit"] <= 0:
        res.net_return = best["return"] if best else None
        return res

    res.size = best["size"]
    res.capital = best["capital"]
    res.profit = best["profit"]
    res.net_return = best["return"]
    if days is not None and res.net_return is not None:
        res.annualized_return = annualize(res.net_return, days, min_days)
        res.excess_annualized_return = res.annualized_return - risk_free_rate
    return res


def evaluate_pair(
    poly_book: BinaryBook,
    kalshi_book: BinaryBook,
    fees: FeeModel,
    *,
    inverted: bool = False,
    resolution: datetime | None = None,
    now: datetime | None = None,
    min_days: float = 1.0,
    risk_free_rate: float = 0.0,
) -> ArbResult:
    """Best direction for a matched pair: highest fee-adjusted profit, else highest top-of-book gross return."""
    now = now or datetime.now(timezone.utc)
    days = days_until(resolution, now)
    results = []
    for poly_side, kalshi_side in directions(inverted):
        results.append(
            evaluate_direction(
                poly_book.yes_asks if poly_side == "YES" else poly_book.no_asks,
                kalshi_book.yes_asks if kalshi_side == "YES" else kalshi_book.no_asks,
                poly_side,
                kalshi_side,
                fees,
                days,
                poly_min_size=poly_book.min_order_size or 1.0,
                min_days=min_days,
                risk_free_rate=risk_free_rate,
            )
        )
    return max(
        results,
        key=lambda r: (r.profit, r.gross_return_top if r.gross_return_top is not None else -math.inf),
    )
