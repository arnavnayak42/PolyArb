"""Venue-neutral domain objects shared by the clients, matcher and arb engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Level:
    """One price level of an ask ladder: `size` contracts available at `price` dollars (0-1)."""

    price: float
    size: float


def sort_asks(levels: list[Level]) -> list[Level]:
    """Best (cheapest) ask first, dropping empty or out-of-range levels."""
    return sorted((lv for lv in levels if lv.size > 0 and 0 < lv.price < 1), key=lambda lv: lv.price)


@dataclass
class BinaryBook:
    """Ask ladders for both sides of a binary market, each sorted best-first."""

    yes_asks: list[Level] = field(default_factory=list)
    no_asks: list[Level] = field(default_factory=list)
    min_order_size: float = 0.0

    def best(self, side: str) -> float | None:
        ladder = self.yes_asks if side == "YES" else self.no_asks
        return ladder[0].price if ladder else None


@dataclass
class PolyMarket:
    market_id: str
    question: str
    event_title: str
    rules: str
    end_date: datetime | None
    yes_token: str
    no_token: str
    fee_rate: float
    fee_exponent: float
    liquidity: float
    slug: str
    resolution_source: str = ""
    # For two-outcome markets that aren't literally Yes/No (e.g. "Navy vs. UAB" with outcomes
    # [Navy, UAB]) we treat the first outcome as YES and the second as NO.
    yes_outcome: str = "Yes"
    no_outcome: str = "No"

    @property
    def match_text(self) -> str:
        text = self.question
        if self.event_title and self.event_title.lower() not in self.question.lower():
            text = f"{self.event_title} {text}"
        if self.yes_outcome.lower() != "yes":
            text = f"{text} {self.yes_outcome}"
        return text


@dataclass
class KalshiMarket:
    ticker: str
    event_ticker: str
    series_ticker: str
    title: str
    event_title: str
    yes_sub_title: str
    rules: str
    close_time: datetime | None
    expected_expiration: datetime | None
    category: str = ""
    settlement_sources: list[str] = field(default_factory=list)
    yes_ask: float | None = None
    no_ask: float | None = None

    @property
    def resolution_time(self) -> datetime | None:
        return self.expected_expiration or self.close_time

    @property
    def match_text(self) -> str:
        parts = [self.title]
        if self.event_title and self.event_title.lower() not in self.title.lower():
            parts.insert(0, self.event_title)
        if self.yes_sub_title and self.yes_sub_title.lower() not in self.title.lower():
            parts.append(self.yes_sub_title)
        return " ".join(parts)
