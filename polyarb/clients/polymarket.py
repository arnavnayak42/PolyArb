"""Polymarket public APIs: Gamma (market metadata) and CLOB (order books). No auth needed for reads."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import requests

from polyarb.clients.http import make_session, parse_ts, to_float
from polyarb.models import BinaryBook, Level, PolyMarket, sort_asks

log = logging.getLogger(__name__)

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
BOOKS_BATCH = 100

# Used only when a market says fees are enabled but ships no feeSchedule. Deliberately the common
# "Other/General" taker rate: overstating fees can only make us under-report arbs, never invent them.
FALLBACK_FEE_RATE = 0.05


def _json_list(value) -> list:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value or "[]")
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def parse_fee(m: dict) -> tuple[float, float]:
    if not m.get("feesEnabled"):
        return 0.0, 1.0
    sched = m.get("feeSchedule") or {}
    if not sched:
        return FALLBACK_FEE_RATE, 1.0
    return to_float(sched.get("rate"), FALLBACK_FEE_RATE), to_float(sched.get("exponent"), 1.0)


def parse_market(m: dict) -> PolyMarket | None:
    """Two-outcome markets with a live order book; everything else is skipped.

    Yes/No markets map directly. Other two-outcome markets (team vs team, Over/Under) map the first
    outcome to YES; the matcher then compares against Kalshi's "will <first outcome> ..." phrasing.
    """
    outcomes = [str(o).strip() for o in _json_list(m.get("outcomes"))]
    tokens = [str(t) for t in _json_list(m.get("clobTokenIds"))]
    if len(outcomes) != 2 or len(tokens) != 2:
        return None
    is_yes_no = [o.lower() for o in outcomes] == ["yes", "no"]
    if m.get("closed") or not m.get("enableOrderBook") or m.get("acceptingOrders") is False:
        return None
    events = m.get("events") or []
    fee_rate, fee_exp = parse_fee(m)
    return PolyMarket(
        market_id=str(m["id"]),
        question=m.get("question") or "",
        event_title=(events[0].get("title") or "") if events else "",
        rules=m.get("description") or "",
        end_date=parse_ts(m.get("endDate")),
        yes_token=tokens[0],
        no_token=tokens[1],
        fee_rate=fee_rate,
        fee_exponent=fee_exp,
        liquidity=to_float(m.get("liquidityNum") or m.get("liquidity"), 0.0),
        slug=m.get("slug") or "",
        resolution_source=m.get("resolutionSource") or "",
        yes_outcome="Yes" if is_yes_no else outcomes[0],
        no_outcome="No" if is_yes_no else outcomes[1],
    )


def parse_book(raw: dict) -> tuple[list[Level], float]:
    """Asks best-first plus min order size. The API lists asks worst-first, so we always re-sort."""
    asks = [Level(float(a["price"]), float(a["size"])) for a in raw.get("asks") or []]
    return sort_asks(asks), to_float(raw.get("min_order_size"), 0.0)


class PolymarketClient:
    def __init__(self, session: requests.Session | None = None, timeout: float = 20.0):
        self.session = session or make_session()
        self.timeout = timeout

    def list_markets(self, min_liquidity: float = 0.0, max_days: float | None = None) -> list[PolyMarket]:
        params: dict = {"active": "true", "closed": "false", "limit": 100}
        if min_liquidity > 0:
            params["liquidity_num_min"] = min_liquidity
        now = datetime.now(timezone.utc)
        params["end_date_min"] = now.isoformat()
        if max_days:
            params["end_date_max"] = (now + timedelta(days=max_days)).isoformat()

        markets: list[PolyMarket] = []
        cursor = None
        while True:
            page_params = dict(params, after_cursor=cursor) if cursor else params
            resp = self.session.get(f"{GAMMA}/markets/keyset", params=page_params, timeout=self.timeout)
            resp.raise_for_status()
            body = resp.json()
            for raw in body.get("markets") or []:
                if (pm := parse_market(raw)) is not None:
                    markets.append(pm)
            cursor = body.get("next_cursor")
            if not cursor or not body.get("markets"):
                break
        log.info("polymarket: %d binary markets", len(markets))
        return markets

    def get_books(self, token_ids: list[str]) -> dict[str, tuple[list[Level], float]]:
        out: dict[str, tuple[list[Level], float]] = {}
        unique = list(dict.fromkeys(token_ids))
        for i in range(0, len(unique), BOOKS_BATCH):
            chunk = unique[i : i + BOOKS_BATCH]
            resp = self.session.post(
                f"{CLOB}/books", json=[{"token_id": t} for t in chunk], timeout=self.timeout
            )
            resp.raise_for_status()
            for raw in resp.json() or []:
                out[str(raw.get("asset_id"))] = parse_book(raw)
        return out

    @staticmethod
    def binary_book(books: dict[str, tuple[list[Level], float]], yes_token: str, no_token: str) -> BinaryBook | None:
        """Combine the YES and NO token books.

        Only each token's own asks are used. Polymarket can also fill a NO buy by matching a YES bid
        (mint/merge), but whether the book endpoint already mirrors that isn't documented, so we don't
        synthesize those levels. Depth may be understated, never double-counted.
        """
        if yes_token not in books or no_token not in books:
            return None
        yes_asks, yes_min = books[yes_token]
        no_asks, no_min = books[no_token]
        return BinaryBook(yes_asks=yes_asks, no_asks=no_asks, min_order_size=max(yes_min, no_min))
