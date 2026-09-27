"""Kalshi public trade API v2 (market data needs no auth)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import requests

from polyarb.clients.http import RateLimiter, make_session, parse_ts, to_float
from polyarb.models import BinaryBook, KalshiMarket, Level, sort_asks

log = logging.getLogger(__name__)

BASE = "https://api.elections.kalshi.com/trade-api/v2"
QUOTES_BATCH = 100


def price(m: dict, key: str) -> float | None:
    """Kalshi now reports prices as dollar strings (`yes_ask_dollars`); older payloads used integer cents."""
    dollars = to_float(m.get(f"{key}_dollars"))
    if dollars is not None:
        return dollars
    cents = to_float(m.get(key))
    return cents / 100 if cents is not None else None


def ask_or_none(value: float | None) -> float | None:
    """An ask of 0 or >= $1 means nobody is offering that side."""
    return value if value is not None and 0 < value < 1 else None


def parse_market(m: dict, event: dict | None = None) -> KalshiMarket | None:
    if m.get("market_type", "binary") != "binary" or m.get("status") not in (None, "active", "open"):
        return None
    event = event or {}
    ticker = m["ticker"]
    event_ticker = m.get("event_ticker") or event.get("event_ticker") or ""
    series = event.get("series_ticker") or event_ticker.split("-")[0]
    rules = "\n".join(filter(None, [m.get("rules_primary"), m.get("rules_secondary")]))
    return KalshiMarket(
        ticker=ticker,
        event_ticker=event_ticker,
        series_ticker=series,
        title=m.get("title") or "",
        event_title=event.get("title") or "",
        yes_sub_title=m.get("yes_sub_title") or "",
        rules=rules,
        close_time=parse_ts(m.get("close_time")),
        expected_expiration=parse_ts(m.get("expected_expiration_time")),
        category=event.get("category") or "",
        settlement_sources=[s.get("name", "") for s in event.get("settlement_sources") or [] if s.get("name")],
        yes_ask=ask_or_none(price(m, "yes_ask")),
        no_ask=ask_or_none(price(m, "no_ask")),
    )


def parse_orderbook(body: dict) -> BinaryBook:
    """Kalshi books only list bids. A NO bid at p is a YES offer at 1-p (and vice versa), so:
    YES asks = 1 - NO bids, NO asks = 1 - YES bids."""
    ob = body.get("orderbook_fp")
    if ob is not None:
        yes_bids = [(float(p), float(q)) for p, q in ob.get("yes_dollars") or []]
        no_bids = [(float(p), float(q)) for p, q in ob.get("no_dollars") or []]
    else:
        ob = body.get("orderbook") or {}
        yes_bids = [(p / 100, float(q)) for p, q in ob.get("yes") or []]
        no_bids = [(p / 100, float(q)) for p, q in ob.get("no") or []]
    yes_asks = sort_asks([Level(round(1 - p, 6), q) for p, q in no_bids])
    no_asks = sort_asks([Level(round(1 - p, 6), q) for p, q in yes_bids])
    return BinaryBook(yes_asks=yes_asks, no_asks=no_asks, min_order_size=1.0)


class KalshiClient:
    def __init__(
        self,
        session: requests.Session | None = None,
        timeout: float = 20.0,
        rps: float = 10.0,
        limiter: RateLimiter | None = None,
    ):
        self.session = session or make_session()
        self.timeout = timeout
        # Pass a shared limiter when several clients (e.g. scan + background rematch) hit Kalshi at once.
        self.limiter = limiter or RateLimiter(rps)
        self._series_cache: dict[str, dict] = {}

    def _get(self, path: str, params: dict | None = None) -> dict:
        self.limiter.wait()
        resp = self.session.get(f"{BASE}{path}", params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def list_markets(self, max_days: float | None = None) -> list[KalshiMarket]:
        """All open binary markets, walked via /events so each market carries its event title,
        series and settlement sources (market titles alone are often context-free, e.g. "Over 13.5 runs")."""
        now = datetime.now(timezone.utc)
        markets: list[KalshiMarket] = []
        cursor = None
        while True:
            params = {"status": "open", "with_nested_markets": "true", "limit": 200}
            if cursor:
                params["cursor"] = cursor
            body = self._get("/events", params)
            for event in body.get("events") or []:
                for raw in event.get("markets") or []:
                    km = parse_market(raw, event)
                    if km is None or (km.yes_ask is None and km.no_ask is None):
                        continue
                    res = km.resolution_time
                    if max_days and res is not None and (res - now).total_seconds() > max_days * 86400:
                        continue
                    markets.append(km)
            cursor = body.get("cursor")
            if not cursor or not body.get("events"):
                break
        log.info("kalshi: %d binary markets with quotes", len(markets))
        return markets

    def series_fee_multiplier(self, series_ticker: str) -> float:
        """Series-level fee multiplier (1.0 = standard 0.07 schedule). Cached; defaults to 1.0 on error,
        which is the conservative (higher-fee) assumption for most series."""
        if series_ticker not in self._series_cache:
            try:
                self._series_cache[series_ticker] = self._get(f"/series/{series_ticker}").get("series") or {}
            except requests.RequestException as exc:
                log.warning("kalshi series %s lookup failed: %s", series_ticker, exc)
                return 1.0
        series = self._series_cache[series_ticker]
        mult = to_float(series.get("fee_multiplier"))
        return mult if mult is not None else 1.0

    def get_quotes(self, tickers: list[str]) -> dict[str, tuple[float | None, float | None]]:
        """Top-of-book (yes_ask, no_ask) for many tickers in few requests; used to pre-screen pairs."""
        out: dict[str, tuple[float | None, float | None]] = {}
        unique = list(dict.fromkeys(tickers))
        for i in range(0, len(unique), QUOTES_BATCH):
            chunk = unique[i : i + QUOTES_BATCH]
            body = self._get("/markets", {"tickers": ",".join(chunk), "limit": len(chunk)})
            for m in body.get("markets") or []:
                out[m["ticker"]] = (ask_or_none(price(m, "yes_ask")), ask_or_none(price(m, "no_ask")))
        return out

    def get_orderbook(self, ticker: str) -> BinaryBook:
        return parse_orderbook(self._get(f"/markets/{ticker}/orderbook"))
