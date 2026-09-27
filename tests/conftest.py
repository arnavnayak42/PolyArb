from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.pool import StaticPool
from sqlalchemy import create_engine

from polyarb.config import Settings
from polyarb.db import init_db
from polyarb.models import BinaryBook, KalshiMarket, Level, PolyMarket

NOW = datetime.now(timezone.utc)


@pytest.fixture
def engine():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    init_db(eng)
    return eng


@pytest.fixture
def settings():
    return Settings(database_url="sqlite://", min_return=0.0, min_match_confidence=0.5, log_gross_floor=-0.02)


class FakePoly:
    def __init__(self, markets, books):
        self.markets = markets
        self.books = books  # token -> (asks, min_size)

    def list_markets(self, **_):
        return self.markets

    def get_books(self, tokens):
        return {t: self.books[t] for t in tokens if t in self.books}


class FakeKalshi:
    def __init__(self, markets, books, multiplier=1.0):
        self.markets = markets
        self.books = books  # ticker -> BinaryBook
        self.multiplier = multiplier
        self.orderbook_calls = 0

    def list_markets(self, **_):
        return self.markets

    def series_fee_multiplier(self, series):
        return self.multiplier

    def get_quotes(self, tickers):
        return {t: (self.books[t].best("YES"), self.books[t].best("NO")) for t in tickers if t in self.books}

    def get_orderbook(self, ticker):
        self.orderbook_calls += 1
        return self.books[ticker]


def make_pm(mid, question, end=None):
    return PolyMarket(mid, question, "", "", end or NOW + timedelta(days=30), f"y{mid}", f"n{mid}", 0.0, 1.0, 1e4, mid)


def make_km(ticker, title, res=None):
    res = res or NOW + timedelta(days=30)
    return KalshiMarket(ticker, "EV", "SER", title, "", "", "", res, res)


def ladder(*pairs):
    return [Level(p, q) for p, q in pairs]


def kbook(yes, no):
    return BinaryBook(yes_asks=ladder(*yes), no_asks=ladder(*no))
