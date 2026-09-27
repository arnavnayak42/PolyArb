import json

import pytest

from polyarb.clients import kalshi, polymarket


def test_kalshi_orderbook_bids_become_opposite_side_asks():
    body = {"orderbook_fp": {"yes_dollars": [["0.3800", "20"], ["0.4000", "10"]], "no_dollars": [["0.5500", "5"]]}}
    book = kalshi.parse_orderbook(body)
    # YES asks come from NO bids: 1 - 0.55 = 0.45
    assert [(lv.price, lv.size) for lv in book.yes_asks] == [(0.45, 5.0)]
    # NO asks come from YES bids, best (highest YES bid -> cheapest NO) first
    assert [(lv.price, lv.size) for lv in book.no_asks] == [(0.60, 10.0), (0.62, 20.0)]


def test_kalshi_orderbook_legacy_cents_format():
    book = kalshi.parse_orderbook({"orderbook": {"yes": [[40, 10]], "no": [[55, 5]]}})
    assert book.yes_asks[0].price == pytest.approx(0.45)
    assert book.no_asks[0].price == pytest.approx(0.60)


def test_kalshi_price_prefers_dollar_fields():
    assert kalshi.price({"yes_ask_dollars": "0.3800", "yes_ask": 99}, "yes_ask") == pytest.approx(0.38)
    assert kalshi.price({"yes_ask": 38}, "yes_ask") == pytest.approx(0.38)
    assert kalshi.ask_or_none(1.0) is None and kalshi.ask_or_none(0.0) is None


def test_polymarket_book_sorted_best_first():
    raw = {"asks": [{"price": "0.99", "size": "10"}, {"price": "0.62", "size": "5"}], "min_order_size": "5"}
    asks, min_size = polymarket.parse_book(raw)
    assert [lv.price for lv in asks] == [0.62, 0.99]
    assert min_size == 5


def _gamma(**overrides):
    m = {
        "id": "1",
        "question": "Navy vs. UAB",
        "outcomes": json.dumps(["Navy", "UAB"]),
        "clobTokenIds": json.dumps(["111", "222"]),
        "enableOrderBook": True,
        "acceptingOrders": True,
        "closed": False,
        "endDate": "2026-10-01T00:00:00Z",
        "feesEnabled": True,
        "feeSchedule": {"rate": 0.05, "exponent": 1},
        "events": [{"title": "Navy vs. UAB"}],
    }
    m.update(overrides)
    return m


def test_polymarket_two_outcome_market_maps_first_outcome_to_yes():
    pm = polymarket.parse_market(_gamma())
    assert pm.yes_outcome == "Navy" and pm.yes_token == "111" and pm.no_token == "222"
    assert pm.fee_rate == 0.05
    assert "Navy" in pm.match_text


def test_polymarket_fee_parsing():
    assert polymarket.parse_fee({"feesEnabled": False}) == (0.0, 1.0)
    assert polymarket.parse_fee({"feesEnabled": True}) == (polymarket.FALLBACK_FEE_RATE, 1.0)


def test_polymarket_skips_multi_outcome_and_closed():
    assert polymarket.parse_market(_gamma(outcomes=json.dumps(["A", "B", "C"]))) is None
    assert polymarket.parse_market(_gamma(closed=True)) is None
