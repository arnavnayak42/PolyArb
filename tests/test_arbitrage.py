from datetime import datetime, timedelta, timezone

import pytest

from polyarb.arbitrage import (
    FeeModel,
    annualize,
    directions,
    evaluate_direction,
    evaluate_pair,
    gross_positive_breakpoints,
    walk_books,
)
from polyarb.models import BinaryBook, Level

NO_FEES = FeeModel(poly_rate=0.0, kalshi_multiplier=0.0)


def L(*pairs):
    return [Level(p, q) for p, q in pairs]


def test_breakpoints_stop_when_combined_ask_reaches_one():
    a = L((0.40, 100))
    b = L((0.55, 50), (0.58, 100), (0.61, 100))
    # 0.40+0.55 and 0.40+0.58 are < 1; 0.40+0.61 is not
    assert gross_positive_breakpoints(a, b) == [50, 100]


def test_walk_finds_profit_maximizing_size_without_fees():
    best, curve = walk_books(L((0.40, 100)), L((0.55, 50), (0.58, 100)), NO_FEES)
    # size 100: cost 40 + 27.5 + 29 = 96.5 -> profit 3.5; size 50: profit 2.5
    assert best["size"] == 100
    assert best["profit"] == pytest.approx(3.5)
    assert [pt["size"] for pt in curve] == [1, 50, 100]


def test_walk_stops_before_unprofitable_depth():
    best, _ = walk_books(L((0.45, 10), (0.52, 100)), L((0.50, 200)), NO_FEES)
    assert best["size"] == 10
    assert best["profit"] == pytest.approx(0.5)


def test_fees_can_kill_a_gross_arb():
    fees = FeeModel(poly_rate=0.05, kalshi_multiplier=1.0)
    res = evaluate_direction(L((0.49, 1000)), L((0.49, 1000)), "YES", "NO", fees, days=30)
    assert res.is_gross_arb  # 0.98 < 1
    assert not res.is_net_arb  # ~0.03 of fees per contract at 50c
    assert res.net_return_top < 0


def test_kalshi_cent_rounding_makes_tiny_orders_unprofitable():
    fees = FeeModel(poly_rate=0.0, kalshi_multiplier=1.0)
    best, curve = walk_books(L((0.48, 1000)), L((0.50, 1000)), fees)
    one = next(pt for pt in curve if pt["size"] == 1)
    assert one["profit"] == pytest.approx(0.0)  # 0.02 edge eaten by a rounded-up 0.02 fee
    assert best["size"] == 1000
    assert best["profit"] == pytest.approx(1000 - 980 - 17.5)


def test_polymarket_min_order_size_respected():
    best, _ = walk_books(L((0.40, 3)), L((0.50, 3)), NO_FEES, min_size=5)
    assert best is None


def test_directions_normal_and_inverted():
    assert directions(False) == [("YES", "NO"), ("NO", "YES")]
    assert directions(True) == [("YES", "YES"), ("NO", "NO")]


def test_evaluate_pair_picks_profitable_direction_and_annualizes():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    poly = BinaryBook(yes_asks=L((0.60, 500)), no_asks=L((0.42, 500)), min_order_size=5)
    kalshi = BinaryBook(yes_asks=L((0.55, 500)), no_asks=L((0.47, 500)))
    res = evaluate_pair(poly, kalshi, NO_FEES, resolution=now + timedelta(days=36.5), now=now, risk_free_rate=0.04)
    # PM NO 0.42 + K YES 0.55 = 0.97 is the arb; PM YES 0.60 + K NO 0.47 = 1.07 is not
    assert (res.poly_side, res.kalshi_side) == ("NO", "YES")
    assert res.size == 500
    assert res.net_return == pytest.approx(0.03 / 0.97)
    assert res.annualized_return == pytest.approx(0.03 / 0.97 * 10)
    assert res.excess_annualized_return == pytest.approx(res.annualized_return - 0.04)


def test_inverted_pair_uses_same_side_on_both_venues():
    poly = BinaryBook(yes_asks=L((0.30, 100)), no_asks=L((0.72, 100)))
    kalshi = BinaryBook(yes_asks=L((0.65, 100)), no_asks=L((0.37, 100)))
    res = evaluate_pair(poly, kalshi, NO_FEES, inverted=True)
    assert (res.poly_side, res.kalshi_side) == ("YES", "YES")
    assert res.gross_cost_top == pytest.approx(0.95)


def test_no_arb_when_books_missing():
    res = evaluate_pair(BinaryBook(), BinaryBook(yes_asks=L((0.5, 10))), NO_FEES)
    assert not res.is_gross_arb and res.size == 0


def test_annualize_floors_days():
    assert annualize(0.01, days=0.1, min_days=1.0) == pytest.approx(3.65)
