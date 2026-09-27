import pytest

from polyarb.fees import kalshi_fee, kalshi_fee_per_contract, polymarket_fee


def test_kalshi_fee_standard_schedule():
    # 0.07 * 100 * 0.5 * 0.5 = 1.75
    assert kalshi_fee([(0.50, 100)]) == pytest.approx(1.75)


def test_kalshi_fee_rounds_up_to_cent_per_order():
    # 0.07 * 1 * 0.25 = 0.0175 -> $0.02; small orders pay proportionally more
    assert kalshi_fee([(0.50, 1)]) == pytest.approx(0.02)
    # 0.07 * 10 * 0.1 * 0.9 = 0.063 -> $0.07
    assert kalshi_fee([(0.10, 10)]) == pytest.approx(0.07)


def test_kalshi_fee_exact_cent_not_bumped_by_float_noise():
    # 0.07 * 400 * 0.5 * 0.5 = 7.00 exactly; must not round up to 7.01
    assert kalshi_fee([(0.50, 400)]) == pytest.approx(7.00)


def test_kalshi_fee_multiplier_and_multiple_levels():
    fills = [(0.40, 50), (0.45, 50)]
    raw = 0.07 * 0.5 * (50 * 0.4 * 0.6 + 50 * 0.45 * 0.55)
    assert kalshi_fee(fills, multiplier=0.5) == pytest.approx(round(raw + 0.005, 2), abs=0.01)
    assert kalshi_fee(fills, multiplier=0.5) >= raw


def test_polymarket_fee_symmetric_and_zero_rate():
    assert polymarket_fee([(0.30, 100)], rate=0.04) == pytest.approx(0.84)
    assert polymarket_fee([(0.70, 100)], rate=0.04) == pytest.approx(0.84)
    assert polymarket_fee([(0.30, 100)], rate=0.0) == 0.0


def test_fee_vanishes_at_extremes():
    assert kalshi_fee_per_contract(0.99) < kalshi_fee_per_contract(0.5) / 10
