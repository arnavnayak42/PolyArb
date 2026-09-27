from sqlalchemy import select

from polyarb.db import MarketPair, Observation, PairStatus, Scan, session_factory
from polyarb.scanner import Scanner
from tests.conftest import FakeKalshi, FakePoly, kbook, ladder, make_km, make_pm


def build(engine, settings):
    poly_markets = [
        make_pm("1", "Will Bitcoin outperform Gold in 2026?"),
        make_pm("2", "Will the Fed cut rates in December 2026?"),
    ]
    kalshi_markets = [
        make_km("KBTC", "Will Bitcoin outperform gold in 2026?"),
        make_km("KFED", "Will the Fed cut rates in December 2026?"),
    ]
    poly_books = {
        # Pair 1: PM YES 0.40 + K NO 0.55 = 0.95 -> arb
        "y1": (ladder((0.40, 200)), 5.0),
        "n1": (ladder((0.62, 200)), 5.0),
        # Pair 2: no arb either way
        "y2": (ladder((0.50, 100)), 5.0),
        "n2": (ladder((0.52, 100)), 5.0),
    }
    kalshi_books = {
        "KBTC": kbook(yes=[(0.58, 100)], no=[(0.55, 100), (0.58, 100)]),
        "KFED": kbook(yes=[(0.53, 100)], no=[(0.55, 100)]),
    }
    poly = FakePoly(poly_markets, poly_books)
    kalshi = FakeKalshi(kalshi_markets, kalshi_books, multiplier=1.0)
    return Scanner(settings, engine, poly=poly, kalshi=kalshi), kalshi


def test_rematch_then_scan_logs_and_detects_arb(engine, settings):
    scanner, kalshi = build(engine, settings)
    stats = scanner.rematch()
    assert stats["matches"] == 2

    report = scanner.scan_once()
    assert report.error is None
    assert report.pairs_checked == 2
    assert report.gross_arbs == 1 and report.net_arbs == 1
    # Only the pair whose top-of-book crossed needed a depth fetch
    assert kalshi.orderbook_calls == 1

    opp = report.opportunities[0]
    assert opp.pair.kalshi_ticker == "KBTC"
    assert opp.result.direction == "PM YES + K NO"
    # 100 @ (0.40 + 0.55) + 100 @ (0.40 + 0.58); Kalshi fees on the NO leg
    assert opp.result.size == 200

    with session_factory(engine)() as s:
        obs = s.scalars(select(Observation)).all()
        # Pair 2 best direction costs 1.05 (gross -4.8%), below the -2% log floor, so not logged
        assert {o.pair_id for o in obs} == {opp.pair.id}
        assert obs[0].is_net_arb and obs[0].profit_curve
        assert s.scalars(select(Scan)).one().net_arbs == 1


def test_rejected_pairs_not_scanned_and_not_rematched(engine, settings):
    scanner, _ = build(engine, settings)
    scanner.rematch()
    Session = session_factory(engine)
    with Session() as s:
        btc = s.scalars(select(MarketPair).where(MarketPair.kalshi_ticker == "KBTC")).one()
        btc.status = PairStatus.REJECTED
        s.commit()
    assert scanner.scan_once().net_arbs == 0
    scanner.rematch()
    with Session() as s:
        assert s.scalars(select(MarketPair).where(MarketPair.kalshi_ticker == "KBTC")).one().status == PairStatus.REJECTED


def test_confirmed_status_and_inversion_survive_rematch(engine, settings):
    scanner, _ = build(engine, settings)
    scanner.rematch()
    Session = session_factory(engine)
    with Session() as s:
        fed = s.scalars(select(MarketPair).where(MarketPair.kalshi_ticker == "KFED")).one()
        fed.status = PairStatus.CONFIRMED
        fed.inverted = True
        s.commit()
    scanner.rematch()
    with Session() as s:
        fed = s.scalars(select(MarketPair).where(MarketPair.kalshi_ticker == "KFED")).one()
        assert fed.status == PairStatus.CONFIRMED and fed.inverted


def test_failed_depth_fetch_skips_pair_instead_of_inventing_size(engine, settings):
    scanner, kalshi = build(engine, settings)
    scanner.rematch()

    def boom(ticker):
        raise RuntimeError("503")

    kalshi.get_orderbook = boom
    report = scanner.scan_once()
    assert report.error is None
    assert report.net_arbs == 0 and report.logged == 0


def test_pairs_dropped_by_latest_rematch_are_not_scanned(engine, settings):
    scanner, _ = build(engine, settings)
    scanner.rematch()
    # Next universe no longer contains the BTC markets, so that candidate pair must go quiet.
    scanner.poly.markets = [m for m in scanner.poly.markets if m.market_id != "1"]
    scanner.rematch()
    with session_factory(engine)() as s:
        assert {p.kalshi_ticker for p in scanner.active_pairs(s)} == {"KFED"}
