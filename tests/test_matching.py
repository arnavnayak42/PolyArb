from datetime import datetime, timedelta, timezone

from polyarb.matching import kalshi_profile, match_markets, normalize, poly_profile, score_pair
from polyarb.models import KalshiMarket, PolyMarket

T0 = datetime(2026, 11, 4, tzinfo=timezone.utc)


def pm(mid, question, event="", rules="", end=T0, yes="Yes"):
    return PolyMarket(mid, question, event, rules, end, f"y{mid}", f"n{mid}", 0.04, 1.0, 1000.0, mid, yes_outcome=yes)


def km(ticker, title, event="", sub="", rules="", res=T0, sources=()):
    return KalshiMarket(ticker, "EV", "SER", title, event, sub, rules, res, res, settlement_sources=list(sources))


def score(p, k):
    return score_pair(p, k, poly_profile(p), kalshi_profile(k))


def test_normalize_strips_accents_and_thousands_separators():
    assert normalize("Will Falcão hit $10,000?") == "will falcao hit 10000?"


def test_identical_questions_score_high():
    r = score(pm("1", "Will Bitcoin outperform Gold in 2026?"), km("K1", "Will Bitcoin outperform gold in 2026?"))
    assert r.confidence > 0.9 and r.flags == []


def test_threshold_mismatch_is_flagged():
    r = score(pm("1", "Will Atlante score over 4.5 goals?"), km("K1", "Will Atlante score over 5.5 goals?"))
    assert any(f.startswith("number_mismatch") for f in r.flags)
    assert r.confidence < 0.4


def test_year_and_cutoff_mismatch():
    r = score(
        pm("1", "Will the Lakers win the 2026 NBA Finals?"),
        km("K1", "Will the Lakers win the 2027 NBA Finals?", res=T0 + timedelta(days=30)),
    )
    assert any(f.startswith("year_mismatch") for f in r.flags)
    assert any(f.startswith("cutoff_mismatch") for f in r.flags)


def test_polarity_opposite_suggests_inversion():
    r = score(pm("1", "Will CPI be above 3% in October?"), km("K1", "Will CPI be below 3% in October?"))
    assert r.suggest_inverted
    assert any(f.startswith("polarity_opposite") for f in r.flags)


def test_source_mismatch():
    r = score(
        pm("1", "Will X win?", rules="Resolves per the Associated Press call."),
        km("K1", "Will X win?", rules="Resolves per Fox News.", sources=["Fox News"]),
    )
    assert any(f.startswith("source_mismatch") for f in r.flags)


def test_draw_market_not_matched_to_win_market():
    r = score(pm("1", "Will Spain vs. Czechia end in a draw?"), km("K1", "Czechia wins", event="Spain vs Czechia"))
    assert any(f.startswith("draw_vs_win") for f in r.flags)
    assert r.confidence < 0.4


def test_match_markets_pairs_same_outcome_not_same_event():
    """Same election, different candidates: each candidate should pair with its own counterpart."""
    candidates = ["Cleitinho Azevedo", "Mateus Simoes", "Eduardo Falcao"]
    filler_p = [pm(f"f{i}", f"Will filler topic {i} happen?") for i in range(20)]
    filler_k = [km(f"KF{i}", f"Will other filler thing {i} occur?") for i in range(20)]
    poly = [pm(str(i), f"Will {c} win the 2026 Minas Gerais governor election?", event="Minas Gerais Governor Election Winner")
            for i, c in enumerate(candidates)] + filler_p
    kalshi = [km(f"K{i}", "Minas Gerais Governor winner?", sub=c) for i, c in enumerate(candidates)] + filler_k
    matches = {m.poly.market_id: m.kalshi.ticker for m in match_markets(poly, kalshi, min_confidence=0.4)}
    assert matches.get("0") == "K0"
    assert matches.get("1") == "K1"
    assert matches.get("2") == "K2"


def test_match_markets_respects_exclusions_and_reservations():
    poly = [pm("1", "Will Bitcoin outperform Gold in 2026?")]
    kalshi = [km("K1", "Will Bitcoin outperform gold in 2026?")]
    assert match_markets(poly, kalshi, exclude={("1", "K1")}) == []
    assert match_markets(poly, kalshi, reserved_kalshi={"K1"}) == []
    assert len(match_markets(poly, kalshi)) == 1


def test_exact_bucket_vs_threshold_flagged():
    r = score(pm("1", "Will Core CPI YoY be 2.2% in September?"), km("K1", "Will Core CPI YoY be above 2.2% in September?"))
    assert any(f.startswith("threshold_vs_exact") for f in r.flags)


def test_market_type_mismatch_total_vs_spread():
    r = score(
        pm("1", "Minnesota vs. Washington: O/U 27.5"),
        km("K1", "Spread", event="Minnesota vs Washington", sub="Washington wins by over 27.5 points"),
    )
    assert any(f.startswith("market_type_mismatch") for f in r.flags)
    assert r.confidence < 0.4


def test_opposite_team_outcome_flagged():
    p = pm("1", "MBA Moscow vs. CSKA Moscow", yes="MBA Moscow")
    p.no_outcome = "CSKA Moscow"
    r = score(p, km("K1", "CSKA Moscow vs MBA Moscow", sub="CSKA Moscow"))
    assert r.suggest_inverted
    assert any(f.startswith("opposite_outcome") for f in r.flags)
    same = score(p, km("K2", "CSKA Moscow vs MBA Moscow", sub="MBA Moscow"))
    assert not any(f.startswith("opposite_outcome") for f in same.flags)


def test_different_state_is_entity_mismatch():
    r = score(
        pm("1", "Will Democrats win exactly 5 House seats in Minnesota in the 2026 midterm elections?"),
        km("K1", "How many House seats will Democrats win in Georgia? Will Democrats win exactly 5 seats?"),
    )
    assert any(f.startswith("entity_mismatch") for f in r.flags)
    assert r.confidence < 0.4


def test_game_period_and_round_types_distinguished():
    r = score(pm("1", "Georgia Tech vs. Stanford: 2Q O/U 19.5"), km("K1", "Georgia Tech vs Stanford: 2nd Half Total Over 19.5"))
    assert any(f.startswith("market_type_mismatch") for f in r.flags)
    r = score(pm("1", "Will Lula win the 2026 Brazilian presidential election?"),
              km("K1", "Brazil presidential election: first round winner? Will Lula win?"))
    assert any(f.startswith("market_type_mismatch") for f in r.flags)


def test_any_other_bucket_flagged():
    r = score(pm("1", "Will any other team win the 2026 Laver Cup?"), km("K1", "Will Team World win the 2026 Laver Cup?"))
    assert any(f.startswith("other_bucket") for f in r.flags)
    assert r.confidence < 0.4


def test_margin_bracket_vs_outright_win():
    r = score(
        pm("1", "Will Lula win the first round of the 2026 Brazilian presidential election by 5–10%?"),
        km("K1", "Will Lula win the first round of the 2026 Brazilian presidential election?"),
    )
    assert any(f.startswith("range_bucket_only_in_PM") for f in r.flags)
    assert r.confidence < 0.4
