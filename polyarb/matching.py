"""Cross-venue event matching and resolution-risk scoring.

Stage 1, retrieval: TF-IDF cosine similarity between market texts (event title + question) proposes
the top few Kalshi candidates for each Polymarket market. Sparse matrix products keep this tractable at
~30k x ~120k markets.

Stage 2, resolution checks: each candidate pair is scored on the things that make "the same question"
resolve differently across venues:
  * text similarity                  (are they even about the same thing?)
  * numeric thresholds               (4.5 vs 5.5 goals, $100k vs $110k)
  * years                            (2026 vs 2027 season)
  * resolution-time gap              (cutoff dates differ)
  * resolution sources               (AP vs Fox call, BLS vs a news report)
  * edge-case wording                (runoff, postponement, overtime clauses on one side only)
  * polarity                         ("above" on one venue, "below" on the other)
  * entities                         (names/places on only one side: "Will Falcão win?" vs "Will Simões
                                      win?", or "...seats in Minnesota" vs "...seats in Georgia")
  * market type                      (spread vs total vs moneyline, goals vs corners, team vs game total)
  * threshold vs exact               ("CPI be 2.2%" bucket vs "CPI above 2.2%")
  * opposite team                    (Polymarket "A vs B" with YES=A, paired with Kalshi "B wins")

The product of these factors is the resolution confidence in [0, 1]. Every penalty also adds a
human-readable flag so a reviewer can see *why* a pair scored low.

Stage 3: greedy one-to-one assignment by confidence, so each market pairs with at most one counterpart.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

from polyarb.models import KalshiMarket, PolyMarket

log = logging.getLogger(__name__)

RETRIEVAL_TOP_K = 5
RETRIEVAL_MIN_COSINE = 0.30
MAX_RESOLUTION_GAP_DAYS = 120
CHUNK = 500
# A token appearing in at most this many markets (across both venues) is treated as a distinctive entity.
RARE_DF = 40

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "be", "is", "will", "and", "or",
    "vs", "v", "who", "what", "which", "when", "does", "do", "this", "that", "with", "from", "as",
}

SOURCE_PATTERNS: dict[str, str] = {
    "associated_press": r"\bassociated press\b|\bap\b|apnews",
    "fox": r"\bfox news\b|\bfox\b",
    "nbc": r"\bnbc\b",
    "cnn": r"\bcnn\b",
    "reuters": r"\breuters\b",
    "decision_desk": r"decision desk",
    "bls": r"\bbls\b|bureau of labor statistics",
    "bea": r"\bbea\b|bureau of economic analysis",
    "federal_reserve": r"federal reserve|\bfomc\b",
    "coinbase": r"\bcoinbase\b",
    "binance": r"\bbinance\b",
    "coingecko": r"\bcoingecko\b",
    "chainlink": r"\bchainlink\b",
    "cme": r"\bcme\b",
    "noaa_nws": r"\bnoaa\b|national weather service|\bnws\b",
    "aaa": r"\baaa\b",
    "espn": r"\bespn\b",
    "official_league": r"\b(nfl|nba|mlb|nhl|mls|ncaa|uefa|fifa|atp|wta|ufc|pga)\b\.?(com)?",
    "rotten_tomatoes": r"rotten tomatoes",
    "billboard": r"\bbillboard\b",
    "box_office_mojo": r"box office mojo|the numbers",
    "credible_reporting": r"consensus of credible reporting|credible report",
}
_SOURCE_RES = {name: re.compile(pat) for name, pat in SOURCE_PATTERNS.items()}

EDGE_CASE_TERMS = [
    "runoff", "tie", "postpone", "cancel", "delay", "recount", "overtime", "extra time", "penalt",
    "abandon", "forfeit", "disqualif", "certif", "inaugurat", "resign", "withdraw", "void", "suspend",
]
_EDGE_RES = {t: re.compile(r"\b" + re.escape(t)) for t in EDGE_CASE_TERMS}

UP_WORDS = {"above", "over", "more", "higher", "exceed", "exceeds", "greater", "least", "increase", "rise"}
DOWN_WORDS = {"below", "under", "less", "lower", "fewer", "decrease", "fall", "drop"}

Draw_WORDS = {"draw", "tie", "tied"}

# Capitalized words that are sentence furniture rather than entities.
CAP_STOP = {
    "will", "the", "who", "what", "which", "when", "how", "yes", "no", "a", "an", "in", "of", "o/u",
    "vs", "over", "under", "is", "be", "does", "do", "to", "on", "at", "by", "for", "and", "or",
}

MARKET_TYPES: dict[str, str] = {
    "spread": r"\bspread\b|wins? by (more than|over)|\bhandicap\b",
    "total": r"\bo/u\b|over/under|\btotal\b|(points|goals|runs) scored",
    "team_total": r"team total|\bscores over\b",
    "corners": r"\bcorners?\b",
    "cards": r"\b(yellow|red) cards?\b|\bbookings?\b",
    "btts": r"both teams to score|neither team to score|\bbtts\b",
    "exact_score": r"exact score|correct score|final score",
    "first_half": r"1st half|first half|\b1h\b",
    "second_half": r"2nd half|second half|\b2h\b",
    "quarter": r"\b[1-4]q\b|(1st|2nd|3rd|4th|first|second|third|fourth) quarter",
    "period": r"(1st|2nd|3rd|first|second|third) period|\bp[1-3]\b",
    "first_round": r"first round|1st round|round one",
    "player_prop": r"touchdowns?|rushing|receiving|passing yards|assists|rebounds|strikeouts|home runs?",
}
_TYPE_RES = {name: re.compile(pat) for name, pat in MARKET_TYPES.items()}
OTHER_RE = re.compile(r"\bany other\b|\bsomeone else\b|\bthe field\b|\banother (team|candidate|player|person)\b")

_NUM_RE = re.compile(r"(?<![a-z])\d+(?:\.\d+)?")
_YEAR_RE = re.compile(r"\b20[2-4]\d\b")
RANGE_RE = re.compile(r"\d+(?:\.\d+)?\s*%?\s*(?:-|\u2013|to)\s*\d+(?:\.\d+)?\s*%|\bby (?:more than |at least )?\d")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))  # Falcão -> Falcao
    text = text.replace("\u2019", "'").lower().replace("**", " ").replace("&", " and ")
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # 10,000 -> 10000
    text = text.replace("$", " ")
    return re.sub(r"\s+", " ", text).strip()


def _canon_num(s: str) -> str:
    return str(float(s)).rstrip("0").rstrip(".") if "." in s else str(int(s))


def extract_numbers(text: str) -> tuple[frozenset[str], frozenset[str]]:
    """(non-year numbers, years) found in normalized text."""
    years = frozenset(_YEAR_RE.findall(text))
    nums = frozenset(_canon_num(n) for n in _NUM_RE.findall(text) if n not in years)
    return nums, years


def extract_sources(text: str) -> frozenset[str]:
    t = text.lower()
    return frozenset(name for name, rx in _SOURCE_RES.items() if rx.search(t))


def extract_edge_terms(text: str) -> frozenset[str]:
    t = text.lower()
    return frozenset(term for term, rx in _EDGE_RES.items() if rx.search(t))


def proper_nouns(raw: str) -> frozenset[str]:
    """Lowercased capitalized words (names, places, teams) from the original, un-normalized text."""
    raw = unicodedata.normalize("NFKD", raw)
    raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
    out = set()
    for w in re.findall(r"[A-Za-z][A-Za-z'.\-]*", raw):
        w = w.rstrip(".'").removesuffix("'s")
        if len(w) >= 3 and w[0].isupper() and w.lower() not in CAP_STOP:
            out.add(w.lower())
    return frozenset(out)


def market_types(text: str) -> frozenset[str]:
    return frozenset(name for name, rx in _TYPE_RES.items() if rx.search(text))


def polarity(text: str) -> int:
    """+1 if the text only uses 'up' comparatives, -1 if only 'down', 0 if neither or both."""
    words = set(re.findall(r"[a-z]+", text))
    up, down = bool(words & UP_WORDS), bool(words & DOWN_WORDS)
    return (1 if up else 0) - (1 if down else 0) if up != down else 0


@dataclass(frozen=True)
class Profile:
    """Pre-extracted features for one market so each pair comparison is cheap."""

    text: str
    numbers: frozenset[str]
    years: frozenset[str]
    sources: frozenset[str]
    edge_terms: frozenset[str]
    polarity: int
    tokens: frozenset[str] = frozenset()
    rare: frozenset[str] = frozenset()
    words: frozenset[str] = frozenset()
    names: frozenset[str] = frozenset()
    types: frozenset[str] = frozenset()


def word_tokens(text: str) -> frozenset[str]:
    return frozenset(t for t in re.findall(r"[a-z]{3,}", text) if t not in STOPWORDS)


def poly_profile(pm: PolyMarket) -> Profile:
    raw = pm.match_text
    text = normalize(raw)
    nums, years = extract_numbers(text)
    return Profile(
        text=text,
        numbers=nums,
        years=years,
        sources=extract_sources(f"{pm.resolution_source}\n{pm.rules}"),
        edge_terms=extract_edge_terms(pm.rules),
        polarity=polarity(text),
        tokens=word_tokens(text),
        words=frozenset(re.findall(r"[a-z0-9]+", text)),
        names=proper_nouns(raw),
        types=market_types(text),
    )


def kalshi_profile(km: KalshiMarket) -> Profile:
    raw = km.match_text
    text = normalize(raw)
    nums, years = extract_numbers(text)
    return Profile(
        text=text,
        numbers=nums,
        years=years,
        sources=extract_sources("\n".join(km.settlement_sources) + "\n" + km.rules),
        edge_terms=extract_edge_terms(km.rules),
        polarity=polarity(text),
        tokens=word_tokens(text),
        words=frozenset(re.findall(r"[a-z0-9]+", text)),
        names=proper_nouns(raw),
        types=market_types(text),
    )


def with_rare(prof: Profile, df: Counter) -> Profile:
    return replace(prof, rare=frozenset(t for t in prof.tokens if df[t] <= RARE_DF))


@dataclass
class MatchResult:
    poly: PolyMarket
    kalshi: KalshiMarket
    text_similarity: float
    confidence: float
    flags: list[str] = field(default_factory=list)
    suggest_inverted: bool = False
    resolution_gap_days: float | None = None


def score_pair(
    pm: PolyMarket, km: KalshiMarket, pp: Profile, kp: Profile, cosine: float | None = None
) -> MatchResult:
    flags: list[str] = []

    fuzzy = fuzz.token_set_ratio(pp.text, kp.text) / 100
    text_sim = fuzzy if cosine is None else 0.5 * fuzzy + 0.5 * cosine

    # Numeric thresholds: a disjoint set almost always means a different line/strike.
    if pp.numbers and kp.numbers:
        shared = pp.numbers & kp.numbers
        jaccard = len(shared) / len(pp.numbers | kp.numbers)
        if not shared:
            number_factor = 0.25
            flags.append(f"number_mismatch: PM{sorted(pp.numbers)} vs K{sorted(kp.numbers)}")
        else:
            number_factor = 0.5 + 0.5 * jaccard
            if jaccard < 1:
                flags.append(f"partial_number_overlap: PM{sorted(pp.numbers)} vs K{sorted(kp.numbers)}")
    elif pp.numbers or kp.numbers:
        # Numbers on one side only: often a bracket/margin/date qualifier the other side lacks
        # ("win by 5-10%" vs "win"; "before Jan 20, 2027" vs "before 2027").
        number_factor = 0.6
        side, nums = ("PM", pp.numbers) if pp.numbers else ("K", kp.numbers)
        flags.append(f"numbers_only_in_{side}: {sorted(nums)}")
        if RANGE_RE.search(pp.text if pp.numbers else kp.text):
            number_factor *= 0.5
            flags.append(f"range_bucket_only_in_{side}: a bracketed outcome vs an unbracketed one")
    else:
        number_factor = 1.0

    year_factor = 1.0
    if pp.years and kp.years and not (pp.years & kp.years):
        year_factor = 0.4
        flags.append(f"year_mismatch: PM{sorted(pp.years)} vs K{sorted(kp.years)}")

    gap_days = None
    date_factor = 0.9  # unknown resolution time on one side
    pm_res, km_res = pm.end_date, km.resolution_time
    if pm_res is not None and km_res is not None:
        gap_days = abs((pm_res - km_res).total_seconds()) / 86400
        date_factor = 1.0 if gap_days <= 1 else max(0.3, 1 - (gap_days - 1) / 60)
        if gap_days > 1:
            flags.append(f"cutoff_mismatch: resolution times {gap_days:.1f} days apart")

    source_factor = 0.95
    if pp.sources and kp.sources:
        if pp.sources & kp.sources:
            source_factor = 1.0
        else:
            source_factor = 0.75
            flags.append(f"source_mismatch: PM{sorted(pp.sources)} vs K{sorted(kp.sources)}")

    edge_diff = pp.edge_terms ^ kp.edge_terms
    edge_factor = max(0.85, 1 - 0.03 * len(edge_diff))
    for term in sorted(edge_diff):
        side = "PM" if term in pp.edge_terms else "K"
        flags.append(f"edge_case_only_in_{side}: '{term}'")

    polarity_factor = 1.0
    suggest_inverted = False
    if pp.polarity and kp.polarity and pp.polarity != kp.polarity:
        polarity_factor = 0.5
        suggest_inverted = True
        flags.append("polarity_opposite: one venue says above/over, the other below/under; check inversion")

    # Entities: names/places exclusive to *both* sides means same event, different outcome
    # (or a different state/team). Rare tokens exclusive to one side are a weaker warning.
    entity_factor = 1.0
    only_p = (pp.rare | pp.names) - kp.words
    only_k = (kp.rare | kp.names) - pp.words
    rare_p, rare_k = pp.rare - kp.words, kp.rare - pp.words
    if only_p and only_k:
        entity_factor = 0.35
        flags.append(f"entity_mismatch: PM{sorted(only_p)[:4]} vs K{sorted(only_k)[:4]}")
    elif rare_p or rare_k:
        entity_factor = 0.8
        side, extra = ("PM", rare_p) if rare_p else ("K", rare_k)
        flags.append(f"entity_only_in_{side}: {sorted(extra)[:4]}")

    # "Will A vs B end in a draw?" and "A wins" share every entity but are different outcomes.
    draw_p, draw_k = bool(pp.tokens & Draw_WORDS), bool(kp.tokens & Draw_WORDS)
    if draw_p != draw_k:
        entity_factor *= 0.3
        flags.append("draw_vs_win: only one side is a draw/tie outcome")

    # "Any other team/candidate" buckets are the complement of a named field, not a named outcome.
    other_p, other_k = bool(OTHER_RE.search(pp.text)), bool(OTHER_RE.search(kp.text))
    if other_p != other_k:
        entity_factor *= 0.3
        flags.append("other_bucket: only one side is an 'any other' outcome")

    # Market type: moneyline vs spread vs total, goals vs corners, etc. Any difference is a different bet.
    type_factor = 1.0
    if pp.types != kp.types:
        type_factor = 0.35
        flags.append(f"market_type_mismatch: PM{sorted(pp.types) or ['plain']} vs K{sorted(kp.types) or ['plain']}")

    # A comparative on one side only: exact bucket ("be 2.2%") vs threshold ("above 2.2%"), or
    # touch ("reach") vs close-above semantics.
    if bool(pp.polarity) != bool(kp.polarity) and (pp.numbers & kp.numbers):
        polarity_factor *= 0.5
        flags.append("threshold_vs_exact: only one side says above/below for a shared number")

    # Two-outcome Polymarket markets ("A vs B", YES = A) must pair with Kalshi's "A wins", not "B wins".
    outcome_factor = 1.0
    if pm.yes_outcome.lower() != "yes":
        k_side = normalize(km.yes_sub_title or km.title)
        s_yes = fuzz.partial_ratio(normalize(pm.yes_outcome), k_side)
        s_no = fuzz.partial_ratio(normalize(pm.no_outcome), k_side)
        if s_no > s_yes + 10:
            outcome_factor = 0.3
            suggest_inverted = True
            flags.append(
                f"opposite_outcome: PM YES is '{pm.yes_outcome}' but Kalshi YES is '{km.yes_sub_title or km.title}'"
                " (inversion is only a hedge if a draw is impossible)"
            )

    confidence = (
        text_sim * number_factor * year_factor * date_factor * source_factor * edge_factor * polarity_factor
        * entity_factor * type_factor * outcome_factor
    )
    return MatchResult(
        poly=pm,
        kalshi=km,
        text_similarity=round(text_sim, 4),
        confidence=round(min(max(confidence, 0.0), 1.0), 4),
        flags=flags,
        suggest_inverted=suggest_inverted,
        resolution_gap_days=None if gap_days is None else round(gap_days, 2),
    )


def _top_k_rows(sim, k: int, min_value: float) -> list[list[tuple[int, float]]]:
    """Top-k (column, value) per row of a CSR matrix, above min_value."""
    out = []
    for r in range(sim.shape[0]):
        start, end = sim.indptr[r], sim.indptr[r + 1]
        if start == end:
            out.append([])
            continue
        vals = sim.data[start:end]
        cols = sim.indices[start:end]
        keep = vals >= min_value
        vals, cols = vals[keep], cols[keep]
        if len(vals) > k:
            idx = np.argpartition(-vals, k)[:k]
            vals, cols = vals[idx], cols[idx]
        order = np.argsort(-vals)
        out.append([(int(cols[i]), float(vals[i])) for i in order])
    return out


def match_markets(
    poly: list[PolyMarket],
    kalshi: list[KalshiMarket],
    *,
    min_confidence: float = 0.4,
    top_k: int = RETRIEVAL_TOP_K,
    exclude: set[tuple[str, str]] | None = None,
    reserved_poly: set[str] | None = None,
    reserved_kalshi: set[str] | None = None,
) -> list[MatchResult]:
    """`exclude`: (poly_id, kalshi_ticker) pairs a human rejected; never proposed again.
    `reserved_*`: markets already in a confirmed pair; not re-paired with anything else."""
    exclude = exclude or set()
    if not poly or not kalshi:
        return []

    poly_profiles = [poly_profile(p) for p in poly]
    kalshi_profiles = [kalshi_profile(k) for k in kalshi]
    df = Counter(t for prof in (*poly_profiles, *kalshi_profiles) for t in prof.tokens)
    poly_profiles = [with_rare(p, df) for p in poly_profiles]
    kalshi_profiles = [with_rare(k, df) for k in kalshi_profiles]

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b[a-z][a-z]+\b|\b\d+(?:\.\d+)?\b",
        stop_words=list(STOPWORDS),
        ngram_range=(1, 2),
        sublinear_tf=True,
        # Tokens in >2% of markets ("2026", "win", "points") carry no pairing signal. Absolute count with
        # a floor so small universes (tests, narrow filters) don't prune every token.
        max_df=max(50, int(0.02 * (len(poly) + len(kalshi)))),
        min_df=1,
        dtype=np.float32,
    )
    vectorizer.fit([p.text for p in poly_profiles] + [k.text for k in kalshi_profiles])
    P = vectorizer.transform([p.text for p in poly_profiles])
    KT = vectorizer.transform([k.text for k in kalshi_profiles]).T.tocsr()

    scored: list[MatchResult] = []
    for start in range(0, P.shape[0], CHUNK):
        sim = (P[start : start + CHUNK] @ KT).tocsr()
        for offset, cands in enumerate(_top_k_rows(sim, top_k, RETRIEVAL_MIN_COSINE)):
            i = start + offset
            pm = poly[i]
            for j, cos in cands:
                km = kalshi[j]
                if pm.end_date and km.resolution_time:
                    if abs((pm.end_date - km.resolution_time).total_seconds()) > MAX_RESOLUTION_GAP_DAYS * 86400:
                        continue
                if (pm.market_id, km.ticker) in exclude:
                    continue
                res = score_pair(pm, km, poly_profiles[i], kalshi_profiles[j], cos)
                if res.confidence >= min_confidence:
                    scored.append(res)

    # Greedy one-to-one: highest confidence first, each market used at most once.
    scored.sort(key=lambda r: r.confidence, reverse=True)
    used_poly: set[str] = set(reserved_poly or ())
    used_kalshi: set[str] = set(reserved_kalshi or ())
    matches = []
    for r in scored:
        if r.poly.market_id in used_poly or r.kalshi.ticker in used_kalshi:
            continue
        used_poly.add(r.poly.market_id)
        used_kalshi.add(r.kalshi.ticker)
        matches.append(r)
    log.info("matching: %d candidate pairs scored, %d one-to-one matches", len(scored), len(matches))
    return matches
