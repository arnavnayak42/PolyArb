"""Discovery (re-matching) and the 60-second scan loop."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import func, insert, or_, select
from sqlalchemy.engine import Engine

from polyarb.arbitrage import ArbResult, FeeModel, directions, evaluate_pair
from polyarb.clients.http import RateLimiter
from polyarb.clients.kalshi import KalshiClient
from polyarb.clients.polymarket import PolymarketClient
from polyarb.config import Settings
from polyarb.db import MarketPair, Observation, PairStatus, Scan, as_utc, session_factory, utcnow
from polyarb.matching import MatchResult, match_markets
from polyarb.models import BinaryBook, Level

log = logging.getLogger(__name__)

# Pairs are stored above this confidence so they can be reviewed; scanning uses settings.min_match_confidence.
STORE_MIN_CONFIDENCE = 0.40
KALSHI_BOOK_WORKERS = 8


@dataclass
class Opportunity:
    pair: MarketPair
    result: ArbResult


@dataclass
class ScanReport:
    scan_id: int
    started_at: datetime
    duration_sec: float
    pairs_checked: int
    books_walked: int
    gross_arbs: int
    net_arbs: int
    logged: int
    opportunities: list[Opportunity] = field(default_factory=list)
    error: str | None = None


def _kalshi_top_book(yes_ask: float | None, no_ask: float | None) -> BinaryBook:
    """Placeholder book from top-of-book quotes only; used for pairs that can't possibly be arbs,
    where we log prices but never walk depth."""
    return BinaryBook(
        yes_asks=[Level(yes_ask, math.inf)] if yes_ask else [],
        no_asks=[Level(no_ask, math.inf)] if no_ask else [],
        min_order_size=1.0,
    )


def might_be_gross_arb(pair: MarketPair, poly_book: BinaryBook, k_yes: float | None, k_no: float | None) -> bool:
    for poly_side, kalshi_side in directions(pair.inverted):
        p = poly_book.best(poly_side)
        k = k_yes if kalshi_side == "YES" else k_no
        if p is not None and k is not None and p + k < 1:
            return True
    return False


class Scanner:
    def __init__(
        self,
        settings: Settings,
        engine: Engine,
        poly: PolymarketClient | None = None,
        kalshi: KalshiClient | None = None,
    ):
        self.settings = settings
        self.engine = engine
        self.Session = session_factory(engine)
        self.kalshi_limiter = RateLimiter(settings.kalshi_rps)
        self.poly = poly or PolymarketClient(timeout=settings.http_timeout_sec)
        self.kalshi = kalshi or KalshiClient(timeout=settings.http_timeout_sec, limiter=self.kalshi_limiter)
        self._rematch_thread: threading.Thread | None = None
        self.last_rematch: dict | None = None

    # ---------------------------------------------------------------- discovery

    def rematch(self, poly: PolymarketClient | None = None, kalshi: KalshiClient | None = None) -> dict:
        """Fetch both universes, match, and upsert pairs. Human decisions (status, inversion, notes) persist."""
        poly = poly or self.poly
        kalshi = kalshi or self.kalshi
        t0 = time.monotonic()
        poly_markets = poly.list_markets(
            min_liquidity=self.settings.poly_min_liquidity, max_days=self.settings.max_days_to_resolution
        )
        kalshi_markets = kalshi.list_markets(max_days=self.settings.max_days_to_resolution)
        poly_ids = {m.market_id for m in poly_markets}
        kalshi_ids = {m.ticker for m in kalshi_markets}

        with self.Session() as session:
            existing = {(p.poly_market_id, p.kalshi_ticker): p for p in session.scalars(select(MarketPair))}
            rejected = {k for k, p in existing.items() if p.status == PairStatus.REJECTED}
            confirmed = [p for p in existing.values() if p.status == PairStatus.CONFIRMED]

            matches = match_markets(
                poly_markets,
                kalshi_markets,
                min_confidence=STORE_MIN_CONFIDENCE,
                exclude=rejected,
                reserved_poly={p.poly_market_id for p in confirmed},
                reserved_kalshi={p.kalshi_ticker for p in confirmed},
            )

            now = utcnow()
            created = updated = 0
            for m in matches:
                pair = existing.get((m.poly.market_id, m.kalshi.ticker))
                if pair is None:
                    pair = MarketPair(poly_market_id=m.poly.market_id, kalshi_ticker=m.kalshi.ticker)
                    session.add(pair)
                    existing[(m.poly.market_id, m.kalshi.ticker)] = pair
                    created += 1
                else:
                    updated += 1
                _apply_match(pair, m, now)

            # Confirmed pairs stay live as long as both markets still exist, even if the matcher's
            # scores drifted; the human already vouched for them.
            for p in confirmed:
                if p.poly_market_id in poly_ids and p.kalshi_ticker in kalshi_ids:
                    p.last_seen_at = now

            scannable = [
                p
                for p in existing.values()
                if p.last_seen_at == now and p.kalshi_fee_multiplier is None and self._is_scannable(p)
            ]
            for p in scannable:
                p.kalshi_fee_multiplier = kalshi.series_fee_multiplier(p.kalshi_series)
            session.commit()

        stats = {
            "poly_markets": len(poly_markets),
            "kalshi_markets": len(kalshi_markets),
            "matches": len(matches),
            "created": created,
            "updated": updated,
            "seconds": round(time.monotonic() - t0, 1),
            "finished_at": utcnow(),
        }
        self.last_rematch = stats
        log.info("rematch done: %s", stats)
        return stats

    def rematch_in_background(self) -> bool:
        """Start a rematch thread unless one is running. Uses its own HTTP clients but shares the
        Kalshi rate limiter with the scan loop."""
        if self._rematch_thread and self._rematch_thread.is_alive():
            return False

        def work():
            try:
                self.rematch(
                    poly=PolymarketClient(timeout=self.settings.http_timeout_sec),
                    kalshi=KalshiClient(timeout=self.settings.http_timeout_sec, limiter=self.kalshi_limiter),
                )
            except Exception:
                log.exception("background rematch failed")

        self._rematch_thread = threading.Thread(target=work, name="rematch", daemon=True)
        self._rematch_thread.start()
        return True

    @property
    def rematch_running(self) -> bool:
        return bool(self._rematch_thread and self._rematch_thread.is_alive())

    # ---------------------------------------------------------------- scanning

    def _is_scannable(self, p: MarketPair) -> bool:
        if p.status == PairStatus.REJECTED:
            return False
        return p.status == PairStatus.CONFIRMED or p.resolution_confidence >= self.settings.min_match_confidence

    def active_pairs(self, session) -> list[MarketPair]:
        """Pairs produced (or, if confirmed, re-validated) by the most recent successful rematch.

        Every rematch stamps its pairs with the same `last_seen_at`, so anything older was dropped by
        the matcher or its markets closed. The stale guard stops scanning entirely if rematching has
        been failing for a long time rather than scanning an ever-older universe."""
        now = utcnow()
        latest = session.scalar(select(func.max(MarketPair.last_seen_at)))
        if latest is None:
            return []
        latest = as_utc(latest)
        stale_cutoff = max(
            latest,  # every pair from one rematch shares this exact timestamp
            now - timedelta(seconds=max(6 * self.settings.rematch_interval_sec, 3 * 3600)),
        )
        rows = session.scalars(
            select(MarketPair).where(
                MarketPair.status != PairStatus.REJECTED,
                MarketPair.last_seen_at >= stale_cutoff,
                or_(
                    MarketPair.status == PairStatus.CONFIRMED,
                    MarketPair.resolution_confidence >= self.settings.min_match_confidence,
                ),
            )
        ).all()
        return [p for p in rows if p.resolution_time is None or p.resolution_time > now]

    def scan_once(self) -> ScanReport:
        t0 = time.monotonic()
        started = utcnow()
        with self.Session() as session:
            scan = Scan(started_at=started)
            session.add(scan)
            session.flush()
            report = ScanReport(scan.id, started, 0.0, 0, 0, 0, 0, 0)
            try:
                pairs = self.active_pairs(session)
                report.pairs_checked = len(pairs)
                if pairs:
                    rows = self._evaluate(pairs, scan.id, started, report)
                    if rows:
                        session.execute(insert(Observation), rows)
                    report.logged = len(rows)
            except Exception as exc:  # keep the loop alive; the failed scan is recorded
                log.exception("scan %s failed", scan.id)
                report.error = f"{type(exc).__name__}: {exc}"
                scan.ok = False
                scan.error = report.error[:2000]
            scan.pairs_checked = report.pairs_checked
            scan.books_walked = report.books_walked
            scan.gross_arbs = report.gross_arbs
            scan.net_arbs = report.net_arbs
            scan.observations_logged = report.logged
            scan.finished_at = utcnow()
            session.commit()
        report.duration_sec = round(time.monotonic() - t0, 2)
        report.opportunities.sort(key=lambda o: o.result.net_return or 0, reverse=True)
        return report

    def _evaluate(self, pairs: list[MarketPair], scan_id: int, now: datetime, report: ScanReport) -> list[dict]:
        s = self.settings
        tokens = [t for p in pairs for t in (p.poly_yes_token, p.poly_no_token)]
        poly_books = self.poly.get_books(tokens)
        quotes = self.kalshi.get_quotes([p.kalshi_ticker for p in pairs])

        staged: list[tuple[MarketPair, BinaryBook, tuple[float | None, float | None]]] = []
        need_depth: list[str] = []
        for p in pairs:
            pb = PolymarketClient.binary_book(poly_books, p.poly_yes_token, p.poly_no_token)
            q = quotes.get(p.kalshi_ticker)
            if pb is None or q is None:
                continue
            staged.append((p, pb, q))
            if might_be_gross_arb(p, pb, *q):
                need_depth.append(p.kalshi_ticker)

        kalshi_books: dict[str, BinaryBook] = {}
        if need_depth:
            with ThreadPoolExecutor(max_workers=KALSHI_BOOK_WORKERS) as pool:
                for ticker, book in zip(need_depth, pool.map(self._safe_orderbook, need_depth)):
                    if book is not None:
                        kalshi_books[ticker] = book

        rows: list[dict] = []
        needs_depth = set(need_depth)
        for p, pb, (k_yes, k_no) in staged:
            kb = kalshi_books.get(p.kalshi_ticker)
            if kb is None:
                if p.kalshi_ticker in needs_depth:
                    continue  # depth fetch failed; never walk the infinite-size placeholder
                kb = _kalshi_top_book(k_yes, k_no)
            fees = FeeModel(
                poly_rate=p.poly_fee_rate,
                poly_exponent=p.poly_fee_exponent,
                kalshi_multiplier=p.kalshi_fee_multiplier if p.kalshi_fee_multiplier is not None else 1.0,
                kalshi_base=s.kalshi_base_fee,
            )
            res = evaluate_pair(
                pb,
                kb,
                fees,
                inverted=p.inverted,
                resolution=p.resolution_time,
                now=now,
                min_days=s.min_days_for_annualization,
                risk_free_rate=s.risk_free_rate,
            )
            if res.book_walked:
                report.books_walked += 1
            if res.is_gross_arb:
                report.gross_arbs += 1
            if res.is_net_arb:
                report.net_arbs += 1
                if (res.net_return or 0) >= s.min_return:
                    report.opportunities.append(Opportunity(p, res))

            near = res.gross_return_top is not None and res.gross_return_top > s.log_gross_floor
            if s.log_all_pairs or near or res.is_net_arb:
                rows.append(_observation_row(scan_id, p, pb, kb, res, now))
        return rows

    def _safe_orderbook(self, ticker: str) -> BinaryBook | None:
        try:
            return self.kalshi.get_orderbook(ticker)
        except Exception as exc:
            log.warning("kalshi orderbook %s failed: %s", ticker, exc)
            return None

    def run_forever(self, on_report: Callable[[ScanReport], None] | None = None) -> None:
        s = self.settings
        with self.Session() as session:
            has_pairs = session.scalars(select(MarketPair.id).limit(1)).first() is not None
        if not has_pairs:
            log.info("no pairs yet; running initial rematch in the foreground (takes a few minutes)")
            self.rematch()
        next_rematch = time.monotonic() if has_pairs else time.monotonic() + s.rematch_interval_sec

        while True:
            tick = time.monotonic()
            if tick >= next_rematch and self.rematch_in_background():
                next_rematch = tick + s.rematch_interval_sec
            report = self.scan_once()
            if on_report:
                on_report(report)
            time.sleep(max(0.0, s.scan_interval_sec - (time.monotonic() - tick)))


def _apply_match(pair: MarketPair, m: MatchResult, now: datetime) -> None:
    pm, km = m.poly, m.kalshi
    pair.poly_question = pm.question
    pair.poly_event_title = pm.event_title
    pair.poly_yes_outcome = pm.yes_outcome
    pair.poly_slug = pm.slug
    pair.poly_yes_token = pm.yes_token
    pair.poly_no_token = pm.no_token
    pair.poly_end = pm.end_date
    pair.poly_fee_rate = pm.fee_rate
    pair.poly_fee_exponent = pm.fee_exponent
    pair.poly_rules = pm.rules
    pair.kalshi_series = km.series_ticker
    pair.kalshi_title = km.match_text
    pair.kalshi_resolution = km.resolution_time
    pair.kalshi_category = km.category
    pair.kalshi_rules = km.rules
    pair.text_similarity = m.text_similarity
    pair.resolution_confidence = m.confidence
    pair.resolution_gap_days = m.resolution_gap_days
    pair.flags = m.flags
    pair.suggest_inverted = m.suggest_inverted
    pair.last_seen_at = now


def _finite(x: float | None) -> float | None:
    return x if x is not None and math.isfinite(x) else None


def _observation_row(
    scan_id: int, p: MarketPair, pb: BinaryBook, kb: BinaryBook, res: ArbResult, now: datetime
) -> dict:
    return {
        "scan_id": scan_id,
        "pair_id": p.id,
        "ts": now,
        "direction": res.direction,
        "poly_yes_ask": pb.best("YES"),
        "poly_no_ask": pb.best("NO"),
        "kalshi_yes_ask": kb.best("YES"),
        "kalshi_no_ask": kb.best("NO"),
        "poly_price": res.poly_price,
        "kalshi_price": res.kalshi_price,
        "gross_cost_top": res.gross_cost_top,
        "gross_return_top": _finite(res.gross_return_top),
        "net_return_top": _finite(res.net_return_top),
        "book_walked": res.book_walked,
        "max_size": res.size,
        "capital": res.capital,
        "profit": res.profit,
        "net_return": _finite(res.net_return),
        "annualized_return": _finite(res.annualized_return),
        "excess_annualized_return": _finite(res.excess_annualized_return),
        "days_to_resolution": res.days_to_resolution,
        "resolution_confidence": p.resolution_confidence,
        "pair_status": p.status,
        "is_net_arb": res.is_net_arb,
        "profit_curve": res.curve or None,
    }
