"""Read-only FastAPI service over the scan log (the backend for a v2 React dashboard)."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from polyarb import analytics
from polyarb.config import Settings, get_settings
from polyarb.db import MarketPair, Observation, PairStatus, Scan, init_db, make_engine, session_factory

app = FastAPI(title="PolyArb", version="0.1.0", description="Polymarket x Kalshi arbitrage research API")


@lru_cache
def _engine() -> Engine:
    engine = make_engine(get_settings().database_url)
    init_db(engine)
    return engine


def engine_dep() -> Engine:
    return _engine()


def _pair_dict(p: MarketPair) -> dict:
    return {
        "id": p.id,
        "status": p.status,
        "inverted": p.inverted,
        "resolution_confidence": p.resolution_confidence,
        "flags": p.flags,
        "polymarket": {"id": p.poly_market_id, "question": p.poly_question, "yes_outcome": p.poly_yes_outcome,
                       "slug": p.poly_slug, "resolves": p.poly_end, "fee_rate": p.poly_fee_rate},
        "kalshi": {"ticker": p.kalshi_ticker, "title": p.kalshi_title, "resolves": p.kalshi_resolution,
                   "fee_multiplier": p.kalshi_fee_multiplier, "category": p.kalshi_category},
    }


@app.get("/health")
def health(engine: Engine = Depends(engine_dep)) -> dict:
    with session_factory(engine)() as s:
        last = s.scalars(select(Scan).order_by(Scan.id.desc()).limit(1)).first()
        return {"ok": True, "last_scan": None if last is None else {"id": last.id, "at": last.started_at, "ok": last.ok}}


@app.get("/opportunities/latest")
def latest_opportunities(
    min_return: float = Query(0.0, description="minimum fee-adjusted return"),
    confirmed_only: bool = False,
    engine: Engine = Depends(engine_dep),
) -> dict:
    with session_factory(engine)() as s:
        scan = s.scalars(select(Scan).where(Scan.ok.is_(True)).order_by(Scan.id.desc()).limit(1)).first()
        if scan is None:
            return {"scan": None, "opportunities": []}
        q = (
            select(Observation, MarketPair)
            .join(MarketPair, MarketPair.id == Observation.pair_id)
            .where(Observation.scan_id == scan.id, Observation.is_net_arb.is_(True), Observation.net_return >= min_return)
            .order_by(Observation.net_return.desc())
        )
        if confirmed_only:
            q = q.where(MarketPair.status == PairStatus.CONFIRMED)
        rows = s.execute(q).all()
        return {
            "scan": {"id": scan.id, "at": scan.started_at, "pairs_checked": scan.pairs_checked},
            "opportunities": [
                {
                    "pair": _pair_dict(p),
                    "direction": o.direction,
                    "poly_price": o.poly_price,
                    "kalshi_price": o.kalshi_price,
                    "gross_return_top": o.gross_return_top,
                    "net_return": o.net_return,
                    "annualized_return": o.annualized_return,
                    "excess_annualized_return": o.excess_annualized_return,
                    "days_to_resolution": o.days_to_resolution,
                    "max_size": o.max_size,
                    "capital": o.capital,
                    "profit": o.profit,
                    "profit_curve": o.profit_curve,
                }
                for o, p in rows
            ],
        }


@app.get("/pairs")
def list_pairs(
    status: str | None = None,
    min_confidence: float | None = None,
    limit: int = Query(100, le=1000),
    engine: Engine = Depends(engine_dep),
) -> list[dict]:
    with session_factory(engine)() as s:
        q = select(MarketPair).order_by(MarketPair.resolution_confidence.desc()).limit(limit)
        if status:
            q = q.where(MarketPair.status == status)
        if min_confidence is not None:
            q = q.where(MarketPair.resolution_confidence >= min_confidence)
        return [_pair_dict(p) for p in s.scalars(q)]


@app.get("/pairs/{pair_id}/history")
def pair_history(pair_id: int, limit: int = Query(1000, le=10000), engine: Engine = Depends(engine_dep)) -> dict:
    with session_factory(engine)() as s:
        p = s.get(MarketPair, pair_id)
        if p is None:
            raise HTTPException(404, "pair not found")
        obs = s.scalars(
            select(Observation).where(Observation.pair_id == pair_id).order_by(Observation.ts.desc()).limit(limit)
        ).all()
        return {
            "pair": _pair_dict(p),
            "observations": [
                {"ts": o.ts, "gross_return_top": o.gross_return_top, "net_return": o.net_return,
                 "max_size": o.max_size, "profit": o.profit, "is_net_arb": o.is_net_arb}
                for o in obs
            ],
        }


@app.get("/scans")
def scans(limit: int = Query(50, le=1000), engine: Engine = Depends(engine_dep)) -> list[dict]:
    with session_factory(engine)() as s:
        rows = s.scalars(select(Scan).order_by(Scan.id.desc()).limit(limit)).all()
        return [
            {"id": r.id, "started_at": r.started_at, "finished_at": r.finished_at, "ok": r.ok, "error": r.error,
             "pairs_checked": r.pairs_checked, "gross_arbs": r.gross_arbs, "net_arbs": r.net_arbs}
            for r in rows
        ]


@app.get("/analytics/summary")
def analytics_summary(
    threshold: float = 0.0,
    include_candidates: bool = False,
    min_confidence: float | None = None,
    engine: Engine = Depends(engine_dep),
    settings: Settings = Depends(get_settings),
) -> dict:
    statuses = (PairStatus.CONFIRMED, PairStatus.CANDIDATE) if include_candidates else (PairStatus.CONFIRMED,)
    cfg = analytics.AnalyticsConfig(
        threshold=threshold,
        statuses=statuses,
        min_confidence=min_confidence,
        scan_interval_sec=settings.scan_interval_sec,
    )
    summary, _ = analytics.run(engine, cfg)
    return summary


@app.get("/stats")
def stats(engine: Engine = Depends(engine_dep)) -> dict:
    with session_factory(engine)() as s:
        by_status = dict(s.execute(select(MarketPair.status, func.count()).group_by(MarketPair.status)).all())
        return {
            "pairs": by_status,
            "scans": s.scalar(select(func.count()).select_from(Scan)),
            "observations": s.scalar(select(func.count()).select_from(Observation)),
        }
