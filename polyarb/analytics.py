"""Research analytics over the scan log.

Definitions (these matter; state them in any writeup):

* Opportunity episode: a maximal run of scans in which the same pair shows a fee-adjusted profitable
  arb (profit > 0 at the profit-maximizing size, net return >= threshold). Up to `gap_tolerance`
  missed scans are bridged, so one flaky scan doesn't split an episode. Scanner downtime (gap
  between scans > 3x the scan interval) always ends an episode.
* Lifetime: time from the first scan that saw the episode to the first scan that didn't. Because
  we sample every ~60s, this is an upper bound with ~1 interval resolution, and arbs that live
  less than one interval are invisible (survivorship bias toward longer-lived arbs).
* Censoring: episodes still alive at the last scan (or right before downtime) have unknown lifetime.
  The median lifetime ("half-life") is estimated with Kaplan-Meier, which uses censored episodes
  correctly instead of dropping them or treating them as ended.
* Survival after fees: of pair-scans (and episodes) with a gross top-of-book arb (combined ask < $1),
  the share that remain profitable after both venues' taker fees and walking the book.
* Annualized return: simple, net_return * 365 / days_to_resolution (floored at MIN_DAYS), using the
  later of the two venues' resolution times.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.engine import Engine

from polyarb.db import MarketPair, Observation, PairStatus, Scan

DOWNTIME_FACTOR = 3.0
SIZE_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
CAPITAL_BUCKETS = [0, 10, 100, 1_000, 10_000, np.inf]
CAPITAL_LABELS = ["<$10", "$10-100", "$100-1k", "$1k-10k", ">$10k"]


@dataclass(frozen=True)
class AnalyticsConfig:
    threshold: float = 0.0
    statuses: tuple[str, ...] = (PairStatus.CONFIRMED,)
    min_confidence: float | None = None
    gap_tolerance: int = 1
    scan_interval_sec: float = 60.0
    since: datetime | None = None
    until: datetime | None = None


# ------------------------------------------------------------------------------------------ loading


def load_scans(engine: Engine, cfg: AnalyticsConfig) -> pd.DataFrame:
    q = select(Scan.id, Scan.started_at, Scan.ok, Scan.pairs_checked).where(Scan.ok.is_(True))
    if cfg.since:
        q = q.where(Scan.started_at >= cfg.since)
    if cfg.until:
        q = q.where(Scan.started_at <= cfg.until)
    with engine.connect() as conn:
        df = pd.read_sql(q.order_by(Scan.started_at), conn)
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    return df


def load_observations(engine: Engine, cfg: AnalyticsConfig) -> pd.DataFrame:
    """Only rows with a gross or net arb are needed for the research metrics."""
    q = (
        select(
            Observation.scan_id,
            Observation.pair_id,
            Observation.ts,
            Observation.direction,
            Observation.poly_price,
            Observation.kalshi_price,
            Observation.gross_return_top,
            Observation.net_return_top,
            Observation.max_size,
            Observation.capital,
            Observation.profit,
            Observation.net_return,
            Observation.annualized_return,
            Observation.excess_annualized_return,
            Observation.days_to_resolution,
            Observation.is_net_arb,
            MarketPair.status,
            MarketPair.resolution_confidence,
            MarketPair.kalshi_category,
            MarketPair.poly_question,
            MarketPair.kalshi_title,
        )
        .join(MarketPair, MarketPair.id == Observation.pair_id)
        .where((Observation.gross_return_top > 0) | Observation.is_net_arb.is_(True))
    )
    if cfg.statuses:
        q = q.where(MarketPair.status.in_(cfg.statuses))
    if cfg.min_confidence is not None:
        q = q.where(MarketPair.resolution_confidence >= cfg.min_confidence)
    if cfg.since:
        q = q.where(Observation.ts >= cfg.since)
    if cfg.until:
        q = q.where(Observation.ts <= cfg.until)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


# ------------------------------------------------------------------------------------------ episodes


def index_scans(scans: pd.DataFrame, scan_interval_sec: float) -> pd.DataFrame:
    """Ordinal position and session id for each successful scan. A new session starts after downtime."""
    scans = scans.sort_values("started_at").reset_index(drop=True)
    scans["k"] = np.arange(len(scans))
    gaps = scans["started_at"].diff().dt.total_seconds()
    scans["session"] = (gaps > DOWNTIME_FACTOR * scan_interval_sec).cumsum()
    return scans


def build_episodes(obs: pd.DataFrame, scans: pd.DataFrame, mask: pd.Series, cfg: AnalyticsConfig) -> pd.DataFrame:
    """Group the pair-scans selected by `mask` into episodes."""
    cols = [
        "pair_id", "start_ts", "end_ts", "n_scans", "lifetime_min", "censored", "peak_net_return",
        "median_net_return", "median_annualized", "median_excess_annualized", "peak_profit",
        "size_at_peak", "capital_at_peak", "days_to_resolution", "category", "any_net",
    ]
    if obs.empty or scans.empty:
        return pd.DataFrame(columns=cols)

    scans = index_scans(scans, cfg.scan_interval_sec)
    ts_by_k = scans["started_at"].to_numpy()
    session_by_k = scans["session"].to_numpy()
    last_k = len(scans) - 1

    sel = obs[mask].merge(scans[["id", "k", "session"]], left_on="scan_id", right_on="id", how="inner")
    if sel.empty:
        return pd.DataFrame(columns=cols)
    sel = sel.sort_values(["pair_id", "k"])

    # New episode when the pair skips more than gap_tolerance scans or crosses a downtime boundary.
    prev_k = sel.groupby("pair_id")["k"].shift()
    prev_session = sel.groupby("pair_id")["session"].shift()
    new_ep = prev_k.isna() | ((sel["k"] - prev_k) > cfg.gap_tolerance + 1) | (sel["session"] != prev_session)
    sel["episode"] = new_ep.cumsum()

    rows = []
    for _, ep in sel.groupby("episode", sort=False):
        first_k, end_k = int(ep["k"].iloc[0]), int(ep["k"].iloc[-1])
        next_k = end_k + 1
        censored = next_k > last_k or session_by_k[next_k] != session_by_k[end_k]
        start_ts = pd.Timestamp(ts_by_k[first_k])
        if censored:
            lifetime = (pd.Timestamp(ts_by_k[end_k]) - start_ts).total_seconds() + cfg.scan_interval_sec
        else:
            lifetime = (pd.Timestamp(ts_by_k[next_k]) - start_ts).total_seconds()
        net_rows = ep[ep["is_net_arb"]]
        peak = net_rows.loc[net_rows["profit"].idxmax()] if not net_rows.empty else ep.iloc[0]
        rows.append(
            {
                "pair_id": int(ep["pair_id"].iloc[0]),
                "start_ts": start_ts,
                "end_ts": pd.Timestamp(ts_by_k[end_k]),
                "n_scans": len(ep),
                "lifetime_min": lifetime / 60,
                "censored": bool(censored),
                "peak_net_return": net_rows["net_return"].max() if not net_rows.empty else np.nan,
                "median_net_return": net_rows["net_return"].median() if not net_rows.empty else np.nan,
                "median_annualized": net_rows["annualized_return"].median() if not net_rows.empty else np.nan,
                "median_excess_annualized": (
                    net_rows["excess_annualized_return"].median() if not net_rows.empty else np.nan
                ),
                "peak_profit": float(peak["profit"]) if not net_rows.empty else 0.0,
                "size_at_peak": int(peak["max_size"]) if not net_rows.empty else 0,
                "capital_at_peak": float(peak["capital"]) if not net_rows.empty else 0.0,
                "days_to_resolution": float(ep["days_to_resolution"].iloc[0]) if ep["days_to_resolution"].notna().any() else np.nan,
                "category": ep["kalshi_category"].iloc[0],
                "any_net": not net_rows.empty,
            }
        )
    return pd.DataFrame(rows, columns=cols)


def net_mask(obs: pd.DataFrame, threshold: float) -> pd.Series:
    return obs["is_net_arb"].astype(bool) & (obs["net_return"].fillna(-np.inf) >= threshold) & (obs["profit"] > 0)


def gross_mask(obs: pd.DataFrame) -> pd.Series:
    return obs["gross_return_top"].fillna(-np.inf) > 0


# ------------------------------------------------------------------------------------------ survival


def kaplan_meier(durations: pd.Series, censored: pd.Series) -> pd.DataFrame:
    """Product-limit survival estimate S(t). `censored` = True means the end wasn't observed."""
    df = pd.DataFrame({"t": durations.astype(float), "event": ~censored.astype(bool)})
    if df.empty:
        return pd.DataFrame(columns=["t", "at_risk", "events", "survival"])
    table = df.groupby("t").agg(events=("event", "sum"), total=("event", "size")).sort_index()
    at_risk = len(df) - table["total"].cumsum().shift(fill_value=0)
    hazard = table["events"] / at_risk
    survival = (1 - hazard).cumprod()
    return pd.DataFrame(
        {"t": table.index, "at_risk": at_risk.to_numpy(), "events": table["events"].to_numpy(), "survival": survival.to_numpy()}
    ).reset_index(drop=True)


def km_median(km: pd.DataFrame) -> float | None:
    """Smallest t with S(t) <= 0.5; None if survival never drops that far (too much censoring)."""
    hit = km[km["survival"] <= 0.5]
    return float(hit["t"].iloc[0]) if not hit.empty else None


def km_survival_at(km: pd.DataFrame, t: float) -> float | None:
    if km.empty:
        return None
    before = km[km["t"] <= t]
    return 1.0 if before.empty else float(before["survival"].iloc[-1])


# ------------------------------------------------------------------------------------------ summary


def _quantiles(s: pd.Series) -> dict:
    s = s.dropna()
    if s.empty:
        return {}
    return {f"p{int(q * 100)}": float(s.quantile(q)) for q in SIZE_QUANTILES} | {"mean": float(s.mean())}


def summarize(obs: pd.DataFrame, scans: pd.DataFrame, cfg: AnalyticsConfig) -> dict:
    out: dict = {
        "config": {
            "threshold": cfg.threshold,
            "statuses": list(cfg.statuses),
            "min_confidence": cfg.min_confidence,
            "gap_tolerance_scans": cfg.gap_tolerance,
            "scan_interval_sec": cfg.scan_interval_sec,
        }
    }
    if scans.empty:
        out["error"] = "no successful scans in range"
        return out

    first, last = scans["started_at"].min(), scans["started_at"].max()
    days_observed = max((last - first).total_seconds() / 86400, 1 / 1440)
    out["coverage"] = {
        "scans": int(len(scans)),
        "first_scan": first.isoformat(),
        "last_scan": last.isoformat(),
        "days_observed": round(days_observed, 3),
        "median_pairs_per_scan": float(scans["pairs_checked"].median()),
    }

    g_mask, n_mask = gross_mask(obs), net_mask(obs, cfg.threshold)
    gross_eps = build_episodes(obs, scans, g_mask, cfg)
    net_eps = build_episodes(obs, scans, n_mask, cfg)

    n_gross_obs, n_net_obs = int(g_mask.sum()), int((g_mask & n_mask).sum())
    out["fee_survival"] = {
        "gross_pair_scans": n_gross_obs,
        "net_pair_scans": n_net_obs,
        "pct_pair_scans_surviving_fees": (100 * n_net_obs / n_gross_obs) if n_gross_obs else None,
        "gross_episodes": int(len(gross_eps)),
        "gross_episodes_with_any_net_scan": int(gross_eps["any_net"].sum()) if len(gross_eps) else 0,
        "pct_episodes_surviving_fees": (100 * gross_eps["any_net"].mean()) if len(gross_eps) else None,
    }

    out["opportunities"] = {
        "net_episodes": int(len(net_eps)),
        "distinct_pairs": int(net_eps["pair_id"].nunique()) if len(net_eps) else 0,
        "episodes_per_day": len(net_eps) / days_observed,
    }

    if len(net_eps):
        km = kaplan_meier(net_eps["lifetime_min"], net_eps["censored"])
        out["lifetime"] = {
            "km_median_half_life_min": km_median(km),
            "naive_median_min": float(net_eps["lifetime_min"].median()),
            "pct_censored": float(100 * net_eps["censored"].mean()),
            "survival_5min": km_survival_at(km, 5),
            "survival_60min": km_survival_at(km, 60),
            "survival_1day": km_survival_at(km, 1440),
            "resolution_note": f"lifetimes are upper bounds at ~{cfg.scan_interval_sec:.0f}s resolution",
        }
        out["returns"] = {
            "median_net_return_pct": 100 * float(net_eps["median_net_return"].median()),
            "mean_annualized_pct": 100 * float(net_eps["median_annualized"].mean()),
            "median_annualized_pct": 100 * float(net_eps["median_annualized"].median()),
            "median_excess_annualized_pct": 100 * float(net_eps["median_excess_annualized"].median()),
            "median_days_to_resolution": float(net_eps["days_to_resolution"].median()),
        }
        buckets = pd.cut(net_eps["capital_at_peak"], CAPITAL_BUCKETS, labels=CAPITAL_LABELS, right=False)
        out["size_distribution"] = {
            "contracts": _quantiles(net_eps["size_at_peak"]),
            "capital_usd": _quantiles(net_eps["capital_at_peak"]),
            "profit_usd": _quantiles(net_eps["peak_profit"]),
            "capital_buckets": {str(k): int(v) for k, v in buckets.value_counts().sort_index().items()},
        }
        by_cat = net_eps.groupby(net_eps["category"].replace("", "Unknown")).agg(
            episodes=("pair_id", "size"),
            median_net_return=("median_net_return", "median"),
            median_lifetime_min=("lifetime_min", "median"),
        )
        out["by_category"] = {
            cat: {
                "episodes": int(r.episodes),
                "median_net_return_pct": 100 * float(r.median_net_return),
                "median_lifetime_min": float(r.median_lifetime_min),
            }
            for cat, r in by_cat.sort_values("episodes", ascending=False).iterrows()
        }
    return out


def run(engine: Engine, cfg: AnalyticsConfig) -> tuple[dict, pd.DataFrame]:
    scans = load_scans(engine, cfg)
    obs = load_observations(engine, cfg)
    summary = summarize(obs, scans, cfg)
    episodes = build_episodes(obs, scans, net_mask(obs, cfg.threshold), cfg) if not obs.empty else pd.DataFrame()
    return summary, episodes
