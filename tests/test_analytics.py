from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert

from polyarb import analytics
from polyarb.analytics import AnalyticsConfig, build_episodes, kaplan_meier, km_median, net_mask
from polyarb.db import MarketPair, Observation, PairStatus, Scan

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
CFG = AnalyticsConfig(threshold=0.0, gap_tolerance=1, scan_interval_sec=60)


def scans_df(n, gap_after=None):
    times = []
    t = T0
    for k in range(n):
        times.append(t)
        t += timedelta(seconds=60)
        if gap_after is not None and k == gap_after:
            t += timedelta(minutes=30)  # scanner downtime
    return pd.DataFrame({"id": range(1, n + 1), "started_at": pd.to_datetime(times, utc=True), "ok": True,
                         "pairs_checked": 10})


def obs_df(entries):
    """entries: (pair_id, scan_id, net_return)."""
    rows = []
    for pair_id, scan_id, r in entries:
        rows.append({
            "scan_id": scan_id, "pair_id": pair_id, "ts": T0, "gross_return_top": r + 0.01, "net_return": r,
            "is_net_arb": r > 0, "profit": 10 * r, "max_size": 100, "capital": 95.0, "annualized_return": r * 12,
            "excess_annualized_return": r * 12 - 0.04, "days_to_resolution": 30.0, "kalshi_category": "Politics",
        })
    return pd.DataFrame(rows)


def test_episode_lifetime_and_censoring():
    scans = scans_df(10)
    obs = obs_df([(1, 3, 0.02), (1, 4, 0.02), (1, 5, 0.02), (2, 9, 0.01), (2, 10, 0.01)])
    eps = build_episodes(obs, scans, net_mask(obs, 0.0), CFG).set_index("pair_id")
    # Pair 1 seen at scans 3-5, gone at scan 6: lifetime = t(6) - t(3) = 3 minutes, observed end
    assert eps.loc[1, "lifetime_min"] == pytest.approx(3.0)
    assert not eps.loc[1, "censored"]
    # Pair 2 still alive at the final scan: right-censored
    assert eps.loc[2, "censored"]


def test_gap_tolerance_bridges_single_missed_scan():
    scans = scans_df(10)
    obs = obs_df([(1, 2, 0.02), (1, 4, 0.02)])
    assert len(build_episodes(obs, scans, net_mask(obs, 0.0), CFG)) == 1
    strict = AnalyticsConfig(gap_tolerance=0, scan_interval_sec=60)
    assert len(build_episodes(obs, scans, net_mask(obs, 0.0), strict)) == 2


def test_downtime_splits_and_censors_episode():
    scans = scans_df(10, gap_after=4)  # 30 min outage between scan 5 and 6
    obs = obs_df([(1, 4, 0.02), (1, 5, 0.02), (1, 6, 0.02)])
    eps = build_episodes(obs, scans, net_mask(obs, 0.0), CFG).sort_values("start_ts")
    assert len(eps) == 2
    assert eps.iloc[0]["censored"]  # ended because we stopped looking, not because the arb closed


def test_kaplan_meier_matches_hand_computation():
    km = kaplan_meier(pd.Series([1, 2, 3, 4]), pd.Series([False] * 4))
    assert list(km["survival"]) == pytest.approx([0.75, 0.5, 0.25, 0.0])
    assert km_median(km) == 2

    # Censoring at t=2: S(1)=3/4, S(3)=3/4 * (1 - 1/2) = 0.375
    km = kaplan_meier(pd.Series([1, 2, 3, 4]), pd.Series([False, True, False, False]))
    assert km.set_index("t").loc[3, "survival"] == pytest.approx(0.375)
    assert km_median(km) == 3


def test_km_median_not_reached_when_mostly_censored():
    km = kaplan_meier(pd.Series([5, 10, 15]), pd.Series([False, True, True]))
    assert km_median(km) is None


def _seed(engine):
    """Two confirmed pairs and one candidate over 6 scans."""
    with engine.begin() as conn:
        for pid, status in [(1, PairStatus.CONFIRMED), (2, PairStatus.CONFIRMED), (3, PairStatus.CANDIDATE)]:
            conn.execute(insert(MarketPair).values(
                id=pid, poly_market_id=str(pid), poly_question=f"Q{pid}", poly_yes_token="y", poly_no_token="n",
                kalshi_ticker=f"K{pid}", kalshi_title=f"K{pid}", text_similarity=0.9, resolution_confidence=0.9,
                flags=[], status=status, kalshi_category="Politics",
            ))
        for k in range(1, 7):
            conn.execute(insert(Scan).values(id=k, started_at=T0 + timedelta(minutes=k - 1), ok=True, pairs_checked=3))

        def ob(scan, pair, gross, net, profit):
            return dict(scan_id=scan, pair_id=pair, ts=T0 + timedelta(minutes=scan - 1), direction="PM YES + K NO",
                        gross_return_top=gross, net_return=net, is_net_arb=profit > 0, profit=profit, max_size=100,
                        capital=97.0, annualized_return=(net or 0) * 12, excess_annualized_return=(net or 0) * 12 - 0.04,
                        days_to_resolution=30.0, resolution_confidence=0.9, pair_status="confirmed")

        conn.execute(insert(Observation), [
            ob(1, 1, 0.03, 0.02, 2.0), ob(2, 1, 0.03, 0.02, 2.0),  # pair 1: net arb for 2 scans
            ob(4, 2, 0.01, -0.01, -1.0), ob(5, 2, 0.01, -0.01, -1.0),  # pair 2: gross only, killed by fees
            ob(3, 3, 0.10, 0.09, 9.0),  # candidate pair: excluded from confirmed-only stats
        ])


def test_summary_end_to_end(engine):
    _seed(engine)
    summary, episodes = analytics.run(engine, CFG)
    fs = summary["fee_survival"]
    assert fs["gross_pair_scans"] == 4 and fs["net_pair_scans"] == 2
    assert fs["pct_pair_scans_surviving_fees"] == pytest.approx(50.0)
    assert fs["gross_episodes"] == 2 and fs["gross_episodes_with_any_net_scan"] == 1
    assert summary["opportunities"]["net_episodes"] == 1
    assert summary["lifetime"]["km_median_half_life_min"] == pytest.approx(2.0)
    assert summary["returns"]["median_net_return_pct"] == pytest.approx(2.0)
    assert len(episodes) == 1

    with_candidates = AnalyticsConfig(statuses=(PairStatus.CONFIRMED, PairStatus.CANDIDATE), scan_interval_sec=60)
    assert analytics.run(engine, with_candidates)[0]["opportunities"]["net_episodes"] == 2


def test_api_summary_and_stats(engine):
    from polyarb import api

    _seed(engine)
    api.app.dependency_overrides[api.engine_dep] = lambda: engine
    try:
        client = TestClient(api.app)
        assert client.get("/stats").json()["observations"] == 5
        body = client.get("/analytics/summary").json()
        assert body["opportunities"]["net_episodes"] == 1
        assert client.get("/pairs", params={"status": "candidate"}).json()[0]["id"] == 3
        assert client.get("/pairs/99/history").status_code == 404
    finally:
        api.app.dependency_overrides.clear()
