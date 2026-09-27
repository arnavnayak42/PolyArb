"""Command-line entry point: `polyarb <command>` or `python -m polyarb <command>`."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from datetime import datetime, timezone

from rich.console import Console
from sqlalchemy import func, select

from polyarb.config import get_settings
from polyarb.db import MarketPair, PairStatus, init_db, make_engine, session_factory, utcnow

console = Console()


def _engine():
    settings = get_settings()
    engine = make_engine(settings.database_url)
    init_db(engine)
    return settings, engine


def cmd_initdb(args) -> None:
    settings, _ = _engine()
    console.print(f"[green]schema ready[/green] at {settings.database_url}")


def cmd_match(args) -> None:
    from polyarb.scanner import Scanner

    settings, engine = _engine()
    console.print("Fetching both universes and matching (a few minutes; Polymarket pagination is slow)...")
    scanner = Scanner(settings, engine)
    stats = scanner.rematch()
    console.print(stats)
    with session_factory(engine)() as session:
        n_scan = len(scanner.active_pairs(session))
    console.print(
        f"{n_scan} pairs will be scanned (confidence >= {settings.min_match_confidence} or confirmed). "
        "Review them with [bold]polyarb pairs review[/bold]."
    )


def _query_pairs(
    session, status: str | None, min_conf: float | None, limit: int, order: str = "confidence", current_only: bool = False
):
    q = select(MarketPair)
    if current_only:
        # Only pairs produced by the latest rematch; older ones were dropped by the matcher.
        latest = session.scalar(select(func.max(MarketPair.last_seen_at)))
        if latest is not None:
            q = q.where(MarketPair.last_seen_at >= latest)
    if status:
        q = q.where(MarketPair.status == status)
    if min_conf is not None:
        q = q.where(MarketPair.resolution_confidence >= min_conf)
    q = q.order_by(MarketPair.resolution_confidence.desc() if order == "confidence" else MarketPair.id)
    return list(session.scalars(q.limit(limit)))


def cmd_pairs(args) -> None:
    from polyarb.display import pairs_table, render_pair

    _, engine = _engine()
    Session = session_factory(engine)
    with Session() as session:
        if args.pairs_cmd == "list":
            pairs = _query_pairs(session, args.status, args.min_confidence, args.limit)
            console.print(pairs_table(pairs, f"{len(pairs)} pairs"))
        elif args.pairs_cmd == "show":
            for pid in args.ids:
                if (p := session.get(MarketPair, pid)) is None:
                    console.print(f"[red]no pair {pid}[/red]")
                else:
                    render_pair(p, console)
        elif args.pairs_cmd in ("confirm", "reject", "reset"):
            status = {"confirm": PairStatus.CONFIRMED, "reject": PairStatus.REJECTED, "reset": PairStatus.CANDIDATE}[
                args.pairs_cmd
            ]
            for pid in args.ids:
                if (p := session.get(MarketPair, pid)) is None:
                    console.print(f"[red]no pair {pid}[/red]")
                    continue
                p.status = status
                if args.note:
                    p.review_note = args.note
                console.print(f"pair {pid} → {status}")
            session.commit()
        elif args.pairs_cmd == "invert":
            for pid in args.ids:
                if (p := session.get(MarketPair, pid)) is not None:
                    p.inverted = not p.inverted
                    console.print(f"pair {pid} inverted={p.inverted}")
            session.commit()
        elif args.pairs_cmd == "review":
            review_loop(session, args)


def review_loop(session, args) -> None:
    from polyarb.display import render_pair

    pairs = _query_pairs(session, PairStatus.CANDIDATE, args.min_confidence, args.limit, current_only=True)
    if not pairs:
        console.print("No candidate pairs to review.")
        return
    console.print(
        f"Reviewing {len(pairs)} candidates, highest confidence first. Read both rule texts: the question is "
        "whether every real-world outcome resolves the same way on both venues.\n"
        "[bold]c[/bold]onfirm  [bold]r[/bold]eject  [bold]i[/bold]nvert+confirm (YES here = NO there)  "
        "[bold]s[/bold]kip  [bold]q[/bold]uit"
    )
    done = 0
    for p in pairs:
        render_pair(p, console)
        choice = console.input("[bold]> [/bold]").strip().lower()[:1]
        if choice == "q":
            break
        if choice == "c":
            p.status = PairStatus.CONFIRMED
        elif choice == "r":
            p.status = PairStatus.REJECTED
            p.review_note = console.input("reason (optional): ").strip()
        elif choice == "i":
            p.status = PairStatus.CONFIRMED
            p.inverted = True
        else:
            continue
        p.updated_at = utcnow()
        session.commit()
        done += 1
    console.print(f"Reviewed {done} pairs.")


def cmd_scan(args) -> None:
    from polyarb.display import render_scan
    from polyarb.scanner import Scanner

    settings, engine = _engine()
    overrides = {}
    if args.min_return is not None:
        overrides["min_return"] = args.min_return
    if args.interval is not None:
        overrides["scan_interval_sec"] = args.interval
    settings = dataclasses.replace(settings, **overrides)
    scanner = Scanner(settings, engine)

    def show(report):
        render_scan(report, console, settings.min_return)

    if args.once:
        show(scanner.scan_once())
        return
    console.print(
        f"Scanning every {settings.scan_interval_sec}s, re-matching every {settings.rematch_interval_sec}s "
        f"in the background. Showing arbs with fee-adjusted return >= {settings.min_return:.2%}. Ctrl-C to stop."
    )
    try:
        scanner.run_forever(on_report=show)
    except KeyboardInterrupt:
        console.print("stopped")


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def cmd_analytics(args) -> None:
    from polyarb import analytics
    from polyarb.display import render_summary

    settings, engine = _engine()
    statuses: tuple[str, ...]
    if args.all_pairs:
        statuses = (PairStatus.CONFIRMED, PairStatus.CANDIDATE)
    else:
        statuses = (PairStatus.CONFIRMED,)
    cfg = analytics.AnalyticsConfig(
        threshold=args.threshold,
        statuses=statuses,
        min_confidence=args.min_confidence,
        gap_tolerance=args.gap_tolerance,
        scan_interval_sec=settings.scan_interval_sec,
        since=_parse_dt(args.since),
        until=_parse_dt(args.until),
    )
    summary, episodes = analytics.run(engine, cfg)
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        render_summary(summary, console)
        if not args.all_pairs:
            console.print(
                "[dim]Confirmed pairs only (the credible numbers). Add --all-pairs to include unreviewed candidates.[/dim]"
            )
    if args.export_episodes:
        episodes.to_csv(args.export_episodes, index=False)
        console.print(f"wrote {len(episodes)} episodes to {args.export_episodes}")


def cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("polyarb.api:app", host=args.host, port=args.port, reload=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="polyarb", description="Polymarket x Kalshi arbitrage scanner")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("initdb", help="create tables").set_defaults(func=cmd_initdb)
    sub.add_parser("match", help="fetch markets from both venues and (re)build candidate pairs").set_defaults(
        func=cmd_match
    )

    pairs = sub.add_parser("pairs", help="inspect and review matched pairs")
    pairs.set_defaults(func=cmd_pairs)
    psub = pairs.add_subparsers(dest="pairs_cmd", required=True)
    pl = psub.add_parser("list")
    pl.add_argument("--status", choices=[PairStatus.CANDIDATE, PairStatus.CONFIRMED, PairStatus.REJECTED])
    pl.add_argument("--min-confidence", type=float)
    pl.add_argument("--limit", type=int, default=50)
    pr = psub.add_parser("review", help="interactive confirm/reject loop over candidates")
    pr.add_argument("--min-confidence", type=float, default=0.55)
    pr.add_argument("--limit", type=int, default=100)
    for name in ("show", "confirm", "reject", "reset", "invert"):
        sp = psub.add_parser(name)
        sp.add_argument("ids", type=int, nargs="+")
        if name in ("confirm", "reject", "reset"):
            sp.add_argument("--note", default="")

    scan = sub.add_parser("scan", help="run the scanner (every 60s by default)")
    scan.add_argument("--once", action="store_true", help="single scan, then exit")
    scan.add_argument("--min-return", type=float, help="display threshold, e.g. 0.01 for 1%%")
    scan.add_argument("--interval", type=int, help="seconds between scans")
    scan.set_defaults(func=cmd_scan)

    an = sub.add_parser("analytics", help="research summary over the scan log")
    an.add_argument("--threshold", type=float, default=0.0, help="min fee-adjusted return to count as an arb")
    an.add_argument("--all-pairs", action="store_true", help="include unreviewed candidate pairs")
    an.add_argument("--min-confidence", type=float)
    an.add_argument("--gap-tolerance", type=int, default=1, help="missed scans bridged within an episode")
    an.add_argument("--since", help="ISO datetime (UTC if no tz)")
    an.add_argument("--until")
    an.add_argument("--json", action="store_true")
    an.add_argument("--export-episodes", metavar="CSV")
    an.set_defaults(func=cmd_analytics)

    sv = sub.add_parser("serve", help="FastAPI server")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
