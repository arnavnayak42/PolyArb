"""Terminal rendering with rich."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from polyarb.db import MarketPair
from polyarb.scanner import ScanReport


def pct(x: float | None, digits: int = 2) -> str:
    return "-" if x is None else f"{100 * x:.{digits}f}%"


def money(x: float | None) -> str:
    return "-" if x is None else f"${x:,.2f}"


def pair_label(p: MarketPair) -> str:
    q = p.poly_question
    if p.poly_yes_outcome and p.poly_yes_outcome.lower() != "yes":
        q = f"{q} [{p.poly_yes_outcome}]"
    return q


def render_scan(report: ScanReport, console: Console, min_return: float) -> None:
    status = (
        f"scan #{report.scan_id} {report.started_at:%H:%M:%S}Z  "
        f"pairs={report.pairs_checked} walked={report.books_walked} "
        f"gross={report.gross_arbs} net={report.net_arbs} logged={report.logged} "
        f"({report.duration_sec:.1f}s)"
    )
    if report.error:
        console.print(f"[red]{status}  ERROR {report.error}[/red]")
        return
    console.print(f"[dim]{status}[/dim]")
    if not report.opportunities:
        return

    table = Table(title=f"Fee-adjusted arbs >= {pct(min_return)}", show_lines=False, expand=True)
    table.add_column("Pair", overflow="fold", ratio=4)
    table.add_column("Legs", no_wrap=True)
    table.add_column("PM", justify="right")
    table.add_column("K", justify="right")
    table.add_column("Gross", justify="right")
    table.add_column("Net", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Capital", justify="right")
    table.add_column("Profit", justify="right")
    table.add_column("Days", justify="right")
    table.add_column("APR", justify="right")
    table.add_column("Conf", justify="right")
    for opp in report.opportunities:
        r, p = opp.result, opp.pair
        conf_style = "green" if p.status == "confirmed" else ("yellow" if p.resolution_confidence >= 0.7 else "red")
        table.add_row(
            f"#{p.id} {pair_label(p)}",
            r.direction,
            f"{r.poly_price:.3f}",
            f"{r.kalshi_price:.3f}",
            pct(r.gross_return_top),
            pct(r.net_return),
            f"{r.size:,}",
            money(r.capital),
            money(r.profit),
            "-" if r.days_to_resolution is None else f"{r.days_to_resolution:.1f}",
            pct(r.annualized_return, 1),
            Text(f"{p.resolution_confidence:.2f}{'*' if p.status == 'confirmed' else ''}", style=conf_style),
        )
    console.print(table)
    console.print("[dim]Conf* = human-confirmed pair. Unconfirmed pairs may be resolution mismatches, not arbs.[/dim]")


def render_pair(p: MarketPair, console: Console, rules_chars: int = 900) -> None:
    flags = "\n".join(f"  • {f}" for f in p.flags) or "  (none)"
    header = (
        f"[bold]Pair #{p.id}[/bold]  status={p.status}  confidence={p.resolution_confidence:.3f}  "
        f"text_sim={p.text_similarity:.3f}  inverted={p.inverted}"
        + ("  [yellow](matcher suggests inversion)[/yellow]" if p.suggest_inverted and not p.inverted else "")
    )
    body = (
        f"{header}\n\n"
        f"[cyan]Polymarket[/cyan] {p.poly_market_id}  https://polymarket.com/market/{p.poly_slug}\n"
        f"  {pair_label(p)}\n  event: {p.poly_event_title}\n  resolves: {p.poly_end}  fee rate: {p.poly_fee_rate}\n\n"
        f"[magenta]Kalshi[/magenta] {p.kalshi_ticker}  ({p.kalshi_category})\n"
        f"  {p.kalshi_title}\n  resolves: {p.kalshi_resolution}  fee multiplier: {p.kalshi_fee_multiplier}\n\n"
        f"[bold]Flags[/bold]\n{flags}\n\n"
        f"[cyan]PM rules[/cyan]\n{(p.poly_rules or '')[:rules_chars]}\n\n"
        f"[magenta]K rules[/magenta]\n{(p.kalshi_rules or '')[:rules_chars]}"
    )
    console.print(Panel(body, expand=True))


def pairs_table(pairs: list[MarketPair], title: str) -> Table:
    table = Table(title=title, expand=True)
    table.add_column("ID", justify="right")
    table.add_column("Status")
    table.add_column("Conf", justify="right")
    table.add_column("Polymarket", overflow="fold", ratio=3)
    table.add_column("Kalshi", overflow="fold", ratio=3)
    table.add_column("Flags", overflow="fold", ratio=2)
    for p in pairs:
        table.add_row(
            str(p.id),
            p.status + (" (inv)" if p.inverted else ""),
            f"{p.resolution_confidence:.2f}",
            pair_label(p),
            p.kalshi_title,
            "; ".join(f.split(":")[0] for f in p.flags) or "-",
        )
    return table


def render_summary(summary: dict, console: Console) -> None:
    if "error" in summary:
        console.print(f"[red]{summary['error']}[/red]")
        return
    cov = summary["coverage"]
    fs = summary["fee_survival"]
    opp = summary["opportunities"]
    lines = [
        f"[bold]Coverage[/bold]  {cov['scans']:,} scans over {cov['days_observed']:.2f} days "
        f"({cov['first_scan'][:16]} → {cov['last_scan'][:16]}), ~{cov['median_pairs_per_scan']:.0f} pairs/scan",
        f"[bold]Scope[/bold]     pair statuses {summary['config']['statuses']}, net threshold {pct(summary['config']['threshold'])}",
        "",
        f"[bold]Opportunities[/bold]  {opp['net_episodes']:,} fee-adjusted arb episodes across "
        f"{opp['distinct_pairs']:,} pairs ({opp['episodes_per_day']:.1f}/day)",
        f"[bold]Fee survival[/bold]   {fs['gross_pair_scans']:,} gross-arb pair-scans → {fs['net_pair_scans']:,} net "
        f"({_fmt(fs['pct_pair_scans_surviving_fees'])}%);  episodes: {fs['gross_episodes']:,} gross → "
        f"{fs['gross_episodes_with_any_net_scan']:,} ever net ({_fmt(fs['pct_episodes_surviving_fees'])}%)",
    ]
    if "lifetime" in summary:
        lt, rt, sz = summary["lifetime"], summary["returns"], summary["size_distribution"]
        km = lt["km_median_half_life_min"]
        lines += [
            f"[bold]Half-life[/bold]      Kaplan-Meier median {('%.1f min' % km) if km is not None else 'not reached (>50% censored/surviving)'}"
            f"; naive median {lt['naive_median_min']:.1f} min; {lt['pct_censored']:.0f}% censored",
            f"                 P(alive at 5m)={_fmt(lt['survival_5min'], 2)}  P(60m)={_fmt(lt['survival_60min'], 2)}  "
            f"P(1d)={_fmt(lt['survival_1day'], 2)}   [dim]{lt['resolution_note']}[/dim]",
            f"[bold]Returns[/bold]        median net {rt['median_net_return_pct']:.2f}% per trade; annualized mean "
            f"{rt['mean_annualized_pct']:.1f}% / median {rt['median_annualized_pct']:.1f}%; "
            f"median excess over risk-free {rt['median_excess_annualized_pct']:.1f}%; "
            f"median {rt['median_days_to_resolution']:.0f} days to resolution",
            f"[bold]Size[/bold]           capital at peak: p25 {money(sz['capital_usd'].get('p25'))}, "
            f"median {money(sz['capital_usd'].get('p50'))}, p90 {money(sz['capital_usd'].get('p90'))};  "
            f"profit median {money(sz['profit_usd'].get('p50'))}, p90 {money(sz['profit_usd'].get('p90'))}",
            "                 buckets: " + ", ".join(f"{k}: {v}" for k, v in sz["capital_buckets"].items()),
        ]
    console.print(Panel("\n".join(lines), title="PolyArb research summary", expand=True))

    if summary.get("by_category"):
        t = Table(title="By category")
        t.add_column("Category")
        t.add_column("Episodes", justify="right")
        t.add_column("Median net", justify="right")
        t.add_column("Median lifetime", justify="right")
        for cat, r in summary["by_category"].items():
            t.add_row(cat, str(r["episodes"]), f"{r['median_net_return_pct']:.2f}%", f"{r['median_lifetime_min']:.1f} min")
        console.print(t)


def _fmt(x: float | None, digits: int = 1) -> str:
    return "-" if x is None else f"{x:.{digits}f}"
