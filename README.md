# PolyArb

PolyArb looks for arbitrage between Polymarket and Kalshi. Both sites list a lot of the same questions (elections, sports, economic data), and sometimes you can buy YES on one and NO on the other for less than $1 combined. I wanted to know how often that actually happens once you account for fees and order book depth, and how long those opportunities last before someone takes them.

The scanner matches equivalent markets across the two sites, checks both order books every 60 seconds, and logs anything close to an arb in Postgres. An analytics script then works out how many opportunities there were, how many survived fees, how long they lasted, and what they returned.

It doesn't place any trades. See [Limitations](#limitations) for why.

## Results

I'm running the scanner for a few weeks starting in late September 2026. I'll post the numbers here when that's done. The summary comes from `polyarb analytics`.

## How it works

### The arb

Take a question listed on both sites. If you buy YES on one and NO on the other, exactly one of them pays $1 when the market resolves. So if the two contracts plus fees cost less than $1, the profit is locked in no matter what happens:

```
cost(YES on site A) + cost(NO on site B) + fees  <  $1
```

The scanner checks both directions (Polymarket YES with Kalshi NO, and Polymarket NO with Kalshi YES). Sometimes the two sites word the question in opposite directions, like "above 3%" vs "below 3%". In that case I mark the pair as inverted during review and the hedge becomes YES on both.

This only works if both sites resolve the question the same way, which turned out to be the hard part (see Matching below).

### Fees

Both sites charge taker fees that scale with `p × (1 − p)`, so fees are biggest at 50¢ and close to zero near 0¢ or $1.

| Site | Fee | Where the rate comes from |
|---|---|---|
| Kalshi | `0.07 × multiplier × contracts × p × (1 − p)`, rounded up to the cent per order | Each series has a `fee_multiplier` (some are 0.5) |
| Polymarket | `contracts × rate × p × (1 − p)`, rounded to 5 decimals | Each market has its own rate: 0 for geopolitics and some sports lines, 3% to 7% elsewhere |

The Kalshi rounding matters more than you'd expect. A 2¢ edge on a single contract at 50¢ gets wiped out completely by the rounded-up 2¢ fee. If a Polymarket market says it has fees but doesn't list a rate, I assume 5%. Guessing high means I might miss an arb, but I won't report a fake one.

### Order book depth

The best price usually only covers a few contracts, so the scanner walks both order books together. The first contract pair costs the best ask on each side, and later ones get more expensive as each book runs out at a price level. At every point where either book moves to a new price, it calculates the exact profit with fees, and it keeps the size that makes the most total money. It saves that size along with the capital needed, the profit, the return, and the full profit curve.

Some quirks of the two APIs:

- Kalshi only publishes bids. A NO bid at 40¢ is effectively a YES offer at 60¢, so the YES asks come from the NO bids and the other way around.
- Polymarket lists asks from worst to best, so I re-sort them. I also only count each token's own asks, which might understate depth slightly but never double counts it.
- Orders are whole contracts, and Polymarket's minimum order size is respected.

### Annualized return

`annualized = net return × 365 / days until resolution`

I use whichever site resolves later, since your money is stuck until both markets settle. Days are floored at 1 so a same-day sports market doesn't show a 5000% APR. The analytics also report the return above a risk-free rate (4% by default), since a 2% arb that takes a year to resolve is worse than a Treasury bill.

### Matching

This is the part that took the most work. When I first ran the scanner live, every big "arb" it found was a bad match. Some examples:

- "Will core CPI be 2.2%?" paired with "Will core CPI be above 2.2%?"
- A game's over/under paired with a point spread on the same game
- "Team A vs Team B" (where YES means Team A) paired with "Team B wins"
- "Democrats win 5 House seats in Minnesota" paired with the same question for Georgia
- "Lula wins by 5 to 10%" paired with "Lula wins"

Matching now happens in three steps:

1. **Find candidates.** TF-IDF similarity on the event title and question gives each Polymarket market its 5 closest Kalshi markets. There are about 37k Polymarket markets and 106k Kalshi markets, and this step takes around 35 seconds.
2. **Score them.** Each candidate pair gets checked for anything that could make the two resolve differently: different numbers, years, deadlines, resolution sources, market types, names or places, margin brackets, "any other" buckets, and draw vs win. These multiply into a confidence score from 0 to 1, and every problem leaves a flag explaining what it found.
3. **Review by hand.** `polyarb pairs review` shows both sites' rules side by side and I confirm, reject, or invert each pair. The headline stats only use pairs I've confirmed. Unconfirmed pairs still get logged, and `--all-pairs` includes them.

### Measuring how long arbs last

An episode is a stretch of consecutive scans where the same pair stays profitable after fees. One missed scan doesn't end it, but a gap longer than 3 minutes (the scanner being down) does.

The lifetime runs from the first scan that saw the arb to the first scan where it was gone. Since scans are a minute apart, that's an upper bound with about a minute of precision.

Some arbs are still open when data collection stops, so I don't know when they would have ended. Dropping those would bias the result, so the median lifetime uses a Kaplan-Meier estimate, which counts them as lasting at least as long as I saw them.

I also track how many arbs survive fees. That's the share of pairs whose best prices add up to under $1 that are still profitable after fees and walking the book.

## Code layout

```
polyarb/
  clients/
    http.py         shared session with retries, rate limiter
    polymarket.py   market list and batch order books
    kalshi.py       market list, batch quotes, order books, series fees
  models.py         order book and market types used by both sites
  fees.py           fee formulas for both sites
  arbitrage.py      walks the books, finds the best size, annualizes
  matching.py       candidate search, confidence scoring, one-to-one pairing
  db.py             database tables: market_pairs, scans, observations
  scanner.py        the 60 second scan loop and background re-matching
  analytics.py      episodes, Kaplan-Meier, summary stats
  api.py            read-only FastAPI endpoints for a future dashboard
  display.py, cli.py
tests/
```

Every 60 seconds the scanner:

1. Loads the current pairs.
2. Grabs all the Polymarket order books and all the Kalshi best prices in a few batched requests.
3. Fetches full Kalshi order books only for pairs where the best prices already add up to under $1.
4. Evaluates each pair and saves the results.

Every 15 minutes a background thread re-runs matching to pick up new markets. It takes a few minutes, so it runs separately from the scan loop and shares the same Kalshi rate limit.

Every scan gets logged. Pair results are saved when the pair is within 2% of being an arb, or for every pair if you set `LOG_ALL_PAIRS=true`. Pairs further out than that don't affect any of the stats.

## Setup

```bash
brew install postgresql@16 && brew services start postgresql@16
/opt/homebrew/opt/postgresql@16/bin/createdb polyarb

python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # the defaults work with a local Postgres
polyarb initdb
```

Both sites' market data is public, so you don't need any API keys.

## Usage

```bash
polyarb match                         # build candidate pairs, takes about 5 minutes
polyarb pairs review                  # confirm, reject, or invert pairs
polyarb pairs list --status confirmed
polyarb pairs show 42

polyarb scan --once                   # run a single scan
polyarb scan --min-return 0.01        # keep scanning, show arbs returning 1% or more after fees

polyarb analytics                     # summary using confirmed pairs
polyarb analytics --all-pairs --json
polyarb analytics --since 2026-10-01 --export-episodes episodes.csv

polyarb serve                         # API docs at http://127.0.0.1:8000/docs
pytest
```

Settings live in `.env`. The options are listed in `.env.example`.

## Limitations

- You can't actually trade these as they are. The two sides are on different sites, so prices can move between your first and second order. International Polymarket also blocks US users. The numbers show what was available, not what a real strategy would have made.
- Anything that opens and closes within a minute is invisible to the scanner, so the lifetimes lean toward longer-lasting arbs.
- Even a pair I've confirmed could resolve differently in some edge case. Confirming is my judgment call after reading both sets of rules.
- Some costs aren't included: deposit and withdrawal fees, bridging, having money sitting on two sites, and maker vs taker pricing. Every order is treated as a taker order.
- Polymarket can also fill a NO order against YES bids behind the scenes. I don't count that liquidity, so Polymarket depth is a bit understated.
