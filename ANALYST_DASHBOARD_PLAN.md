# Analyst Ratings Dashboard — Plan

## What This Is

A standalone interactive HTML dashboard that aggregates Wall Street analyst data for
a configurable list of stocks — consensus ratings, price targets (avg / min / max /
median vs current price), individual firm upgrades/downgrades, and rating history.
Similar to the analyst panel on TradingView or the Bloomberg consensus screen.

**All data comes from yfinance — no API key or paid subscription required.**

---

## Data Available (confirmed from yfinance)

| Field | yfinance source | Example (NVDA) |
|---|---|---|
| Current price | `ticker.analyst_price_targets['current']` | $199.88 |
| Target high | `ticker.info['targetHighPrice']` | $380.00 |
| Target low | `ticker.info['targetLowPrice']` | $140.00 |
| Target mean | `ticker.info['targetMeanPrice']` | $268.61 |
| Target median | `ticker.info['targetMedianPrice']` | $265.00 |
| Consensus rating | `ticker.info['recommendationKey']` | strong_buy |
| Avg rating score | `ticker.info['recommendationMean']` | 1.28 (1=Strong Buy, 5=Strong Sell) |
| # analysts | `ticker.info['numberOfAnalystOpinions']` | 56 |
| Rating breakdown | `ticker.recommendations` | strongBuy/buy/hold/sell/strongSell counts × 4 months |
| Firm-level history | `ticker.upgrades_downgrades` | Firm, ToGrade, FromGrade, Action, currentPriceTarget, priorPriceTarget, date |

---

## Architecture

### New files

```
analyst_data.py        # Data layer: fetch + clean analyst data for all tickers
analyst_dashboard.py   # Dashboard generator: builds analyst_dashboard.html
analyst_dashboard.html # Output artifact
```

No modifications to tsmom.py, ml_engine.py, or the existing dashboard.

---

## `analyst_data.py` — Data Layer

```python
WATCHLIST = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA",
    "SPY", "QQQ",   # ETFs (fewer analyst opinions, but still present)
]

def fetch_analyst_snapshot(ticker: str) -> dict:
    """
    Pull all analyst data for one ticker.

    Returns
    -------
    dict with keys:
        ticker, current_price, target_high, target_low, target_mean,
        target_median, upside_mean (%), consensus_key, consensus_score,
        n_analysts, strong_buy, buy, hold, sell, strong_sell
    """

def fetch_upgrades_downgrades(ticker: str, days: int = 90) -> pd.DataFrame:
    """
    Recent firm-level rating changes for one ticker.

    Returns DataFrame with columns:
        date, firm, to_grade, from_grade, action,
        current_target, prior_target, target_change ($)
    Filtered to last `days` calendar days.
    """

def fetch_all_snapshots(tickers: list) -> pd.DataFrame:
    """
    Fetch fetch_analyst_snapshot() for each ticker, return as one DataFrame.
    One row per ticker, all metrics as columns.
    Handles yfinance failures gracefully (NaN for missing fields).
    """

def fetch_all_upgrades(tickers: list, days: int = 90) -> pd.DataFrame:
    """
    Fetch fetch_upgrades_downgrades() for each ticker, concatenate.
    Adds 'ticker' column for filtering.
    """
```

---

## `analyst_dashboard.py` — Dashboard Generator

### Dashboard Layout (single page, 4 sections)

---

#### Section 1: Consensus Summary Table (top of page)

HTML table, one row per ticker. Columns:

| Ticker | Price | Consensus | Score | Analysts | Target Low | Target Mean | Target High | Upside % | Buy% | Hold% | Sell% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NVDA | $199.88 | STRONG BUY | 1.29 | 56 | $140 | $268 | $380 | +34.2% | 100% | 4% | 2% |

Color coding:
- Consensus cell: green (strong buy/buy), yellow (hold), red (sell/strong sell)
- Upside % cell: green if > +15%, yellow if 0–15%, red if negative
- Score: 1.0–1.5 = Strong Buy, 1.5–2.5 = Buy, 2.5–3.5 = Hold, 3.5–4.5 = Sell, 4.5–5.0 = Strong Sell

```python
def build_consensus_table_html(snapshots: pd.DataFrame) -> str:
    # Plain HTML table (same pattern as existing stats table — no Plotly)
```

---

#### Section 2: Price Target Visual — "Target Range Bar" Chart

For each ticker: a horizontal bar chart showing:
- Full bar span = target low → target high
- Marker = target mean (average analyst target)
- Vertical line = current price
- Label = upside/downside % from current to mean

Similar to the TradingView "Analyst Price Targets" visual.

```python
def build_price_target_chart(snapshots: pd.DataFrame) -> go.Figure:
    # go.Scatter with error bars, or go.Bar with custom markers
    # One row per ticker, horizontal orientation
    # Color bar green if current < mean (upside), red if current > mean (downside)
```

---

#### Section 3: Rating Breakdown Donut / Bar Charts

For each ticker (or top N tickers): stacked bar or donut showing
Strong Buy / Buy / Hold / Sell / Strong Sell counts.

Two display options:
- **Stacked bar** (all tickers side by side, good for comparison)
- **Donut per ticker** (better for individual detail)

Plan: stacked horizontal bar — one bar per ticker, segments colored:
- Strong Buy: #3FB950 (green)
- Buy: #79C0FF (light blue)
- Hold: #F28C28 (amber)
- Sell: #FF7B72 (salmon)
- Strong Sell: #F85149 (red)

```python
def build_rating_breakdown_chart(snapshots: pd.DataFrame) -> go.Figure:
    # go.Bar traces, barmode='stack', horizontal
    # showlegend=True for segment labels
```

---

#### Section 4: Recent Upgrades / Downgrades Feed

Table of the last 90 days of firm-level rating actions across all tickers.
Sorted by date descending (most recent first).

Columns: Date | Ticker | Firm | Action | From → To | Price Target | Change ($) | Prior Target

Color coding per row:
- Upgrade (action = 'up'): green left border
- Downgrade (action = 'down'): red left border
- Maintain (action = 'main'/'reit'): grey left border

```python
def build_upgrades_table_html(upgrades: pd.DataFrame) -> str:
    # Plain HTML table with CSS row coloring
    # Sortable by column via simple JS (click header to sort)
```

---

### HTML Page Structure

```
<page header: "Analyst Ratings Dashboard | {date}">

[Section 1] CONSENSUS OVERVIEW        <- HTML table
[Section 2] PRICE TARGETS             <- Plotly chart
[Section 3] RATING BREAKDOWN          <- Plotly chart
[Section 4] RECENT UPGRADES/DOWNGRADES <- HTML table (sortable)

<footer: data source, last updated timestamp>
```

Dark theme matching the existing TSMOM dashboard (`COLOR_BG = "#0D1117"` etc.)

---

## Entry Point

```python
# analyst_dashboard.py __main__
if __name__ == "__main__":
    snapshots = fetch_all_snapshots(WATCHLIST)
    upgrades  = fetch_all_upgrades(WATCHLIST, days=90)
    build_analyst_dashboard(snapshots, upgrades, output_path="analyst_dashboard.html")
    print("Open analyst_dashboard.html in browser.")
```

Run time: ~2–5 seconds (one yfinance call per ticker, no ML training).

---

## Implementation Phases

### Phase 1 — Data Layer (`analyst_data.py`)
- `fetch_analyst_snapshot()` for a single ticker
- `fetch_upgrades_downgrades()` for a single ticker
- `fetch_all_snapshots()` + `fetch_all_upgrades()` wrappers
- Test: run on 5 tickers, print DataFrame, verify no NaN explosions

### Phase 2 — HTML Tables
- Consensus summary table with color coding
- Upgrades/downgrades feed with row coloring + JS sort

### Phase 3 — Plotly Charts
- Price target range bar chart
- Rating breakdown stacked bar chart

### Phase 4 — Assembly + Styling
- Wire all sections into `build_analyst_dashboard()`
- Match dark theme to TSMOM dashboard
- Add last-updated timestamp in footer

---

## Files to Create

| File | Purpose |
|---|---|
| `analyst_data.py` | Fetch + clean all analyst data via yfinance |
| `analyst_dashboard.py` | Build and write `analyst_dashboard.html` |
| `analyst_dashboard.html` | Output artifact |

No changes to existing files. Runs fully independently.

---

## Dependencies

All already installed:
```
yfinance>=0.2.40    # analyst data
pandas>=2.0.0       # DataFrames
plotly>=5.20.0      # charts
```

No new packages needed.

---

## Verification

```bash
python analyst_dashboard.py
# open analyst_dashboard.html
```

Check:
1. All tickers in WATCHLIST have data (no blank rows)
2. Current price matches live quote
3. Upside % = (target_mean - current) / current — verify formula
4. Upgrades feed sorted by date descending, last 90 days only
5. Color coding fires correctly (green for upgrades, red for downgrades)
