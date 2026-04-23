"""
Portfolio Builder Engine
========================
Powers the Portfolio Builder tab in the analyst dashboard.

Given a weekly deposit amount and starting date, detects the current market
regime and constructs an optimal DCA allocation using:
  - Market regime (SPY MA200, realized vol, momentum)
  - Analyst ML sentiment overlay (tilts individual stock weights)
  - Regime-calibrated Monte Carlo growth projection (500 sims, 10 years)

Entry point: get_portfolio_summary(deposit=30.0, analyst_scores=None)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from datetime import date, timedelta

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

START_DATE     = date(2026, 4, 23)
WEEKLY_DEPOSIT = 30.0
PROJECTION_YEARS = 10
N_SIMS = 500

# Regime-based base allocations  {ticker: weight}
BASE_ALLOC = {
    "bull_low_vol": {
        "QQQ":  0.35,
        "SPY":  0.20,
        "NVDA": 0.15,
        "MSFT": 0.10,
        "AAPL": 0.10,
        "GLD":  0.05,
        "AMZN": 0.05,
    },
    "bull_high_vol": {
        "SPY":  0.30,
        "QQQ":  0.20,
        "GLD":  0.15,
        "MSFT": 0.15,
        "AAPL": 0.10,
        "TLT":  0.10,
    },
    "bear_low_vol": {
        "TLT":  0.30,
        "GLD":  0.25,
        "SPY":  0.20,
        "MSFT": 0.15,
        "AAPL": 0.10,
    },
    "crisis": {
        "GLD":  0.35,
        "TLT":  0.30,
        "SPY":  0.15,
        "MSFT": 0.10,
        "AAPL": 0.10,
    },
}

REGIME_LABELS = {
    "bull_low_vol":  "Bull Market / Low Volatility",
    "bull_high_vol": "Bull Market / Elevated Volatility",
    "bear_low_vol":  "Bear Market / Low Volatility",
    "crisis":        "Crisis / High Volatility",
}

REGIME_COLORS = {
    "bull_low_vol":  "#3FB950",
    "bull_high_vol": "#F28C28",
    "bear_low_vol":  "#FF7B72",
    "crisis":        "#F85149",
}

REGIME_DESCRIPTIONS = {
    "bull_low_vol":  "SPY is above its 200-day moving average and market volatility is calm. Favoring growth-oriented tech ETFs and high-conviction momentum stocks.",
    "bull_high_vol": "SPY is above its 200-day moving average but volatility is elevated — possibly a correction within an uptrend. Balancing growth with defensive assets.",
    "bear_low_vol":  "SPY is below its 200-day moving average, signaling a downtrend. Rotating into bonds and gold while maintaining minimal equity exposure via quality names.",
    "crisis":        "SPY is below its 200-day moving average and volatility is high — a stress regime. Capital preservation is the priority: gold and long-duration Treasuries dominate.",
}

# Regime return parameters (annualized, calibrated to historical data)
REGIME_PARAMS = {
    "bull_low_vol":  {"mu": 0.18, "sigma": 0.12},
    "bull_high_vol": {"mu": 0.10, "sigma": 0.20},
    "bear_low_vol":  {"mu": 0.02, "sigma": 0.15},
    "crisis":        {"mu": -0.05, "sigma": 0.35},
}

# Individual stocks eligible for analyst overlay (not ETFs)
OVERLAY_ELIGIBLE = {"NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "AMD", "PLTR", "CRM"}

TICKER_NAMES = {
    "QQQ": "Invesco QQQ (Nasdaq-100 ETF)",
    "SPY": "SPDR S&P 500 ETF",
    "GLD": "SPDR Gold Trust ETF",
    "TLT": "iShares 20+ Year Treasury ETF",
    "NVDA": "NVIDIA",
    "MSFT": "Microsoft",
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet (Google)",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
    "AMD": "AMD",
}


# ─────────────────────────────────────────────────────────────────────────────
# REGIME DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_market_regime() -> dict:
    """Detect current market regime using SPY price data."""
    spy = yf.Ticker("SPY")
    hist = spy.history(period="1y")

    if hist.empty or len(hist) < 60:
        return {
            "regime": "bull_low_vol",
            "spy_price": None,
            "spy_ma200": None,
            "spy_ma50": None,
            "vol_20d": None,
            "mom_60d": None,
            "error": "Insufficient data",
        }

    closes = hist["Close"]
    spy_price = float(closes.iloc[-1])
    ma200 = float(closes.rolling(min(200, len(closes))).mean().iloc[-1])
    ma50  = float(closes.rolling(min(50,  len(closes))).mean().iloc[-1])

    log_ret  = np.log(closes / closes.shift(1)).dropna()
    vol_20d  = float(log_ret.iloc[-20:].std() * np.sqrt(252) * 100)
    mom_60d  = float((closes.iloc[-1] / closes.iloc[-61] - 1) * 100) if len(closes) >= 61 else 0.0

    above_ma200 = spy_price > ma200
    golden_cross = ma50 > ma200

    if above_ma200 and vol_20d < 18:
        regime = "bull_low_vol"
    elif above_ma200 and vol_20d >= 18:
        regime = "bull_high_vol"
    elif not above_ma200 and vol_20d < 25:
        regime = "bear_low_vol"
    else:
        regime = "crisis"

    return {
        "regime":       regime,
        "spy_price":    round(spy_price, 2),
        "spy_ma200":    round(ma200, 2),
        "spy_ma50":     round(ma50, 2),
        "vol_20d":      round(vol_20d, 1),
        "mom_60d":      round(mom_60d, 1),
        "golden_cross": golden_cross,
        "above_ma200":  above_ma200,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRICES
# ─────────────────────────────────────────────────────────────────────────────

def fetch_current_prices(tickers: list) -> dict:
    """Fetch latest close prices for a list of tickers."""
    prices = {}
    for tk in tickers:
        try:
            info = yf.Ticker(tk).info
            p = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
            prices[tk] = float(p) if p else None
        except Exception:
            prices[tk] = None
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# ANALYST OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

def _analyst_overlay(regime: str, analyst_scores: dict) -> dict:
    """
    Tilt weights based on ML sentiment scores.
    Score > 70 → +5% to that ticker (capped by eligibility + available budget).
    Score < 40 → -5%.
    Only applied to individual stocks, not ETFs.
    """
    base = dict(BASE_ALLOC[regime])

    if not analyst_scores:
        return base

    for ticker, score in analyst_scores.items():
        if ticker not in base or ticker not in OVERLAY_ELIGIBLE:
            continue
        if score > 70:
            delta = min(0.05, score / 100 * 0.08)
            base[ticker] = base.get(ticker, 0) + delta
        elif score < 40:
            delta = min(0.05, (100 - score) / 100 * 0.08)
            base[ticker] = max(0.0, base.get(ticker, 0) - delta)

    total = sum(base.values())
    if total > 0:
        base = {k: v / total for k, v in base.items() if v > 0}

    return base


# ─────────────────────────────────────────────────────────────────────────────
# ALLOCATION
# ─────────────────────────────────────────────────────────────────────────────

def build_allocation(regime: str, deposit: float, analyst_scores: dict,
                     prices: dict) -> list:
    """
    Returns a list of dicts describing this week's buy orders.
    Each entry: {ticker, name, weight, dollar, price, shares, is_etf, reason}
    """
    weights = _analyst_overlay(regime, analyst_scores)
    rows = []
    for ticker, weight in sorted(weights.items(), key=lambda x: -x[1]):
        dollar  = round(weight * deposit, 2)
        price   = prices.get(ticker)
        shares  = round(dollar / price, 6) if price and price > 0 else None
        is_etf  = ticker not in OVERLAY_ELIGIBLE
        score   = analyst_scores.get(ticker) if analyst_scores else None

        if score is not None and not is_etf:
            if score > 70:
                reason = f"Overweighted — ML sentiment {score:.0f}/100 (bullish)"
            elif score < 40:
                reason = f"Underweighted — ML sentiment {score:.0f}/100 (bearish)"
            else:
                reason = f"Base weight — ML sentiment {score:.0f}/100 (neutral)"
        elif is_etf:
            reason = "Core ETF — regime-driven base allocation"
        else:
            reason = "Base weight — no analyst score available"

        rows.append({
            "ticker":  ticker,
            "name":    TICKER_NAMES.get(ticker, ticker),
            "weight":  round(weight * 100, 1),
            "dollar":  dollar,
            "price":   round(price, 2) if price else None,
            "shares":  shares,
            "is_etf":  is_etf,
            "reason":  reason,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO PROJECTION
# ─────────────────────────────────────────────────────────────────────────────

def project_growth_monte_carlo(deposit: float, regime: str,
                                years: int = PROJECTION_YEARS,
                                n_sims: int = N_SIMS) -> dict:
    """
    Simulate portfolio growth with weekly DCA deposits.
    Returns percentile paths: p10, p25, median, p75, p90.
    Also includes a no-growth baseline (total deposits only).
    """
    params     = REGIME_PARAMS[regime]
    mu_w       = params["mu"]    / 52          # weekly mean log return
    sigma_w    = params["sigma"] / np.sqrt(52) # weekly sigma
    n_weeks    = years * 52

    rng = np.random.default_rng(42)
    paths = np.zeros((n_sims, n_weeks + 1))

    for s in range(n_sims):
        nav = 0.0
        weekly_returns = rng.normal(mu_w, sigma_w, n_weeks)
        for w in range(n_weeks):
            nav = (nav + deposit) * (1 + weekly_returns[w])
            paths[s, w + 1] = nav

    # Build weekly date axis (every 7 days from today)
    today  = date.today()
    dates  = [(today + timedelta(weeks=w)).isoformat() for w in range(n_weeks + 1)]
    total_deposited = [round(deposit * w, 2) for w in range(n_weeks + 1)]

    def pct(q): return [round(float(v), 2) for v in np.percentile(paths, q, axis=0)]

    return {
        "dates":            dates,
        "total_deposited":  total_deposited,
        "p10":              pct(10),
        "p25":              pct(25),
        "median":           pct(50),
        "p75":              pct(75),
        "p90":              pct(90),
        "final_p10":        round(float(np.percentile(paths[:, -1], 10)), 2),
        "final_p25":        round(float(np.percentile(paths[:, -1], 25)), 2),
        "final_median":     round(float(np.percentile(paths[:, -1], 50)), 2),
        "final_p75":        round(float(np.percentile(paths[:, -1], 75)), 2),
        "final_p90":        round(float(np.percentile(paths[:, -1], 90)), 2),
        "total_deposited_final": round(deposit * n_weeks, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MILESTONES
# ─────────────────────────────────────────────────────────────────────────────

def _compute_milestones(deposit: float, regime: str) -> list:
    params  = REGIME_PARAMS[regime]
    mu_w    = params["mu"]    / 52
    sigma_w = params["sigma"] / np.sqrt(52)
    rng     = np.random.default_rng(42)

    milestones = []
    for label, target in [("$500", 500), ("$1K", 1000), ("$5K", 5000),
                           ("$10K", 10000), ("$25K", 25000)]:
        nav = 0.0
        weeks_hit = None
        for w in range(52 * 30):
            r   = rng.normal(mu_w, sigma_w)
            nav = (nav + deposit) * (1 + r)
            if nav >= target:
                weeks_hit = w + 1
                break
        if weeks_hit:
            years_f  = weeks_hit / 52
            hit_date = (date.today() + timedelta(weeks=weeks_hit)).isoformat()
            milestones.append({
                "target":    label,
                "weeks":     weeks_hit,
                "years":     round(years_f, 1),
                "date":      hit_date,
                "deposited": round(deposit * weeks_hit, 2),
            })
        else:
            milestones.append({"target": label, "weeks": None, "years": None,
                                "date": None, "deposited": None})
    return milestones


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def get_portfolio_summary(deposit: float = WEEKLY_DEPOSIT,
                          analyst_scores: dict = None) -> dict:
    """
    Full portfolio summary used by the /api/portfolio endpoint.

    Parameters
    ----------
    deposit        : weekly DCA amount in USD
    analyst_scores : {ticker: score(0-100)} from analyst_data ML pipeline

    Returns
    -------
    dict with regime, indicators, allocation, projection, milestones
    """
    regime_info = detect_market_regime()
    regime      = regime_info["regime"]
    tickers     = list(BASE_ALLOC[regime].keys())
    prices      = fetch_current_prices(tickers)

    allocation  = build_allocation(regime, deposit, analyst_scores or {}, prices)
    projection  = project_growth_monte_carlo(deposit, regime)
    milestones  = _compute_milestones(deposit, regime)

    weeks_since_start = max(0, (date.today() - START_DATE).days // 7)
    total_deposited   = round(deposit * weeks_since_start, 2)

    return {
        "ok":            True,
        "deposit":       deposit,
        "start_date":    START_DATE.isoformat(),
        "today":         date.today().isoformat(),
        "weeks_running": weeks_since_start,
        "total_deposited": total_deposited,
        "regime":        regime,
        "regime_label":  REGIME_LABELS[regime],
        "regime_color":  REGIME_COLORS[regime],
        "regime_desc":   REGIME_DESCRIPTIONS[regime],
        "indicators":    regime_info,
        "allocation":    allocation,
        "projection":    projection,
        "milestones":    milestones,
    }


if __name__ == "__main__":
    result = get_portfolio_summary()
    r = result
    print(f"Regime:    {r['regime_label']}")
    print(f"Deposit:   ${r['deposit']}/week  (started {r['start_date']})")
    print(f"\nThis week's buys:")
    for row in r["allocation"]:
        shares_str = f"{row['shares']:.4f} shares" if row["shares"] else "N/A"
        print(f"  {row['ticker']:<6} {row['weight']:>5.1f}%  ${row['dollar']:>5.2f}  ({shares_str})")
    print(f"\n10-Year projection (median): ${r['projection']['final_median']:,.0f}")
    print(f"  Best case (p90):           ${r['projection']['final_p90']:,.0f}")
    print(f"  Worst case (p10):          ${r['projection']['final_p10']:,.0f}")
    print(f"  Total deposited:           ${r['projection']['total_deposited_final']:,.0f}")
