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


# ─────────────────────────────────────────────────────────────────────────────
# 2-MONTH SPRINT PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────

SPRINT_UNIVERSE = ["NVDA", "AMD", "META", "PLTR", "TSLA",
                   "MSFT", "GOOGL", "AMZN", "AAPL", "CRM"]

SPRINT_WEIGHTS = [0.30, 0.22, 0.18, 0.15, 0.15]  # top-5 concentrated

def _fetch_sprint_signals(tickers: list) -> pd.DataFrame:
    """
    Fetch 90-day price history and compute momentum + vol signals for each ticker.
    Returns DataFrame with columns: ticker, ret_21d, ret_63d, vol_20d, rsi_14, price
    """
    end   = date.today()
    start = end - timedelta(days=120)
    rows  = []
    for tk in tickers:
        try:
            hist = yf.Ticker(tk).history(start=start.isoformat(), end=end.isoformat())
            if hist.empty or len(hist) < 25:
                continue
            closes = hist["Close"].dropna()
            price  = float(closes.iloc[-1])
            ret_21 = float(closes.iloc[-1] / closes.iloc[-22] - 1) if len(closes) >= 22 else 0.0
            ret_63 = float(closes.iloc[-1] / closes.iloc[-64] - 1) if len(closes) >= 64 else ret_21
            log_r  = np.log(closes / closes.shift(1)).dropna()
            vol_20 = float(log_r.iloc[-20:].std() * np.sqrt(252))

            # Wilder RSI-14
            delta  = closes.diff()
            gain   = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
            loss   = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
            rs     = gain.iloc[-1] / (loss.iloc[-1] + 1e-9)
            rsi    = float(100 - 100 / (1 + rs))

            rows.append({"ticker": tk, "price": price,
                         "ret_21d": ret_21, "ret_63d": ret_63,
                         "vol_20d": vol_20, "rsi_14": rsi})
        except Exception:
            continue
    return pd.DataFrame(rows)


def _fetch_quick_analyst_scores(tickers: list) -> dict:
    """Fast analyst score for each ticker (simplified, no upgrades/downgrades)."""
    scores = {}
    for tk in tickers:
        try:
            info = yf.Ticker(tk).info
            rec_score  = info.get("recommendationMean")   # 1=SB, 5=SS
            n_analysts = info.get("numberOfAnalystOpinions", 0) or 0
            current_p  = info.get("currentPrice") or info.get("regularMarketPrice")
            target_p   = info.get("targetMeanPrice")

            c1 = (5 - min(5, max(1, rec_score))) / 4 * 40 if rec_score else 20.0
            if current_p and target_p and current_p > 0:
                upside  = (target_p - current_p) / current_p * 100
                c2 = (min(100, max(-50, upside)) + 50) / 150 * 40
            else:
                c2 = 20.0
            c3 = min(20.0, n_analysts / 30 * 20)
            scores[tk] = round(min(100, max(0, c1 + c2 + c3)), 1)
        except Exception:
            scores[tk] = 50.0
    return scores


def _score_tickers(signals: pd.DataFrame, analyst_scores: dict) -> pd.DataFrame:
    """
    Composite sprint score (0-100):
      - 30% momentum_21d  (normalized, >0 rewarded, >20% → max pts)
      - 20% momentum_63d
      - 25% analyst ML sentiment
      - 15% RSI position  (50-70 ideal, penalise >75 overbought)
      - 10% low vol bonus (lower vol = more predictable momentum)
    """
    df = signals.copy()

    # Momentum scores (sigmoid-like, clipped)
    df["s_mom21"] = df["ret_21d"].clip(-0.30, 0.30).apply(
        lambda x: (x / 0.30 + 1) / 2 * 30)
    df["s_mom63"] = df["ret_63d"].clip(-0.40, 0.40).apply(
        lambda x: (x / 0.40 + 1) / 2 * 20)

    # Analyst score (0-100 → 0-25)
    df["s_analyst"] = df["ticker"].map(analyst_scores).fillna(50) / 100 * 25

    # RSI: 50-70 = full marks; below 40 or above 75 = penalised
    def rsi_pts(r):
        if 50 <= r <= 70:  return 15.0
        if 40 <= r < 50:   return 10.0 + (r - 40) / 10 * 5
        if 70 < r <= 80:   return 15.0 - (r - 70) / 10 * 10
        return max(0, 5.0 - abs(r - 55) / 20 * 5)
    df["s_rsi"] = df["rsi_14"].apply(rsi_pts)

    # Low-vol bonus: ann vol < 35% ideal for momentum (penalise >70%)
    df["s_vol"] = df["vol_20d"].apply(
        lambda v: 10 * max(0, 1 - max(0, v - 0.35) / 0.35))

    df["score"] = df[["s_mom21","s_mom63","s_analyst","s_rsi","s_vol"]].sum(axis=1).clip(0, 100)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def _sprint_monte_carlo(deposit_total: float, portfolio_ret: float,
                         portfolio_vol: float, weeks: int = 8,
                         n_sims: int = 500) -> dict:
    """
    Monte Carlo simulation over `weeks` weeks.
    portfolio_ret / portfolio_vol are ANNUAL figures.
    Returns weekly paths at percentiles + summary stats.
    """
    mu_w    = portfolio_ret  / 52
    sigma_w = portfolio_vol  / np.sqrt(52)
    rng     = np.random.default_rng(42)

    paths = np.zeros((n_sims, weeks + 1))
    paths[:, 0] = deposit_total  # all money in from week 0

    for s in range(n_sims):
        weekly_r = rng.normal(mu_w, sigma_w, weeks)
        for w in range(weeks):
            paths[s, w + 1] = paths[s, w] * (1 + weekly_r[w])

    today = date.today()
    dates = [(today + timedelta(weeks=w)).isoformat() for w in range(weeks + 1)]

    def pct(q): return [round(float(v), 2) for v in np.percentile(paths, q, axis=0)]

    final = paths[:, -1]
    return {
        "dates":         dates,
        "p10":           pct(10),
        "p25":           pct(25),
        "median":        pct(50),
        "p75":           pct(75),
        "p90":           pct(90),
        "final_p10":     round(float(np.percentile(final, 10)), 2),
        "final_p25":     round(float(np.percentile(final, 25)), 2),
        "final_median":  round(float(np.percentile(final, 50)), 2),
        "final_p75":     round(float(np.percentile(final, 75)), 2),
        "final_p90":     round(float(np.percentile(final, 90)), 2),
        "deposit_total": round(deposit_total, 2),
    }


def get_sprint_portfolio(deposit: float = WEEKLY_DEPOSIT,
                          weeks: int = 8) -> dict:
    """
    2-month maximum-return portfolio.
    Scores SPRINT_UNIVERSE on momentum + analyst signals, picks top 5,
    allocates concentratedly, projects 8-week Monte Carlo.
    """
    deposit_total = round(deposit * weeks, 2)

    print("[Sprint] Fetching momentum signals...")
    signals = _fetch_sprint_signals(SPRINT_UNIVERSE)
    if signals.empty:
        return {"ok": False, "error": "Could not fetch market data"}

    print("[Sprint] Fetching analyst scores...")
    analyst_scores = _fetch_quick_analyst_scores(SPRINT_UNIVERSE)

    print("[Sprint] Scoring and ranking...")
    ranked = _score_tickers(signals, analyst_scores)
    top    = ranked.head(5).reset_index(drop=True)

    # Allocation: concentrated weights on top 5
    alloc_weights = SPRINT_WEIGHTS[:len(top)]
    total_w = sum(alloc_weights)
    alloc_weights = [w / total_w for w in alloc_weights]

    # Portfolio expected return and vol (weighted)
    port_ret = sum(
        alloc_weights[i] * (top.iloc[i]["ret_63d"] * (52 / 13))   # annualise 63d
        for i in range(len(top))
    )
    port_vol = sum(
        alloc_weights[i] * top.iloc[i]["vol_20d"]
        for i in range(len(top))
    )
    port_ret = min(max(port_ret, -0.50), 2.00)  # sanity bounds

    allocation = []
    for i, (_, row) in enumerate(top.iterrows()):
        w    = alloc_weights[i]
        dol  = round(w * deposit_total, 2)
        price = row["price"]
        allocation.append({
            "rank":     i + 1,
            "ticker":   row["ticker"],
            "name":     TICKER_NAMES.get(row["ticker"], row["ticker"]),
            "score":    round(float(row["score"]), 1),
            "weight":   round(w * 100, 1),
            "dollar":   dol,
            "price":    round(price, 2),
            "shares":   round(dol / price, 4) if price > 0 else None,
            "ret_21d":  round(float(row["ret_21d"]) * 100, 1),
            "ret_63d":  round(float(row["ret_63d"]) * 100, 1),
            "vol_20d":  round(float(row["vol_20d"]) * 100, 1),
            "rsi_14":   round(float(row["rsi_14"]), 1),
            "analyst_score": analyst_scores.get(row["ticker"], 50.0),
        })

    projection = _sprint_monte_carlo(deposit_total, port_ret, port_vol, weeks=weeks)

    # Ranked full table for reference
    ranked_full = []
    for _, row in ranked.iterrows():
        ranked_full.append({
            "ticker":     row["ticker"],
            "score":      round(float(row["score"]), 1),
            "ret_21d":    round(float(row["ret_21d"]) * 100, 1),
            "ret_63d":    round(float(row["ret_63d"]) * 100, 1),
            "vol_20d":    round(float(row["vol_20d"]) * 100, 1),
            "rsi_14":     round(float(row["rsi_14"]), 1),
            "analyst":    analyst_scores.get(row["ticker"], 50.0),
            "selected":   row["ticker"] in [a["ticker"] for a in allocation],
        })

    end_date = (date.today() + timedelta(weeks=weeks)).isoformat()

    return {
        "ok":            True,
        "mode":          "sprint",
        "deposit":       deposit,
        "weeks":         weeks,
        "deposit_total": deposit_total,
        "end_date":      end_date,
        "today":         date.today().isoformat(),
        "allocation":    allocation,
        "projection":    projection,
        "ranked_all":    ranked_full,
        "port_ret_ann":  round(port_ret * 100, 1),
        "port_vol_ann":  round(port_vol * 100, 1),
    }


if __name__ == "__main__":
    print("=== 2-Month Sprint ===")
    sp = get_sprint_portfolio()
    if sp["ok"]:
        print(f"Top picks (deposit ${sp['deposit_total']} over {sp['weeks']} weeks):")
        for a in sp["allocation"]:
            print(f"  #{a['rank']} {a['ticker']:<6} score={a['score']}  "
                  f"{a['weight']}%  ${a['dollar']}  "
                  f"mom21={a['ret_21d']:+.1f}%  rsi={a['rsi_14']:.0f}")
        proj = sp["projection"]
        gain_med = (proj["final_median"] / sp["deposit_total"] - 1) * 100
        gain_p90 = (proj["final_p90"]    / sp["deposit_total"] - 1) * 100
        gain_p10 = (proj["final_p10"]    / sp["deposit_total"] - 1) * 100
        print(f"\n8-week projection on ${sp['deposit_total']}:")
        print(f"  Median:    ${proj['final_median']:,.0f}  ({gain_med:+.1f}%)")
        print(f"  Best(p90): ${proj['final_p90']:,.0f}  ({gain_p90:+.1f}%)")
        print(f"  Worst(p10):${proj['final_p10']:,.0f}  ({gain_p10:+.1f}%)")
    print("\n=== 10-Year DCA ===")
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
