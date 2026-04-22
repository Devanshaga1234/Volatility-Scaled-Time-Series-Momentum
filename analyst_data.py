"""
Analyst Data Engine
===================
Fetches Wall Street analyst ratings, price targets, and upgrade/downgrade
history from yfinance. Computes ML-enhanced analytics:

  1. Composite Sentiment Score (0-100) — weighted combo of rating, upside, buy%
  2. Target Revision Trend            — are analysts raising or cutting targets?
  3. Consensus Rating Drift           — how has buy% changed over 3 months?
  4. Firm Accuracy Leaderboard        — which firms' calls were directionally right?

No external API or paid subscription required — all data from yfinance.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

WATCHLIST = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "META",
    "AMZN", "TSLA", "AMD",  "PLTR",  "CRM",
]

ACCURACY_HORIZON = 63   # trading days to evaluate prediction accuracy (~3 months)
REVISION_WINDOW  = 30   # calendar days for recent target revision trend


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_analyst_snapshot(ticker: str) -> dict:
    """
    Pull consensus analyst metrics for one ticker.

    Returns
    -------
    dict with keys: ticker, current_price, target_mean, target_high,
    target_low, target_median, upside_pct, consensus_key, consensus_score,
    n_analysts, avg_rating_str
    """
    try:
        t    = yf.Ticker(ticker)
        info = t.info
        apt  = t.analyst_price_targets or {}

        current = apt.get("current") or info.get("currentPrice", np.nan)
        mean    = apt.get("mean")    or info.get("targetMeanPrice", np.nan)
        high    = apt.get("high")    or info.get("targetHighPrice", np.nan)
        low     = apt.get("low")     or info.get("targetLowPrice", np.nan)
        median  = apt.get("median")  or info.get("targetMedianPrice", np.nan)

        upside = (mean - current) / current * 100 if (current and mean and current > 0) else np.nan

        return {
            "ticker":          ticker,
            "current_price":   current,
            "target_mean":     mean,
            "target_high":     high,
            "target_low":      low,
            "target_median":   median,
            "upside_pct":      upside,
            "consensus_key":   info.get("recommendationKey", "n/a"),
            "consensus_score": info.get("recommendationMean", np.nan),
            "n_analysts":      info.get("numberOfAnalystOpinions", 0),
            "avg_rating_str":  info.get("averageAnalystRating", "N/A"),
        }
    except Exception:
        return {k: np.nan for k in [
            "ticker", "current_price", "target_mean", "target_high",
            "target_low", "target_median", "upside_pct", "consensus_score", "n_analysts",
        ]} | {"ticker": ticker, "consensus_key": "n/a", "avg_rating_str": "N/A"}


def fetch_rating_breakdown(ticker: str) -> dict:
    """
    Fetch monthly rating breakdown (strongBuy/buy/hold/sell/strongSell)
    for current month and 3 months ago.

    Returns dict with 'current' and 'prior' sub-dicts plus raw period DataFrame.
    """
    try:
        rec = yf.Ticker(ticker).recommendations
        if rec is None or rec.empty:
            return {"current": {}, "prior": {}, "periods": pd.DataFrame()}

        def _buy_pct(row):
            total = row[["strongBuy", "buy", "hold", "sell", "strongSell"]].sum()
            return (row["strongBuy"] + row["buy"]) / total * 100 if total > 0 else 0

        periods = {}
        for p in ["0m", "-1m", "-2m", "-3m"]:
            match = rec[rec["period"] == p]
            if not match.empty:
                periods[p] = match.iloc[0].to_dict()

        current = periods.get("0m",  {})
        prior   = periods.get("-3m", {})

        return {
            "ticker":       ticker,
            "current":      current,
            "prior":        prior,
            "periods":      periods,
            "buy_pct_now":  _buy_pct(pd.Series(current))  if current else np.nan,
            "buy_pct_3m":   _buy_pct(pd.Series(prior))    if prior   else np.nan,
        }
    except Exception:
        return {"ticker": ticker, "current": {}, "prior": {}, "periods": {}, "buy_pct_now": np.nan, "buy_pct_3m": np.nan}


def fetch_upgrades_downgrades(ticker: str, days: int = 90) -> pd.DataFrame:
    """
    Fetch recent firm-level rating actions for one ticker.

    Returns DataFrame with columns:
        date, ticker, Firm, ToGrade, FromGrade, Action,
        priceTargetAction, currentPriceTarget, priorPriceTarget, target_change
    Filtered to last `days` calendar days.
    """
    try:
        ud = yf.Ticker(ticker).upgrades_downgrades
        if ud is None or ud.empty:
            return pd.DataFrame()

        ud = ud.copy()
        ud["ticker"]        = ticker
        ud["target_change"] = ud["currentPriceTarget"] - ud["priorPriceTarget"]

        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        ud = ud[ud.index >= cutoff]

        return ud.reset_index().rename(columns={"GradeDate": "date"})
    except Exception:
        return pd.DataFrame()


def fetch_all_snapshots(tickers: list) -> pd.DataFrame:
    """Fetch fetch_analyst_snapshot() for each ticker; return as one DataFrame."""
    rows = []
    for i, t in enumerate(tickers):
        print(f"[Analyst] Fetching snapshot {i+1}/{len(tickers)}: {t}")
        rows.append(fetch_analyst_snapshot(t))
    return pd.DataFrame(rows).set_index("ticker")


def fetch_all_breakdowns(tickers: list) -> pd.DataFrame:
    """
    Fetch rating breakdown for each ticker.
    Returns DataFrame with one row per ticker, columns for each period's counts.
    """
    rows = []
    for ticker in tickers:
        bd = fetch_rating_breakdown(ticker)
        cur = bd.get("current", {})
        row = {
            "ticker":       ticker,
            "strong_buy":   cur.get("strongBuy",   0),
            "buy":          cur.get("buy",          0),
            "hold":         cur.get("hold",         0),
            "sell":         cur.get("sell",         0),
            "strong_sell":  cur.get("strongSell",   0),
            "buy_pct_now":  bd.get("buy_pct_now",   np.nan),
            "buy_pct_3m":   bd.get("buy_pct_3m",    np.nan),
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker")


def fetch_all_upgrades(tickers: list, days: int = 90) -> pd.DataFrame:
    """Fetch and concatenate upgrades/downgrades for all tickers."""
    frames = []
    for i, t in enumerate(tickers):
        print(f"[Analyst] Fetching upgrades {i+1}/{len(tickers)}: {t}")
        df = fetch_upgrades_downgrades(t, days=days)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("date", ascending=False).reset_index(drop=True)
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# ML ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_sentiment_score(snapshots: pd.DataFrame, breakdowns: pd.DataFrame) -> pd.DataFrame:
    """
    Composite ML sentiment score (0-100) per ticker.

    Three equally-weighted components:

    Component 1 — Rating quality (0-40 pts)
        consensus_score: 1.0 = Strong Buy, 5.0 = Strong Sell
        score = (5 - consensus_score) / 4 * 40

    Component 2 — Price target upside (0-40 pts)
        upside_pct clipped to [-50, +100], normalized to [0, 40]
        score = (clipped_upside + 50) / 150 * 40

    Component 3 — Buy ratio (0-20 pts)
        (strongBuy + buy) / total_analysts * 20

    Interpretation:
        80-100 : Strong bullish consensus
        60-80  : Moderately bullish
        40-60  : Mixed / hold territory
        20-40  : Moderately bearish
        0-20   : Strong bearish consensus
    """
    df = snapshots.copy()

    # Component 1: Rating
    c1 = (5 - df["consensus_score"].clip(1, 5)) / 4 * 40

    # Component 2: Upside
    upside_clipped = df["upside_pct"].clip(-50, 100)
    c2 = (upside_clipped + 50) / 150 * 40

    # Component 3: Buy ratio from breakdown data
    bd = breakdowns.reindex(df.index)
    total = (bd[["strong_buy", "buy", "hold", "sell", "strong_sell"]]
             .fillna(0).sum(axis=1).clip(lower=1))
    buy_ratio = (bd["strong_buy"].fillna(0) + bd["buy"].fillna(0)) / total
    c3 = buy_ratio * 20

    score = (c1 + c2 + c3).clip(0, 100).round(1)

    result = pd.DataFrame({
        "sentiment_score":    score,
        "rating_component":   c1.round(1),
        "upside_component":   c2.round(1),
        "buy_ratio_component": c3.round(1),
    }, index=df.index)

    result["sentiment_label"] = pd.cut(
        result["sentiment_score"],
        bins=[0, 20, 40, 60, 80, 100],
        labels=["Strong Bearish", "Bearish", "Neutral", "Bullish", "Strong Bullish"],
        include_lowest=True,
    )

    return result


def compute_target_revision_trend(upgrades: pd.DataFrame) -> pd.DataFrame:
    """
    Per-ticker average target price change (current - prior) over last REVISION_WINDOW days.

    Positive = analysts raising targets (bullish revision).
    Negative = analysts cutting targets (bearish revision).

    Only includes rows where both targets are non-zero and non-NaN.
    """
    if upgrades.empty:
        return pd.DataFrame(columns=["avg_target_change", "n_revisions"])

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=REVISION_WINDOW)
    # Ensure date column is tz-aware for comparison
    dates = pd.to_datetime(upgrades["date"], utc=True)
    recent = upgrades[dates >= cutoff].copy()

    # Filter out rows with missing/zero targets
    recent = recent[
        recent["currentPriceTarget"].notna() & (recent["currentPriceTarget"] > 0) &
        recent["priorPriceTarget"].notna()  & (recent["priorPriceTarget"]  > 0)
    ]

    if recent.empty:
        return pd.DataFrame(columns=["avg_target_change", "n_revisions"])

    trend = (
        recent.groupby("ticker")["target_change"]
        .agg(avg_target_change="mean", n_revisions="count")
        .round(2)
    )
    return trend


def compute_rating_drift(breakdowns: pd.DataFrame) -> pd.DataFrame:
    """
    Per-ticker change in buy% from 3 months ago to today.

    Positive drift = analysts becoming more bullish.
    Negative drift = analysts becoming more bearish.
    """
    df = breakdowns[["buy_pct_now", "buy_pct_3m"]].copy().dropna()
    df["drift"] = (df["buy_pct_now"] - df["buy_pct_3m"]).round(1)
    return df


def compute_firm_accuracy(upgrades_all: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Evaluate each analyst firm's directional accuracy.

    Method
    ------
    For each upgrade (Action='up') or downgrade (Action='down') that occurred
    at least ACCURACY_HORIZON trading days ago:
      - Record the stock price on the grade date (approximate via close)
      - Record the stock price ACCURACY_HORIZON trading days later
      - "Correct" if: upgrade AND forward return > 0, OR downgrade AND forward return < 0

    Firms with < 3 qualifying ratings are excluded (insufficient sample).

    Returns DataFrame sorted by accuracy descending.
    """
    if upgrades_all.empty:
        return pd.DataFrame()

    # Only evaluate directional actions (not maintains)
    directional = upgrades_all[upgrades_all["Action"].isin(["up", "down"])].copy()
    if directional.empty:
        return pd.DataFrame()

    # Filter to grades old enough to have an outcome
    min_age = pd.Timedelta(days=int(ACCURACY_HORIZON * 1.5))
    dates   = pd.to_datetime(directional["date"], utc=True)
    directional = directional[dates <= (pd.Timestamp.now(tz="UTC") - min_age)].copy()
    if directional.empty:
        return pd.DataFrame()

    # Fetch price history for all tickers (2-year window)
    print("[Analyst] Fetching price history for firm accuracy calculation...")
    try:
        raw = yf.download(tickers, period="2y", auto_adjust=True, progress=False)
        prices = raw["Close"]
    except Exception:
        return pd.DataFrame()

    # Normalize index to date-only for alignment
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()

    results = []
    for _, row in directional.iterrows():
        ticker = row["ticker"]
        if ticker not in prices.columns:
            continue

        grade_date = pd.to_datetime(row["date"]).tz_localize(None).normalize()
        price_series = prices[ticker].dropna()

        # Find closest available price on/after grade_date
        after = price_series[price_series.index >= grade_date]
        if after.empty:
            continue
        entry_price = after.iloc[0]
        entry_date  = after.index[0]

        # Find price ACCURACY_HORIZON trading days later
        future = price_series[price_series.index > entry_date]
        if len(future) < ACCURACY_HORIZON:
            continue
        exit_price = future.iloc[ACCURACY_HORIZON - 1]

        fwd_return   = (exit_price - entry_price) / entry_price
        is_correct   = (row["Action"] == "up" and fwd_return > 0) or \
                       (row["Action"] == "down" and fwd_return < 0)

        results.append({
            "Firm":        row["Firm"],
            "ticker":      ticker,
            "action":      row["Action"],
            "fwd_return":  fwd_return,
            "is_correct":  is_correct,
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    accuracy = (
        df.groupby("Firm")
        .agg(
            n_ratings  = ("is_correct", "count"),
            n_correct  = ("is_correct", "sum"),
            avg_return = ("fwd_return", "mean"),
        )
        .reset_index()
    )
    accuracy = accuracy[accuracy["n_ratings"] >= 3]
    accuracy["accuracy_pct"] = (accuracy["n_correct"] / accuracy["n_ratings"] * 100).round(1)
    accuracy["avg_return"]   = (accuracy["avg_return"] * 100).round(1)
    accuracy = accuracy.sort_values("accuracy_pct", ascending=False).reset_index(drop=True)
    return accuracy


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_analyst_pipeline(tickers: list = None) -> dict:
    """
    Full analyst data pipeline: fetch -> compute analytics -> return results dict.

    Parameters
    ----------
    tickers : list of str, or None (uses WATCHLIST)

    Returns
    -------
    dict with keys:
        snapshots      : pd.DataFrame  — consensus metrics per ticker
        breakdowns     : pd.DataFrame  — rating counts per ticker
        upgrades       : pd.DataFrame  — recent upgrades/downgrades (all tickers)
        sentiment      : pd.DataFrame  — ML composite sentiment scores
        revision_trend : pd.DataFrame  — target price revision trend
        rating_drift   : pd.DataFrame  — 3-month buy% change
        firm_accuracy  : pd.DataFrame  — directional accuracy by firm
    """
    if tickers is None:
        tickers = WATCHLIST

    print("=" * 60)
    print("Analyst Data Pipeline")
    print("=" * 60)

    snapshots  = fetch_all_snapshots(tickers)
    breakdowns = fetch_all_breakdowns(tickers)
    upgrades   = fetch_all_upgrades(tickers, days=180)   # 6 months for accuracy calc

    print("[Analyst] Computing ML analytics...")
    sentiment      = compute_sentiment_score(snapshots, breakdowns)
    revision_trend = compute_target_revision_trend(upgrades)
    rating_drift   = compute_rating_drift(breakdowns)
    firm_accuracy  = compute_firm_accuracy(upgrades, tickers)

    print(f"[Analyst] Done. {len(tickers)} tickers | "
          f"{len(upgrades)} upgrade events | "
          f"{len(firm_accuracy)} firms evaluated")

    return {
        "snapshots":      snapshots,
        "breakdowns":     breakdowns,
        "upgrades":       upgrades,
        "sentiment":      sentiment,
        "revision_trend": revision_trend,
        "rating_drift":   rating_drift,
        "firm_accuracy":  firm_accuracy,
    }
