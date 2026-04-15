"""
Volatility-Scaled Time-Series Momentum (TSMOM) Strategy
========================================================
Implements the momentum strategy from Moskowitz, Ooi, and Pedersen (2012),
"Time Series Momentum," Journal of Financial Economics.

Compares an unscaled (signal-only) TSMOM against a volatility-scaled version
that targets 10% annualized portfolio volatility.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ASSETS = ["SPY", "QQQ", "GLD", "TLT", "EEM", "IWM", "NVDA", "GOOG"]
START_DATE = "2005-01-01"
END_DATE = None  # None = today

LOOKBACK_DAYS = 252       # 12-month signal lookback
VOL_WINDOW = 20           # Rolling realized volatility window (days)
TARGET_VOL = 0.15         # 10% annualized target volatility
MAX_LEVERAGE = 2.0        # Per-asset leverage cap
TRADING_DAYS = 252        # Annualization factor

# Stress periods to shade on charts [label, start, end]
STRESS_PERIODS = [
    ("GFC 2008",  "2008-09-01", "2009-03-31"),
    ("COVID 2020", "2020-02-15", "2020-04-30"),
    ("Rate Shock 2022", "2022-01-01", "2022-12-31"),
]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: DATA LAYER
# ─────────────────────────────────────────────────────────────────────────────

def fetch_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """
    Download adjusted daily close prices from Yahoo Finance.

    Parameters
    ----------
    tickers : list of str
        List of ticker symbols to download.
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str or None
        End date in 'YYYY-MM-DD' format. If None, uses today.

    Returns
    -------
    pd.DataFrame
        DataFrame of adjusted close prices, columns = tickers.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"].copy()
    prices.dropna(how="all", inplace=True)
    print(f"[Data] Downloaded {len(prices)} daily rows for {list(prices.columns)}")
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from price series.

    Log returns are preferred because they are additive across time and
    approximately normally distributed for daily data.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices.

    Returns
    -------
    pd.DataFrame
        Daily log returns (NaN on first row).
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: SIGNAL ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_momentum_signal(log_returns: pd.DataFrame, lookback: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Compute the binary time-series momentum signal for each asset.

    Signal = +1 (long) if trailing `lookback`-day log return is positive,
             -1 (short) otherwise. No intermediate values — pure directional bet.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns per asset.
    lookback : int
        Rolling window in trading days.

    Returns
    -------
    pd.DataFrame
        Signal DataFrame with values in {-1, +1}.
    """
    # Trailing return = sum of log returns over window (= log(P_t / P_{t-lookback}))
    trailing_return = log_returns.rolling(lookback).sum()
    signal = np.sign(trailing_return)
    signal.replace(0, 1, inplace=True)  # treat exactly-zero as long
    return signal


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: VOLATILITY SCALER
# ─────────────────────────────────────────────────────────────────────────────

def compute_realized_vol(log_returns: pd.DataFrame, window: int = VOL_WINDOW) -> pd.DataFrame:
    """
    Compute rolling annualized realized volatility for each asset.

    Uses the standard deviation of daily log returns scaled by √252.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns per asset.
    window : int
        Rolling window in trading days.

    Returns
    -------
    pd.DataFrame
        Annualized realized volatility per asset.
    """
    daily_vol = log_returns.rolling(window).std()
    annualized_vol = daily_vol * np.sqrt(TRADING_DAYS)
    return annualized_vol


def compute_vol_scaled_size(
    realized_vol: pd.DataFrame,
    target_vol: float = TARGET_VOL,
    max_leverage: float = MAX_LEVERAGE,
) -> pd.DataFrame:
    """
    Compute per-asset position sizes using the vol-targeting rule.

    scaled_size = target_vol / realized_vol, capped at max_leverage.

    This is the core mechanic that differentiates TSMOM from simple momentum:
    high-vol assets get smaller positions, low-vol assets get larger ones.

    Parameters
    ----------
    realized_vol : pd.DataFrame
        Annualized realized volatility per asset.
    target_vol : float
        Desired annualized portfolio contribution per asset.
    max_leverage : float
        Hard cap on per-asset notional exposure.

    Returns
    -------
    pd.DataFrame
        Vol-scaled position sizes (pre-signal).
    """
    scaled = target_vol / realized_vol
    scaled = scaled.clip(upper=max_leverage)
    return scaled


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: PORTFOLIO ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def build_portfolio_returns(
    log_returns: pd.DataFrame,
    signal: pd.DataFrame,
    vol_scaled_size: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Construct daily portfolio returns by combining signal and position sizes.

    Each asset contributes equally to the book (equal-weight across N assets).
    Positions are entered at close and returns are measured the following day
    (signal is lagged by 1 to avoid look-ahead bias).

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns per asset.
    signal : pd.DataFrame
        Momentum signal per asset (+1 or -1).
    vol_scaled_size : pd.DataFrame or None
        Per-asset position sizes. If None, uses equal weight (unscaled).

    Returns
    -------
    pd.Series
        Daily portfolio log returns.
    """
    n_assets = log_returns.shape[1]

    if vol_scaled_size is not None:
        # Vol-scaled: signal direction × vol-adjusted size
        positions = signal * vol_scaled_size
    else:
        # Unscaled: pure ±1 signal
        positions = signal.copy()

    # Equal-weight across the book (1/N normalization)
    positions = positions / n_assets

    # Lag positions by 1 day to prevent look-ahead bias
    positions_lagged = positions.shift(1)

    # Daily portfolio return = sum of (position × asset return)
    portfolio_returns = (positions_lagged * log_returns).sum(axis=1)

    # Drop rows where we have no valid positions
    valid_mask = positions_lagged.notna().any(axis=1)
    portfolio_returns = portfolio_returns[valid_mask]

    return portfolio_returns


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: PERFORMANCE ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

def cumulative_returns(daily_returns: pd.Series) -> pd.Series:
    """
    Compute cumulative returns from daily log returns.

    Returns
    -------
    pd.Series
        Cumulative return index starting at 1.0.
    """
    return np.exp(daily_returns.cumsum())


def rolling_sharpe(daily_returns: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    """
    Compute rolling annualized Sharpe ratio (assuming zero risk-free rate).

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio returns.
    window : int
        Rolling window in trading days.

    Returns
    -------
    pd.Series
        Rolling Sharpe ratio.
    """
    roll_mean = daily_returns.rolling(window).mean()
    roll_std = daily_returns.rolling(window).std()
    sharpe = (roll_mean / roll_std) * np.sqrt(TRADING_DAYS)
    return sharpe


def drawdown_series(daily_returns: pd.Series) -> pd.Series:
    """
    Compute the rolling drawdown from the cumulative return peak.

    Returns
    -------
    pd.Series
        Drawdown values as negative fractions (e.g., -0.20 = 20% drawdown).
    """
    cum = cumulative_returns(daily_returns)
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    return drawdown


def summary_stats(daily_returns: pd.Series, label: str) -> dict:
    """
    Compute a standard set of annualized performance statistics.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio returns.
    label : str
        Strategy label for display.

    Returns
    -------
    dict
        Dictionary of performance metrics.
    """
    ann_return = daily_returns.mean() * TRADING_DAYS
    ann_vol = daily_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    max_dd = drawdown_series(daily_returns).min()
    calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "Strategy": label,
        "Ann. Return": f"{ann_return:.2%}",
        "Ann. Vol": f"{ann_vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Calmar Ratio": f"{calmar:.2f}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline() -> dict:
    """
    Execute the full TSMOM pipeline and return all result objects.

    Returns
    -------
    dict
        Contains raw returns, signals, cumulative returns, rolling Sharpe,
        drawdowns, and summary statistics for both strategy variants.
    """
    print("=" * 60)
    print("TSMOM Strategy Pipeline")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────
    prices = fetch_prices(ASSETS, START_DATE, END_DATE)
    log_returns = compute_log_returns(prices)
    print(f"[Data] Date range: {log_returns.index[0].date()} to {log_returns.index[-1].date()}")

    # ── Signals ───────────────────────────────────────────────────────────
    signal = compute_momentum_signal(log_returns, LOOKBACK_DAYS)
    print(f"[Signal] 12-month momentum signal computed for {len(ASSETS)} assets")

    # ── Vol Scaler ────────────────────────────────────────────────────────
    realized_vol = compute_realized_vol(log_returns, VOL_WINDOW)
    vol_size = compute_vol_scaled_size(realized_vol, TARGET_VOL, MAX_LEVERAGE)
    print(f"[Vol] 60-day realized vol computed; target={TARGET_VOL:.0%}, cap={MAX_LEVERAGE}x")

    # ── Portfolio ─────────────────────────────────────────────────────────
    unscaled_returns = build_portfolio_returns(log_returns, signal, vol_scaled_size=None)
    scaled_returns = build_portfolio_returns(log_returns, signal, vol_scaled_size=vol_size)

    # Align both series to the same date range
    common_idx = unscaled_returns.index.intersection(scaled_returns.index)
    unscaled_returns = unscaled_returns[common_idx]
    scaled_returns = scaled_returns[common_idx]
    print(f"[Portfolio] Returns computed: {len(common_idx)} trading days")

    # ── Analytics ─────────────────────────────────────────────────────────
    cum_unscaled = cumulative_returns(unscaled_returns)
    cum_scaled = cumulative_returns(scaled_returns)

    sharpe_unscaled = rolling_sharpe(unscaled_returns)
    sharpe_scaled = rolling_sharpe(scaled_returns)

    dd_unscaled = drawdown_series(unscaled_returns)
    dd_scaled = drawdown_series(scaled_returns)

    stats_unscaled = summary_stats(unscaled_returns, "Unscaled TSMOM")
    stats_scaled = summary_stats(scaled_returns, "Vol-Scaled TSMOM")

    print("\n-- Summary Statistics ------------------------------------------")
    stats_df = pd.DataFrame([stats_unscaled, stats_scaled]).set_index("Strategy")
    print(stats_df.to_string())
    print()

    return {
        "log_returns": log_returns,
        "signal": signal,
        "realized_vol": realized_vol,
        "vol_size": vol_size,
        "unscaled_returns": unscaled_returns,
        "scaled_returns": scaled_returns,
        "cum_unscaled": cum_unscaled,
        "cum_scaled": cum_scaled,
        "sharpe_unscaled": sharpe_unscaled,
        "sharpe_scaled": sharpe_scaled,
        "dd_unscaled": dd_unscaled,
        "dd_scaled": dd_scaled,
        "stats_unscaled": stats_unscaled,
        "stats_scaled": stats_scaled,
    }


if __name__ == "__main__":
    results = run_pipeline()
