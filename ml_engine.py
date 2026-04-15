"""
ML Signal Engine
================
Walk-forward machine learning pipeline for TSMOM signal generation.

Three models compete against the rule-based TSMOM benchmark:
  1. Logistic Regression (L2) -- fast, interpretable baseline
  2. Random Forest            -- ensemble, handles nonlinear interactions
  3. XGBoost                  -- gradient boosting, industry standard

No-lookahead guarantee
----------------------
  - Training window ends TRAIN_GAP days before each retrain date
  - Features use only trailing windows (no forward-looking computation)
  - Target is the forward return starting the day AFTER the feature date
  - Quarterly retraining with a 3-year burn-in before first prediction
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BURN_IN_DAYS    = 756   # 3 years minimum before first prediction
RETRAIN_FREQ    = 63    # Retrain every quarter (~63 trading days)
TRAIN_GAP       = 21    # Exclude last N days from training (prevents target leakage)
FORWARD_HORIZON = 21    # Prediction horizon: sign of next-21-day return

FEATURE_COLS = [
    "mom_21d", "mom_63d", "mom_126d", "mom_252d",
    "vol_20d", "vol_60d", "vol_ratio",
    "skew_60d", "rsi_14d", "cs_rank",
]

# Maps full model name -> short key used in result dict keys
ML_MODELS = {
    "Logistic Regression": "logistic_regression",
    "Random Forest":       "random_forest",
    "XGBoost":             "xgboost",
}


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def _compute_rsi(log_returns: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    RSI via Wilder's EWM smoothing (alpha = 1/window, adjust=False).

    Applied column-wise so each asset gets its own RSI series.
    Values in [0, 100].
    """
    gain = log_returns.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-log_returns).clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def engineer_features(log_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all ML features for every asset and date.

    Returns a DataFrame with MultiIndex columns (feature_name, ticker)
    and a DatetimeIndex matching log_returns.  Slice per asset with:
        features.xs(ticker, axis=1, level=1)

    Features
    --------
    mom_Nd   : N-day trailing log return  (N = 21, 63, 126, 252)
    vol_20d  : 20-day annualized realized volatility
    vol_60d  : 60-day annualized realized volatility
    vol_ratio: vol_20d / vol_60d  (short/long vol regime indicator)
    skew_60d : 60-day rolling skewness of daily log returns
    rsi_14d  : 14-day RSI (Wilder smoothing)
    cs_rank  : cross-sectional percentile rank of 252d momentum
    """
    TRADING_DAYS = 252
    surfaces = {}

    for n in [21, 63, 126, 252]:
        surfaces[f"mom_{n}d"] = log_returns.rolling(n).sum()

    surfaces["vol_20d"]   = log_returns.rolling(20).std() * np.sqrt(TRADING_DAYS)
    surfaces["vol_60d"]   = log_returns.rolling(60).std() * np.sqrt(TRADING_DAYS)
    surfaces["vol_ratio"] = surfaces["vol_20d"] / surfaces["vol_60d"].replace(0, np.nan)
    surfaces["skew_60d"]  = log_returns.rolling(60).skew()
    surfaces["rsi_14d"]   = _compute_rsi(log_returns, 14)
    surfaces["cs_rank"]   = surfaces["mom_252d"].rank(axis=1, pct=True)

    # Build MultiIndex column DataFrame: level 0 = feature, level 1 = ticker
    return pd.concat(surfaces, axis=1)[FEATURE_COLS]


def compute_targets(log_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Binary classification targets: 1 if next FORWARD_HORIZON-day return > 0.

    Target on day t = 1  if log_returns[t+1 : t+horizon+1].sum() > 0
                    = 0  otherwise
    Last FORWARD_HORIZON rows are NaN (incomplete forward window).
    """
    fwd = log_returns.rolling(FORWARD_HORIZON).sum().shift(-FORWARD_HORIZON)
    binary = (np.sign(fwd).replace(0, 1) == 1).astype(float)
    return binary.where(fwd.notna())


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def _train_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Fit Logistic Regression, Random Forest, and XGBoost on (X_train, y_train).

    StandardScaler is fit on X_train and stored alongside LR — tree models
    are scale-invariant and receive raw unscaled features.

    Returns
    -------
    dict with keys: "lr", "rf", "xgb", "scaler"
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    lr.fit(X_scaled, y_train)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=20,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    xgb_clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", verbosity=0, random_state=42,
    )
    xgb_clf.fit(X_train, y_train)

    return {"lr": lr, "rf": rf, "xgb": xgb_clf, "scaler": scaler}


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _get_retraining_dates(index: pd.DatetimeIndex) -> list:
    """
    Return dates at which models are retrained.

    First retrain at BURN_IN_DAYS; subsequent retrains every RETRAIN_FREQ days.
    """
    out = []
    i = BURN_IN_DAYS
    while i < len(index):
        out.append(index[i])
        i += RETRAIN_FREQ
    return out


def run_walk_forward(
    features_wide: pd.DataFrame,
    targets: pd.DataFrame,
    log_returns: pd.DataFrame,
) -> tuple:
    """
    Walk-forward signal generation for all three ML models.

    For each quarterly retraining window:
      1. Train on [window_start : retrain_date - TRAIN_GAP] per asset
      2. Predict P(up) for [retrain_date : next retrain - 1]
      3. Convert to continuous signal: (P(up) - 0.5) * 2 -> [-1, +1]

    Models are trained per-asset (each asset is an independent sample set).
    Features are the same across assets; only the target series differs.

    Parameters
    ----------
    features_wide : pd.DataFrame
        MultiIndex columns (feature, ticker) from engineer_features().
    targets : pd.DataFrame
        Binary targets (n_days x n_assets) from compute_targets().
    log_returns : pd.DataFrame
        Used for index and column references only.

    Returns
    -------
    signal_dfs : dict[str, pd.DataFrame]
        Continuous signals in [-1, +1] for each model name.
    feature_importances : dict[str, pd.Series]
        Normalized average importances (across retrains and assets).
    """
    index   = log_returns.index
    tickers = log_returns.columns.tolist()

    # Probability accumulators (filled during walk-forward)
    probs = {
        name: pd.DataFrame(np.nan, index=index, columns=tickers)
        for name in ML_MODELS
    }

    # Feature importance accumulators
    imp_acc = {name: [] for name in ML_MODELS}

    retrain_dates = _get_retraining_dates(index)
    n_retrains    = len(retrain_dates)

    print(f"[ML] Walk-forward | {n_retrains} windows | "
          f"{retrain_dates[0].date()} -> {retrain_dates[-1].date()}")

    for k, retrain_date in enumerate(retrain_dates):
        t_r_idx       = index.get_loc(retrain_date)
        train_end     = max(0, t_r_idx - TRAIN_GAP)
        predict_end   = (
            index.get_loc(retrain_dates[k + 1])
            if k + 1 < n_retrains else len(index)
        )
        predict_slice = index[t_r_idx:predict_end]

        # Progress reporting (~5 updates total)
        if (k + 1) % max(1, n_retrains // 5) == 0 or k == 0:
            pct = (k + 1) / n_retrains * 100
            print(f"[ML]   {pct:5.1f}%  window {k+1}/{n_retrains}  "
                  f"at {retrain_date.date()}  train_n={train_end}")

        for ticker in tickers:
            # Per-asset feature slice: (n_days, n_features)
            feat = features_wide.xs(ticker, axis=1, level=1)
            tgt  = targets[ticker]

            # Training data — slice up to train_end, drop NaN rows
            X_raw = feat.iloc[:train_end]
            y_raw = tgt.iloc[:train_end]
            mask  = X_raw.notna().all(axis=1) & y_raw.notna()
            X_tr  = X_raw[mask].values
            y_tr  = y_raw[mask].values.astype(int)

            # Skip if insufficient data or only one class present
            if len(X_tr) < 100 or len(np.unique(y_tr)) < 2:
                continue

            trained = _train_models(X_tr, y_tr)

            # Prediction data
            X_pred_raw  = feat.reindex(predict_slice)
            valid_pred  = X_pred_raw.notna().all(axis=1)
            valid_dates = valid_pred[valid_pred].index

            if valid_dates.empty:
                continue

            X_vals   = X_pred_raw.loc[valid_dates].values
            X_scaled = trained["scaler"].transform(X_vals)

            probs["Logistic Regression"].loc[valid_dates, ticker] = (
                trained["lr"].predict_proba(X_scaled)[:, 1]
            )
            probs["Random Forest"].loc[valid_dates, ticker] = (
                trained["rf"].predict_proba(X_vals)[:, 1]
            )
            probs["XGBoost"].loc[valid_dates, ticker] = (
                trained["xgb"].predict_proba(X_vals)[:, 1]
            )

            # Accumulate importances (aggregated later)
            imp_acc["Logistic Regression"].append(np.abs(trained["lr"].coef_[0]))
            imp_acc["Random Forest"].append(trained["rf"].feature_importances_)
            imp_acc["XGBoost"].append(trained["xgb"].feature_importances_)

    # Convert probabilities -> continuous signals in [-1, +1]
    signal_dfs = {name: (df - 0.5) * 2 for name, df in probs.items()}

    # Average importances across all retrains and assets, then normalize
    feature_importances = {}
    for name, arrays in imp_acc.items():
        if arrays:
            avg = np.mean(arrays, axis=0)
            feature_importances[name] = pd.Series(avg / avg.sum(), index=FEATURE_COLS)

    print("[ML] Walk-forward complete")
    return signal_dfs, feature_importances


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_ml_pipeline(log_returns: pd.DataFrame) -> dict:
    """
    Full ML pipeline: feature engineering -> walk-forward signals -> portfolio returns.

    Uses tsmom utilities to apply the same vol-scaling as the benchmark strategies,
    ensuring a fair apples-to-apples performance comparison.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns from tsmom.compute_log_returns().

    Returns
    -------
    dict
        For each model: portfolio returns, cumulative returns, rolling Sharpe,
        drawdown series, and summary stats.  Plus feature_importances dict.

        Key pattern: "cum_logistic_regression", "stats_random_forest", etc.
        Use ML_MODELS dict to iterate: {display_name: key_suffix}.
    """
    from tsmom import (
        build_portfolio_returns,
        compute_realized_vol,
        compute_vol_scaled_size,
        cumulative_returns,
        rolling_sharpe,
        drawdown_series,
        summary_stats,
        VOL_WINDOW,
        TARGET_VOL,
        MAX_LEVERAGE,
    )

    print("\n" + "=" * 60)
    print("ML Signal Pipeline")
    print("=" * 60)

    features_wide = engineer_features(log_returns)
    targets       = compute_targets(log_returns)

    print(f"[ML] Features ({len(FEATURE_COLS)}): {', '.join(FEATURE_COLS)}")
    print(f"[ML] Target: sign of next-{FORWARD_HORIZON}-day return")

    signal_dfs, feature_importances = run_walk_forward(features_wide, targets, log_returns)

    # Recompute vol scaling (same parameters as benchmark)
    realized_vol = compute_realized_vol(log_returns, VOL_WINDOW)
    vol_size     = compute_vol_scaled_size(realized_vol, TARGET_VOL, MAX_LEVERAGE)

    out = {"feature_importances": feature_importances}

    print("\n[ML] Portfolio results:")
    for display_name, key in ML_MODELS.items():
        signal     = signal_dfs[display_name]
        pf_returns = build_portfolio_returns(log_returns, signal, vol_scaled_size=vol_size)
        stats      = summary_stats(pf_returns, display_name)

        out[f"returns_{key}"] = pf_returns
        out[f"cum_{key}"]     = cumulative_returns(pf_returns)
        out[f"sharpe_{key}"]  = rolling_sharpe(pf_returns)
        out[f"dd_{key}"]      = drawdown_series(pf_returns)
        out[f"stats_{key}"]   = stats

        print(f"  {display_name:22s} | Sharpe {stats['Sharpe Ratio']} "
              f"| Return {stats['Ann. Return']} | MaxDD {stats['Max Drawdown']}")

    return out
