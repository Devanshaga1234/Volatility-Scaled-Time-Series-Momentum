# Volatility-Scaled Time-Series Momentum + ML Signal Comparison

A full quantitative research pipeline implementing the **time-series momentum** strategy from Moskowitz, Ooi & Pedersen (2012), extended with a **walk-forward machine learning comparison** across three models.

> Moskowitz, T., Ooi, Y.H., & Pedersen, L.H. (2012).
> *Time Series Momentum.*
> **Journal of Financial Economics**, 104(2), 228–250.

---

## Overview

This project builds two rule-based TSMOM strategies from scratch, then competes them against three ML signal generators — all evaluated on the same assets with the same vol-scaling framework and strict no-lookahead validation.

**Five strategies, one dashboard:**

| Strategy | Signal | Vol Scaling |
|---|---|---|
| Unscaled TSMOM | `sign(252d trailing return)` | No |
| Vol-Scaled TSMOM | `sign(252d trailing return)` | Yes |
| Logistic Regression | Walk-forward classifier (10 features) | Yes |
| Random Forest | Walk-forward classifier (10 features) | Yes |
| XGBoost | Walk-forward classifier (10 features) | Yes |

**Assets:** SPY, QQQ, GLD, TLT, EEM, IWM, NVDA, GOOG — 2005 to present

---

## File Structure

```
.
├── tsmom.py              # TSMOM pipeline: data, signal, vol scaler, portfolio, analytics
├── ml_engine.py          # ML pipeline: features, walk-forward, LR / RF / XGBoost signals
├── dashboard.py          # Two-tab Plotly HTML dashboard generator
├── tsmom_dashboard.html  # Output artifact — open in any browser
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Quickstart

```bash
pip install -r requirements.txt
python dashboard.py          # runs both pipelines, writes tsmom_dashboard.html
```

Open `tsmom_dashboard.html` in any browser. No server needed — fully self-contained.

---

## TSMOM Strategy

### Signal

For each asset on each day, compute the trailing 12-month log return:

```
trailing_return_t = ln(P_t / P_{t-252})

signal_t = +1   if trailing_return_t > 0   (long)
           -1   if trailing_return_t < 0   (short)
```

The binary rule strips out magnitude — position sizing is handled entirely by the vol scaler, keeping the two concerns cleanly separated.

### Volatility Scaling

Rolling 20-day realized volatility, annualized:

```
ann_vol_t = std(r_{t-19} ... r_t) x sqrt(252)
```

Per-asset position size:

```
scaled_size_t = min(target_vol / ann_vol_t,  max_leverage)
              = min(0.15 / ann_vol_t,  2.0)
```

Assets running hot get smaller positions; calm assets get larger ones. When SPY vol spikes to 80% during a crisis, the scaler cuts exposure to `0.15 / 0.80 = 19%` of normal — automatically reducing drawdown without any discretionary intervention.

### Portfolio Construction

```
positions_t    = signal_t x scaled_size_t / N_assets
portfolio_rt   = sum_i( positions_{i, t-1} x log_return_{i, t} )
```

Positions are **lagged by one day** — signals computed at close on day t are applied to returns on day t+1, eliminating lookahead bias.

---

## ML Signal Engine

### Features (10 per asset per day)

| Feature | Description |
|---|---|
| `mom_21d` | 21-day trailing log return |
| `mom_63d` | 63-day trailing log return |
| `mom_126d` | 126-day trailing log return |
| `mom_252d` | 252-day trailing log return |
| `vol_20d` | 20-day annualized realized vol |
| `vol_60d` | 60-day annualized realized vol |
| `vol_ratio` | `vol_20d / vol_60d` — short vs long vol regime |
| `skew_60d` | 60-day rolling return skewness |
| `rsi_14d` | 14-day RSI (Wilder smoothing) |
| `cs_rank` | Cross-sectional percentile rank of 252d momentum |

### Target

Binary classification: `1` if next-21-day return is positive, `0` otherwise.

### Walk-Forward Validation

Strict no-lookahead protocol:

```
Burn-in:       756 days (3 years) before first prediction
Retrain freq:  Every 63 trading days (quarterly)
Train gap:     Final 21 days excluded from training window
               (prevents target leakage — the forward return horizon)
```

Each retraining window trains three models independently per asset:

- **Logistic Regression** (L2, `C=0.1`) — fast, interpretable baseline; features are StandardScaler-normalized
- **Random Forest** (200 trees, `max_depth=4`) — nonlinear, no scaling needed
- **XGBoost** (200 rounds, `lr=0.05`, `max_depth=3`) — gradient boosting, industry standard

Raw `P(up)` probabilities are converted to continuous signals:

```
signal = (P(up) - 0.5) x 2    -> range [-1, +1]
```

This continuous signal feeds directly into the same `build_portfolio_returns()` function used by TSMOM, with identical vol scaling — ensuring a fair apples-to-apples comparison.

---

## Dashboard

Two-tab interactive HTML dashboard built with Plotly.

### Tab 1 — TSMOM Strategy

- Performance table: Unscaled vs Vol-Scaled
- Cumulative returns (log scale) with stress-period shading
- Rolling 60-day Sharpe ratio
- Drawdown (underwater) curves
- Per-asset realized volatility

### Tab 2 — ML Comparison

- Performance table: all 5 strategies
- Cumulative returns: all 5 strategies overlaid (log scale)
- Rolling Sharpe: ML models
- Drawdown curves: all 5 strategies
- Metrics comparison bar chart (Sharpe / Ann. Return / Max Drawdown)
- Feature importance: normalized LR coefficients, RF impurity, XGBoost gain

---

## Performance Metrics

| Metric | Formula | What it measures |
|---|---|---|
| Ann. Return | `mean(r_daily) x 252` | Average yearly profit |
| Ann. Vol | `std(r_daily) x sqrt(252)` | Yearly risk (realized) |
| Sharpe Ratio | `ann_return / ann_vol` | Return per unit of risk |
| Max Drawdown | `min((cum - peak) / peak)` | Worst peak-to-trough loss |
| Calmar Ratio | `ann_return / abs(max_dd)` | Return per unit of drawdown |

---

## Stress Periods

Three major market events are shaded on all time-series charts:

| Period | Dates | Why it matters for TSMOM |
|---|---|---|
| GFC 2008 | Sep 2008 – Mar 2009 | Equities trended down sharply — short signals correct; vol spike tests the scaler |
| COVID 2020 | Feb – Apr 2020 | V-shaped reversal — hardest environment for momentum; signals caught long into crash |
| Rate Shock 2022 | Jan – Dec 2022 | Simultaneous equity + bond selloff; TSMOM eventually flipped short on both correctly |

---

## Why These Assets

| Ticker | Asset | Role in Portfolio |
|---|---|---|
| SPY | S&P 500 ETF | Large-cap US equity benchmark |
| QQQ | Nasdaq-100 ETF | Tech-heavy, higher beta |
| GLD | Gold ETF | Inflation hedge, crisis safe-haven |
| TLT | 20+ Year Treasury ETF | Long duration bonds; risk-off hedge |
| EEM | Emerging Markets ETF | International equity, higher vol |
| IWM | Russell 2000 ETF | Small-cap US equity |
| NVDA | NVIDIA | High-growth single name; strong momentum dynamics |
| GOOG | Alphabet | Mega-cap tech; diversification vs QQQ |

Low pairwise correlation is intentional — GLD and TLT often move opposite to equities in risk-off periods, smoothing portfolio returns and giving the signal room to add value across regimes.

---

## Dependencies

```
yfinance>=0.2.40       # market data
pandas>=2.0.0          # dataframes
numpy>=1.26.0          # numerics
plotly>=5.20.0         # interactive charts
scikit-learn>=1.4.0    # LR, RF, StandardScaler
xgboost>=2.0.0         # gradient boosting
```

---

## References

- Moskowitz, T., Ooi, Y.H., & Pedersen, L.H. (2012). *Time Series Momentum.* Journal of Financial Economics, 104(2), 228–250.
- Jegadeesh, N., & Titman, S. (1993). *Returns to Buying Winners and Selling Losers.* Journal of Finance, 48(1), 65–91.
- Hurst, B., Ooi, Y.H., & Pedersen, L.H. (2017). *A Century of Evidence on Trend-Following Investing.* AQR Capital Management.
- Baz, J., Granger, N., Harvey, C.R., Le Roux, N., & Rattray, S. (2015). *Dissecting Investment Strategies in the Cross Section and Time Series.* SSRN 2695101.
