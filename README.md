# Volatility-Scaled Time-Series Momentum (TSMOM)

A Python implementation of the **time-series momentum** strategy from:

> Moskowitz, T., Ooi, Y.H., & Pedersen, L.H. (2012).
> *Time Series Momentum.*
> **Journal of Financial Economics**, 104(2), 228–250.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Why These Assets](#why-these-assets)
3. [The Math, Step by Step](#the-math-step-by-step)
4. [The Results](#the-results)
5. [What the Results Mean](#what-the-results-mean)
6. [Stress Periods Explained](#stress-periods-explained)
7. [Dashboard Guide](#dashboard-guide)
8. [File Structure & Usage](#file-structure--usage)
9. [References](#references)

---

## What This Project Does

This project builds a **systematic trend-following strategy** from scratch and compares two versions of it side by side:

- **Unscaled TSMOM**: Look at whether each asset has gone up or down over the past year. If up, hold it long. If down, hold it short. Size every position the same.
- **Vol-Scaled TSMOM**: Same directional bets, but size each position inversely proportional to how volatile that asset has been recently. Riskier assets get smaller positions; calmer assets get larger ones.

The core thesis from Moskowitz et al. (2012) is simple: **assets that have gone up recently tend to keep going up, and assets that have gone down tend to keep going down** — over a 12-month horizon. This is "time-series momentum," and it has been documented across dozens of asset classes and over a century of data.

The twist in this implementation is the **volatility scaler** — a mechanism that targets a constant level of portfolio risk. This is where the two strategies meaningfully diverge, especially during market crises.

---

## Why These Assets

| Ticker | Asset | Role |
|--------|-------|------|
| SPY | S&P 500 ETF | Large-cap US equity; the market benchmark |
| QQQ | Nasdaq-100 ETF | Tech-heavy US equity; higher beta, higher vol |
| GLD | Gold ETF | Inflation hedge, crisis safe-haven |
| TLT | 20+ Year Treasury ETF | Long-duration bonds; negative correlation to equities in risk-off |
| EEM | Emerging Markets ETF | Higher growth, higher vol international equity |
| IWM | Russell 2000 ETF | Small-cap US equity; more economically sensitive than SPY |

These six assets were chosen deliberately for **low pairwise correlation**. Equities (SPY, QQQ, IWM, EEM) tend to move together, but GLD and TLT often move in the opposite direction during selloffs. A diversified book means momentum signals across assets will sometimes cancel out — which is a feature, not a bug. It reduces the bet that any single trend continues and smooths out the portfolio return stream.

---

## The Math, Step by Step

### Step 1: Log Returns

Instead of simple percentage returns, we use **log returns**:

```
r_t = ln(P_t / P_{t-1})
```

**Why log returns?**
- They are **additive across time**: the return over a week is just the sum of 5 daily log returns, which makes rolling window calculations exact.
- Simple returns are multiplicative, so `(1 + r1) × (1 + r2) - 1` is messier to work with.
- Log returns are more symmetric and closer to normally distributed, which makes volatility estimates better behaved.
- For daily data the difference is tiny (log(1.01) ≈ 0.00995 vs. 0.01), but it matters when you sum over long periods.

---

### Step 2: The Momentum Signal

For each asset on each day, compute the **trailing 12-month log return**:

```
trailing_return_t = r_{t-251} + r_{t-250} + ... + r_t
                  = sum of 252 daily log returns
                  = ln(P_t / P_{t-252})
```

Then apply a binary rule:

```
signal_t = +1   if trailing_return_t > 0   (long — trend is up)
signal_t = -1   if trailing_return_t < 0   (short — trend is down)
```

**Why 12 months?**
Moskowitz et al. tested many lookback windows and found 12 months to be the most robust. The intuition: shorter windows capture noise; longer windows mean you're holding through full cycle reversals. 12 months is also the standard used in academic cross-sectional momentum literature (Jegadeesh & Titman, 1993).

**Why binary?**
The magnitude of the trailing return does not meaningfully predict the magnitude of future returns. A 30% trailing return does not imply a 3x stronger future signal than a 10% trailing return. So instead of using continuous position sizes based on return magnitude, we strip it down to direction only. Position sizing is handled separately by the vol scaler — keeping the two concerns cleanly separated.

**What "long" and "short" mean operationally:**
- Long (+1): You own the asset. If it goes up 1%, you make 1%.
- Short (-1): You have borrowed and sold the asset. If it goes down 1%, you make 1%. If it goes up 1%, you lose 1%.

This strategy can therefore **profit in down markets** if the short signal is correct — which is why it performs well during sustained bear markets like 2008.

---

### Step 3: Realized Volatility

For each asset, compute the **60-day rolling realized volatility**:

```
daily_vol_t = std(r_{t-59}, r_{t-58}, ..., r_t)
ann_vol_t   = daily_vol_t × sqrt(252)
```

The `sqrt(252)` converts daily standard deviation to annualized standard deviation. This works because, under the assumption that daily returns are independent, variance scales linearly with time:

```
Var(annual) = 252 × Var(daily)
std(annual) = sqrt(252) × std(daily)
```

**Why 60 days?**
60 days (about 3 months) is a sweet spot between:
- Too short (20 days): Estimates are noisy and react too aggressively to single outlier days.
- Too long (252 days): Estimates are slow to update. By the time vol spikes in a crisis, a 252-day window still reflects calm-period vol from 9 months ago.

60 days gives you a vol estimate that is responsive enough to recent market conditions but stable enough not to whipsaw.

**What this looks like in practice:**
- SPY's normal realized vol: ~15% annualized
- SPY during COVID crash (March 2020): ~80% annualized
- SPY during GFC (late 2008): ~60–80% annualized

---

### Step 4: The Volatility Scaler

This is the core mechanic that separates the two strategies.

```
scaled_size_t = target_vol / ann_vol_t
              = 0.10 / ann_vol_t
```

**Capped at 2.0x** (maximum leverage per asset).

**Intuition:** If your target is 10% annualized vol contribution per asset, and an asset is currently running at 20% vol, you should hold half a normal position (`0.10 / 0.20 = 0.5`). If an asset is running at 5% vol (like TLT in calm periods), you can hold twice a normal position (`0.10 / 0.05 = 2.0`, capped there).

**The critical behavior during crises:**
When SPY's vol spikes to 80% in March 2020, the vol scaler says: `0.10 / 0.80 = 0.125`. So you hold only 12.5% of what you'd hold in normal times. The unscaled strategy keeps holding a full ±1 position. The vol-scaled strategy has automatically reduced exposure by 87.5%.

This is sometimes called **"risk parity"** at the asset level, though true risk parity also considers correlations between assets. Here we're applying it per-asset in isolation, which is simpler and still highly effective.

---

### Step 5: Portfolio Construction

**Unscaled portfolio:**
```
position_{i,t} = signal_{i,t}

portfolio_return_t = (1/N) × sum_i( position_{i,t-1} × r_{i,t} )
```

**Vol-scaled portfolio:**
```
position_{i,t} = signal_{i,t} × scaled_size_{i,t}

portfolio_return_t = (1/N) × sum_i( position_{i,t-1} × r_{i,t} )
```

Where N = 6 (number of assets). The `1/N` gives each asset equal weight in the book.

**The t-1 lag is essential.** Positions are formed using today's signal but applied to tomorrow's returns. This prevents look-ahead bias — you cannot trade on information you don't yet have. In real implementation this means: at close on day t, compute your signal, then enter positions that are live for day t+1.

---

### Step 6: Performance Metrics

**Annualized Return:**
```
ann_return = mean(r_daily) × 252
```
The average daily return scaled to an annual figure. Assumes 252 trading days per year.

**Annualized Volatility:**
```
ann_vol = std(r_daily) × sqrt(252)
```
The standard deviation of daily returns scaled to annual. This is the *realized* portfolio volatility — what you actually experienced, not the target.

**Sharpe Ratio:**
```
Sharpe = ann_return / ann_vol
```
Risk-adjusted return per unit of volatility. We assume a zero risk-free rate for simplicity (the Fed Funds rate approximation doesn't change the strategic comparison). A Sharpe above 1.0 is considered strong for a single strategy. Systematic trend funds (like Man AHL) typically run at 0.5–0.8 Sharpe over long periods.

**Maximum Drawdown:**
```
drawdown_t = (cum_return_t - max(cum_return_{0..t})) / max(cum_return_{0..t})
max_drawdown = min(drawdown_t)
```
The largest peak-to-trough decline in cumulative wealth. Expressed as a negative percentage. A -33% max drawdown means at some point your portfolio was worth 33% less than its prior peak.

**Calmar Ratio:**
```
Calmar = ann_return / |max_drawdown|
```
Return per unit of maximum pain. A Calmar of 0.5 means you earned 50 cents per year for every dollar of peak-to-trough loss you endured. Higher is better. Most funds target a Calmar above 0.5.

**Rolling 60-Day Sharpe:**
At each date t, compute the Sharpe using only the last 60 days of returns. This shows how the strategy's risk-adjusted performance has varied across market regimes — you can see it collapse during certain periods and recover in others.

---

## The Results

Results from the backtest: **January 2005 through March 2026** (21 years, ~5,080 trading days).

| Metric | Unscaled TSMOM | Vol-Scaled TSMOM |
|---|---|---|
| Annualized Return | 3.25% | 2.63% |
| Annualized Volatility | 13.43% | 6.30% |
| **Sharpe Ratio** | **0.24** | **0.42** |
| Max Drawdown | -33.18% | -16.03% |
| Calmar Ratio | 0.10 | 0.16 |

---

## What the Results Mean

### The vol-scaled strategy earned slightly less but was dramatically better on risk-adjusted terms

The unscaled strategy returned 3.25% per year vs. 2.63% for the vol-scaled strategy — a 62 basis point difference in raw return. But the unscaled strategy required you to tolerate **13.43% annualized volatility** and a **33% max drawdown** to earn it.

The vol-scaled strategy delivered its 2.63% at only **6.30% annualized vol** and a **16% max drawdown** — roughly half the risk on both measures.

This is the Sharpe ratio story: 0.42 vs. 0.24. The vol-scaled strategy generates **75% more return per unit of risk**.

### The max drawdown is the number that really matters for investors

A 33% drawdown means that at some point your portfolio was worth one-third less than it had been. To recover from a 33% loss, you need a 49% gain just to get back to flat. For real capital — whether personal savings or an institutional allocation — a 33% drawdown is psychologically and practically devastating. Many investors would have exited the strategy at exactly the wrong moment, locking in the loss.

The vol-scaled strategy's 16% max drawdown is still painful, but it is survivable. It requires a 19% gain to recover — a much more reasonable path.

### The Calmar ratio shows the vol-scaled strategy is more "investor-friendly"

Calmar of 0.10 for unscaled vs. 0.16 for vol-scaled. You are earning 60% more annual return per dollar of max drawdown you endure. This matters enormously for real fund management, where investors redeem capital after large drawdowns.

### Why does vol-scaling reduce both return and vol?

The vol scaler reduces position sizes when volatility is high. High volatility periods often coincide with strong trending periods (volatility begets momentum). So by reducing size during high-vol regimes, you sometimes reduce exposure to the strongest trending periods — giving up some return. But you also protect against volatility spikes that reverse, which is the main source of the unscaled strategy's large drawdowns.

This is a classic **return vs. risk trade-off**, and by the Sharpe and Calmar metrics, the vol-scaled version wins clearly.

### In absolute terms, these Sharpe ratios are modest

A Sharpe of 0.42 is not exceptional for a single-factor strategy. Real trend-following CTAs (commodity trading advisors) operate across 50–200 instruments in futures markets, achieving portfolio-level Sharpes of 0.5–0.8 through far greater diversification. This 6-asset equity/bond/gold universe is deliberately simplified — the methodology is the point, not the performance ceiling.

Importantly, trend-following strategies like this have **low correlation to long-only equity** (typically 0.0 to 0.2 with the S&P 500). When added to a traditional 60/40 portfolio, even a modest-Sharpe trend strategy improves the portfolio's overall Sharpe meaningfully — this is why systematic trend is a standalone allocation at most large endowments and pension funds.

---

## Stress Periods Explained

The dashboard highlights three shaded stress periods. Here is what the strategy was doing during each one:

### GFC 2008 (September 2008 – March 2009)

The global financial crisis saw equity markets fall 50%+ globally. What happened to each signal:

- **SPY, QQQ, IWM, EEM**: All had negative 12-month trailing returns by late 2008. Signal went **short** — correctly betting on further declines.
- **TLT**: Treasuries rallied as a flight-to-safety trade. 12-month return turned **positive**. Signal went **long** — correctly riding the bond rally.
- **GLD**: Gold initially sold off (forced deleveraging) then recovered. Mixed signal.

The unscaled strategy was correct directionally but held full ±1 positions as daily volatility spiked to 4–5% per day (vs. the normal 1%). The vol-scaled strategy cut exposure to roughly 20–25% of normal size, reducing both the gains and the whipsaw losses from the extreme daily swings.

### COVID 2020 (February 15 – April 30, 2020)

The fastest bear market in history: the S&P 500 fell 34% in 33 days, then recovered almost entirely within weeks.

This is the stress period where **TSMOM is most challenged**. The 12-month lookback meant signals were still **long equities** going into the crash — the prior 12 months had been strong. The strategy got caught long into the drawdown.

Then equities reversed sharply before the trailing return flipped negative. The vol-scaled strategy reduced equity exposure as vol spiked (SPY vol went to ~80%), which cushioned the drawdown. The unscaled strategy held full long exposure throughout.

This is the key criticism of TSMOM: it is designed for sustained trends, not V-shaped recoveries. In 2020, the diversification into TLT (which rallied strongly) saved the portfolio from worse outcomes.

### Rate Shock 2022 (January – December 2022)

The Fed raised rates from 0.25% to 4.5% in a single year — the fastest hiking cycle in 40 years. Equities fell ~20%, and crucially, **bonds fell simultaneously** (TLT fell ~30%), breaking the traditional stock-bond negative correlation.

The 12-month signals were initially long both equities and bonds entering 2022. As trailing returns turned negative through Q1, the signals flipped short on both — catching the back half of the selloff. By mid-2022, short signals on SPY, QQQ, TLT, and EEM were all in place.

This is a period where TSMOM showed its value: in an environment where the traditional 60/40 portfolio was down 15–20% (both halves falling together), a short bond/short equity signal delivered positive returns in the second half of the year.

---

## Dashboard Guide

Open `tsmom_dashboard.html` in any browser.

**Panel 1: Cumulative Returns (log scale)**
Cumulative wealth of $1 invested at the start. Log scale is used so a move from $1 to $2 looks the same size as a move from $2 to $4 — proportional, not absolute. The divergence between the two lines is where the vol-scaler is doing work. Shaded regions are the three stress periods.

**Panel 2: Rolling 60-Day Sharpe**
Regime-level view of risk-adjusted performance. When this dips below zero, the strategy is losing money on a risk-adjusted basis in that 60-day window. Useful for identifying which market environments are hostile to momentum (typically: sharp reversals, rangebound chop).

**Panel 3: Drawdown Curve**
The "underwater" chart. Every time the curve reaches 0%, the strategy has recovered to a new equity high. The deeper and longer the trough, the worse the drawdown experience. This panel makes it viscerally clear how much smoother the vol-scaled version's drawdown profile is.

**Panel 4: Per-Asset Realized Volatility**
60-day rolling annualized vol for each of the 6 assets. You can see all assets spike together during the 2008 and 2020 stress periods — this is exactly when the vol scaler kicks in most aggressively to cut position sizes. Notice TLT (bonds) is normally the lowest-vol asset, which is why it often receives the largest position sizes in the vol-scaled strategy.

**Summary Stats Table**
At the top of the dashboard. Side-by-side comparison of all key metrics.

---

## File Structure & Usage

```
.
├── tsmom.py            # Core pipeline: data, signal, vol scaler, portfolio, analytics
├── dashboard.py        # Plotly dashboard generator
├── tsmom_dashboard.html  # Output: open in browser
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the pipeline (terminal output only)
```bash
python tsmom.py
```

### Generate the full HTML dashboard
```bash
python dashboard.py
```
Then open `tsmom_dashboard.html` in any browser. No server required — the file is fully self-contained.

---

## References

- Moskowitz, T., Ooi, Y.H., & Pedersen, L.H. (2012). *Time Series Momentum.* Journal of Financial Economics, 104(2), 228–250. — The foundational paper this strategy is based on.
- Jegadeesh, N., & Titman, S. (1993). *Returns to Buying Winners and Selling Losers.* Journal of Finance, 48(1), 65–91. — Original cross-sectional momentum paper that preceded TSMOM.
- Hurst, B., Ooi, Y.H., & Pedersen, L.H. (2017). *A Century of Evidence on Trend-Following Investing.* AQR Capital Management. — Extends the TSMOM evidence to 100 years of data across asset classes.
- Baz, J., Granger, N., Harvey, C.R., Le Roux, N., & Rattray, S. (2015). *Dissecting Investment Strategies in the Cross Section and Time Series.* SSRN 2695101. — Man AHL research on decomposing trend signals.
