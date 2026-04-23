# Multi-Strategy Quant + ML Portfolio Roadmap

## Context

Build a modular, research-first portfolio system on top of the existing TSMOM + ML framework, targeting high returns with controlled risk. Capital: starts at $30/week DCA, targeting $10k+ over time. Assets: US equities/ETFs.

**Goal:** Data-driven portfolio construction combining quant signals, ML, and analyst intelligence — proven in simulation first, live execution later.

### Honest Return Expectations

| Approach | Realistic Ann. Return | Risk |
|---|---|---|
| TSMOM + vol scaling (current) | 3–8% | Low drawdowns |
| ML signals on equities | 8–20% | Moderate |
| Options (covered calls + momentum) | 15–35% | Moderate-high |
| High-vol momentum (NVDA, crypto ETFs) | 20–60%+ | Very high drawdowns possible |
| All combined + risk management | 20–50% target, 50–100% in strong years | Managed |

50–100% annual returns are achievable but require concentrated bets or options leverage — which also introduces the possibility of significant drawdowns. Hard risk controls are non-negotiable on a small account.

---

## Architecture: 5 Core Modules

Builds on top of `tsmom.py` / `ml_engine.py` / `dashboard.py` / `analyst_data.py`.

---

### Phase 1 — `strategy_engine.py`: Multi-Strategy Signal Layer

Extends the existing signal system with additional strategies:

**Strategy A: Mean Reversion (RSI-based)**
- Assets trading >2 std devs from 20-day mean → short-term fade signal
- RSI < 30 = oversold long, RSI > 70 = overbought short
- 5–20 day hold period
- Reuses `_compute_rsi()` from `ml_engine.py`

**Strategy B: Earnings Momentum**
- Buy high-quality momentum names (top `cs_rank`) before earnings, exit after
- Uses yfinance earnings calendar
- High conviction, short hold, high return potential on small account

**Strategy C: Volatility Regime Filter**
- Uses VIX proxy (rolling vol of SPY) to switch strategy weights dynamically
- Low vol regime → increase momentum exposure
- High vol regime → shift to mean reversion + reduce size
- Reuses `compute_realized_vol()` from `tsmom.py`

**Strategy D: Options Signal Generator (research output only)**
- Identify covered call candidates (high IV rank, trending asset)
- Identify cash-secured put candidates (oversold, strong momentum)
- IV rank = current IV vs 52-week range (yfinance options chain)
- Output: signal + recommended strike/expiry — no auto-execution

Functions to build:
```python
compute_mean_reversion_signal(log_returns, window=20) -> pd.DataFrame
compute_vol_regime(log_returns, short=20, long=60) -> pd.Series  # 'low'/'high'/'crisis'
compute_strategy_weights(vol_regime) -> dict                      # {strategy: weight}
compute_options_candidates(prices, log_returns) -> pd.DataFrame
aggregate_signals(signals: dict, weights: dict) -> pd.DataFrame
```

---

### Phase 2 — `portfolio_optimizer.py`: Portfolio Construction Layer

Replaces the current equal-weight 1/N allocation.

**Risk Parity** (build first — no return forecasts needed)
- Each asset contributes equally to portfolio risk
- Rolling 252-day covariance matrix, rebalance quarterly

**Mean-Variance Optimization** (build second)
- Minimize portfolio variance subject to target return
- `scipy.optimize` (already available) for the QP solver

**Kelly Criterion** (for high-conviction ML bets)
- Half-Kelly sizing when ML model confidence > 70% (`P(up) > 0.70`)
- `f* = (p*b - q) / b`, use 0.5 × f* for safety

```python
compute_covariance_matrix(log_returns, window=252) -> pd.DataFrame
optimize_risk_parity(cov_matrix) -> pd.Series
optimize_mean_variance(expected_returns, cov_matrix, target_vol=0.15) -> pd.Series
compute_kelly_size(win_prob, win_return, loss_return) -> float
build_optimal_portfolio(log_returns, signals, method='risk_parity') -> pd.DataFrame
```

---

### Phase 3 — `risk_manager.py`: Risk Controls Layer

Critical for a small account — one bad trade on a small account is catastrophic.

**Position-level rules:**
- Max single position: 20% of portfolio NAV
- Max loss per position: 5% of NAV (stop-loss trigger)
- Max total leverage: 1.5×

**Portfolio-level rules:**
- Drawdown circuit breaker: if portfolio drops >15% from peak → cut all positions 50%
- Vol circuit: if realized 20-day portfolio vol exceeds 2× target → scale down all sizes
- Correlation check: if top-2 positions have correlation >0.85 → reduce smaller one

**Options-specific rules:**
- Max 2% of NAV on any single options trade
- Only sell covered calls on existing equity positions (no naked exposure)
- Max 25% of portfolio in options

```python
check_position_limits(positions, nav) -> pd.DataFrame
check_drawdown_circuit(cum_returns, threshold=0.15) -> bool
check_vol_circuit(portfolio_returns, target_vol, window=20) -> float  # size multiplier
compute_portfolio_var(positions, cov_matrix, confidence=0.95) -> float
apply_risk_filters(raw_positions, portfolio_state) -> pd.DataFrame
```

---

### Phase 4 — `portfolio_runner.py`: Master Orchestrator

Unified pipeline replacing the current two-call `run_pipeline()` + `run_ml_pipeline()` structure:

```python
def run_full_pipeline(config: dict) -> dict:
    # 1. Fetch data              — reuse fetch_prices() from tsmom.py
    # 2. Compute features        — reuse engineer_features() from ml_engine.py
    # 3. Generate all signals    — strategy_engine.py
    # 4. ML walk-forward signals — reuse run_walk_forward() from ml_engine.py
    # 5. Aggregate w/ vol-regime weights — strategy_engine.py
    # 6. Optimize portfolio weights      — portfolio_optimizer.py
    # 7. Apply risk filters              — risk_manager.py
    # 8. Compute returns + analytics     — reuse tsmom.py analytics
    # 9. Return unified results dict     — for dashboard
```

---

### Phase 5 — `portfolio_builder.py`: $30/Week Smart DCA Engine ✅ IMPLEMENTED

**Live module powering the Portfolio Builder tab in the analyst dashboard.**

Takes a weekly deposit amount (default $30, starting 2026-04-23) and outputs a
fully adaptive, data-driven portfolio allocation based on:

1. **Market regime detection** — SPY 200-day MA, 20-day realized vol, 60-day momentum
2. **ML sentiment overlay** — analyst ML scores (0–100) tilt individual stock weights
3. **Regime-based base allocation** — 4 regimes, each with a different asset mix
4. **Monte Carlo projection** — 500 simulations over 10 years, 3 confidence bands

**Market Regimes:**

| Regime | Trigger | Asset Mix |
|---|---|---|
| Bull / Low Vol | SPY > MA200, vol < 18% | Tech-heavy: QQQ 35%, SPY 20%, NVDA 15%, MSFT 10%, AAPL 10%, GLD 5%, AMZN 5% |
| Bull / High Vol | SPY > MA200, vol ≥ 18% | Balanced: SPY 30%, QQQ 20%, GLD 15%, MSFT 15%, AAPL 10%, TLT 10% |
| Bear / Low Vol | SPY < MA200, vol < 25% | Defensive: TLT 30%, GLD 25%, SPY 20%, MSFT 15%, AAPL 10% |
| Crisis | SPY < MA200, vol ≥ 25% | Safe haven: GLD 35%, TLT 30%, SPY 15%, MSFT 10%, AAPL 10% |

**Analyst Overlay:**
- ML sentiment score > 70 → overweight ticker by up to +5%
- ML sentiment score < 40 → underweight ticker by up to −5%
- Only applied to individual stocks (not ETFs)

**Projection Parameters (calibrated to historical regime returns):**

| Regime | Mean Ann. Return | Ann. Volatility |
|---|---|---|
| Bull / Low Vol | 18% | 12% |
| Bull / High Vol | 10% | 20% |
| Bear / Low Vol | 2% | 15% |
| Crisis | −5% | 35% |

```python
detect_market_regime() -> dict              # regime key + all indicators
fetch_current_prices(tickers) -> dict       # {ticker: price}
build_allocation(regime, analyst_scores) -> list  # [{ticker, weight, dollar, shares}]
project_growth_monte_carlo(deposit, regime, years=10, n_sims=500) -> dict
get_portfolio_summary(deposit=30.0, analyst_scores=None) -> dict  # API entry point
```

---

### Phase 6 — Dashboard: "Portfolio Analysis" Tab (backtest-focused)

Add a tab to the TSMOM dashboard (index.html):

- Efficient frontier plot (return vs vol scatter of strategy combinations)
- Rolling correlation heatmap (between assets)
- Strategy contribution chart (stacked bar: monthly PnL by strategy)
- Risk attribution table (which positions drive portfolio vol)
- Options candidates table (HTML table, live output)

Modify `dashboard.py` — add chart builders and wire into `build_dashboard()`.

---

### Phase 7 — Live Trading (future, after Phases 1–5 proven)

- Broker: **Alpaca** (commission-free, paper trading, Python SDK)
- Paper trade minimum 3 months before real money
- Broker adapter: wraps `portfolio_runner.py` outputs → Alpaca orders
- New dependency: `alpaca-py`

---

## Files to Create / Modify

| File | Status | Reuses From |
|---|---|---|
| `portfolio_builder.py` | **Done** | `analyst_data.py` (sentiment scores) |
| `strategy_engine.py` | Planned | `_compute_rsi()` (ml_engine), `compute_realized_vol()` (tsmom) |
| `portfolio_optimizer.py` | Planned | `compute_log_returns()` (tsmom) |
| `risk_manager.py` | Planned | `drawdown_series()`, `compute_realized_vol()` (tsmom) |
| `portfolio_runner.py` | Planned | All of tsmom.py, ml_engine.py, new modules |
| `dashboard.py` | Planned | Add Tab 4 (backtest portfolio analysis) |
| `analyst_dashboard.py` | **Done** | Added Portfolio Builder tab |
| `serve.py` | **Done** | Added `/api/portfolio` endpoint |
| `requirements.txt` | Planned | Add `cvxpy` (Phase 2), `alpaca-py` (Phase 7) |

## Key Functions — Do Not Rewrite

From `tsmom.py`: `fetch_prices`, `compute_log_returns`, `compute_realized_vol`, `build_portfolio_returns`, `cumulative_returns`, `rolling_sharpe`, `drawdown_series`, `summary_stats`

From `ml_engine.py`: `engineer_features`, `_compute_rsi`, `run_walk_forward`, `_train_models`

---

## Verification Checklist (each phase)

```bash
python portfolio_runner.py   # strategy breakdown + risk stats in terminal
python dashboard.py          # regenerate index.html
# open http://localhost:8050/tsmom, verify tabs render correctly
# open http://localhost:8050/analyst, verify Portfolio Builder tab
```

1. No lookahead bias — all new signals use `.shift(1)` before `build_portfolio_returns()`
2. No survivorship bias — assets fixed before backtest starts
3. Risk circuits fire — test by injecting a -20% return shock manually
4. Options IV rank — verify uses only historical options data, no future IV
5. Portfolio projection — verify Monte Carlo uses weekly, not daily, return parameters
