"""
TSMOM + ML Strategy Dashboard
==============================
Generates a self-contained HTML dashboard comparing:
  - Unscaled TSMOM (rule-based, no vol scaling)
  - Vol-Scaled TSMOM (rule-based, vol-targeting)
  - Logistic Regression signal + vol scaling
  - Random Forest signal + vol scaling
  - XGBoost signal + vol scaling

Run:
    python dashboard.py
Produces 'tsmom_dashboard.html' in the current directory.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tsmom import run_pipeline, STRESS_PERIODS, ASSETS
from ml_engine import run_ml_pipeline, ML_MODELS, FEATURE_COLS
from analyst_data import run_analyst_pipeline, WATCHLIST

# ─────────────────────────────────────────────────────────────────────────────
# COLOR SCHEME
# ─────────────────────────────────────────────────────────────────────────────

COLOR_ANALYST  = "#A371F7"              # Purple — Analyst tab accent
COLOR_UNSCALED = "#4C9BE8"              # Blue  — Unscaled TSMOM
COLOR_SCALED   = "#F28C28"              # Amber — Vol-Scaled TSMOM
COLOR_LR       = "#3FB950"              # Green — Logistic Regression
COLOR_RF       = "#D2A8FF"              # Lavender — Random Forest
COLOR_XGB      = "#FFA657"              # Warm orange — XGBoost
COLOR_STRESS   = "rgba(255, 80, 80, 0.10)"
COLOR_BG       = "#0D1117"
COLOR_PANEL    = "#161B22"
COLOR_GRID     = "#21262D"
COLOR_TEXT     = "#E6EDF3"
COLOR_SUBTEXT  = "#8B949E"

# Display metadata for all 5 strategies (order matters for tables/charts)
STRATEGIES = [
    ("Unscaled TSMOM",      "unscaled",            COLOR_UNSCALED),
    ("Vol-Scaled TSMOM",    "scaled",              COLOR_SCALED),
    ("Logistic Regression", "logistic_regression", COLOR_LR),
    ("Random Forest",       "random_forest",       COLOR_RF),
    ("XGBoost",             "xgboost",             COLOR_XGB),
]


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _stress_shapes() -> list:
    """Plotly shape dicts for stress-period shading bands."""
    shapes = []
    for _, start, end in STRESS_PERIODS:
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=start, x1=end, y0=0, y1=1,
            fillcolor=COLOR_STRESS, line_width=0, layer="below",
        ))
    return shapes


def _stress_annotations() -> list:
    """Text annotations labeling each stress period."""
    annotations = []
    for label, start, end in STRESS_PERIODS:
        mid = str(
            np.datetime64(start)
            + (np.datetime64(end) - np.datetime64(start)) // 2
        )[:10]
        annotations.append(dict(
            x=mid, y=0.985, xref="x", yref="paper",
            text=f"<i>{label}</i>", showarrow=False,
            font=dict(size=9, color="rgba(255,120,120,0.9)"),
            xanchor="center",
        ))
    return annotations


def _line(name: str, color: str, width: float = 1.8) -> dict:
    """Return shared trace kwargs for consistent line styling."""
    return dict(name=name, line=dict(color=color, width=width))


def _apply_dark_axes(fig):
    """Apply dark grid / spike styling to all axes."""
    fig.update_xaxes(
        gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID,
        showspikes=True, spikecolor=COLOR_SUBTEXT, spikethickness=1,
    )
    fig.update_yaxes(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID)


def _parse_pct(s: str) -> float:
    """Parse a formatted stat like '3.25%' or '-16.03%' to float (e.g., 3.25)."""
    return float(s.strip().rstrip("%"))


def _parse_float(s: str) -> float:
    """Parse a plain float string like '0.42'."""
    return float(s.strip())


# ─────────────────────────────────────────────────────────────────────────────
# EXISTING TSMOM CHARTS (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def build_cumulative_returns_chart(fig, results: dict, row: int, col: int):
    """Cumulative return traces (log scale) for the 2 TSMOM baselines."""
    fig.add_trace(go.Scatter(
        x=results["cum_unscaled"].index, y=results["cum_unscaled"].values,
        **_line("Unscaled TSMOM", COLOR_UNSCALED),
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=results["cum_scaled"].index, y=results["cum_scaled"].values,
        **_line("Vol-Scaled TSMOM", COLOR_SCALED),
    ), row=row, col=col)
    fig.update_yaxes(title_text="Cumulative Return (log)", type="log", row=row, col=col)


def build_rolling_sharpe_chart(fig, results: dict, row: int, col: int):
    """Rolling 60-day Sharpe for the 2 TSMOM baselines."""
    fig.add_hline(y=0, line_dash="dot", line_color=COLOR_SUBTEXT,
                  line_width=1, row=row, col=col)
    fig.add_trace(go.Scatter(
        x=results["sharpe_unscaled"].index, y=results["sharpe_unscaled"].values,
        showlegend=False, **_line("Unscaled TSMOM", COLOR_UNSCALED),
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=results["sharpe_scaled"].index, y=results["sharpe_scaled"].values,
        showlegend=False, **_line("Vol-Scaled TSMOM", COLOR_SCALED),
    ), row=row, col=col)
    fig.update_yaxes(title_text="Rolling 60-Day Sharpe", row=row, col=col)


def build_drawdown_chart(fig, results: dict, row: int, col: int):
    """Underwater drawdown curves for the 2 TSMOM baselines."""
    dd_u = results["dd_unscaled"]
    dd_s = results["dd_scaled"]
    fig.add_trace(go.Scatter(
        x=dd_u.index, y=dd_u.values * 100, fill="tozeroy",
        fillcolor="rgba(76,155,232,0.15)", showlegend=False,
        **_line("Unscaled TSMOM", COLOR_UNSCALED),
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=dd_s.index, y=dd_s.values * 100, fill="tozeroy",
        fillcolor="rgba(242,140,40,0.15)", showlegend=False,
        **_line("Vol-Scaled TSMOM", COLOR_SCALED),
    ), row=row, col=col)
    fig.update_yaxes(title_text="Drawdown (%)", row=row, col=col)


def build_rolling_vol_chart(fig, results: dict, row: int, col: int):
    """Per-asset 60-day rolling realized volatility."""
    realized_vol = results["realized_vol"]
    asset_colors = [
        "#58A6FF", "#F78166", "#3FB950", "#D2A8FF",
        "#FFA657", "#79C0FF", "#56D364", "#FF7B72",
    ]
    for i, col_name in enumerate(realized_vol.columns):
        fig.add_trace(go.Scatter(
            x=realized_vol.index,
            y=(realized_vol[col_name] * 100).values,
            name=col_name,
            line=dict(color=asset_colors[i % len(asset_colors)], width=1.2),
            opacity=0.8,
        ), row=row, col=col)
    fig.update_yaxes(title_text="Realized Vol (%, ann.)", row=row, col=col)


# ─────────────────────────────────────────────────────────────────────────────
# ML COMPARISON CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def build_ml_cumulative_chart(fig, results: dict, row: int, col: int):
    """
    All 5 strategy cumulative returns on one chart (log scale).

    TSMOM baselines included for direct comparison. ML series start ~3 years
    later due to the walk-forward burn-in; Plotly handles sparse series cleanly.
    """
    for display, key, color in STRATEGIES:
        series = results[f"cum_{key}"]
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            **_line(display, color),
        ), row=row, col=col)
    fig.update_yaxes(title_text="Cumulative Return (log)", type="log", row=row, col=col)


def build_ml_sharpe_chart(fig, results: dict, row: int, col: int):
    """Rolling 60-day Sharpe for ML models only (3 lines)."""
    fig.add_hline(y=0, line_dash="dot", line_color=COLOR_SUBTEXT,
                  line_width=1, row=row, col=col)
    ml_only = [s for s in STRATEGIES if s[1] not in ("unscaled", "scaled")]
    for display, key, color in ml_only:
        series = results[f"sharpe_{key}"]
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            showlegend=False, **_line(display, color),
        ), row=row, col=col)
    fig.update_yaxes(title_text="Rolling Sharpe (ML)", row=row, col=col)


def build_ml_drawdown_chart(fig, results: dict, row: int, col: int):
    """Drawdown curves for all 5 strategies."""
    fill_alpha = {
        "unscaled":            "rgba(76,155,232,0.12)",
        "scaled":              "rgba(242,140,40,0.12)",
        "logistic_regression": "rgba(63,185,80,0.12)",
        "random_forest":       "rgba(210,168,255,0.12)",
        "xgboost":             "rgba(255,166,87,0.12)",
    }
    for display, key, color in STRATEGIES:
        dd = results[f"dd_{key}"]
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values * 100, fill="tozeroy",
            fillcolor=fill_alpha[key], showlegend=False,
            **_line(display, color),
        ), row=row, col=col)
    fig.update_yaxes(title_text="Drawdown (%)", row=row, col=col)


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def build_stats_table_html(results: dict, strategies=None) -> str:
    """
    Render the performance summary as a plain HTML table string.

    Plain HTML is used instead of a Plotly table because Plotly figures
    rendered inside display:none containers fail to size themselves correctly
    and require brittle JS workarounds to fix. A native HTML table is
    always visible immediately regardless of tab state.

    Parameters
    ----------
    strategies : list of (display, key, color) tuples, or None for all.

    Returns
    -------
    str  — raw HTML fragment (no <html>/<body> wrapper)
    """
    if strategies is None:
        strategies = STRATEGIES
    metrics = ["Ann. Return", "Ann. Vol", "Sharpe Ratio", "Max Drawdown", "Calmar Ratio"]

    # Header row
    header_cells = '<th>Metric</th>'
    for display, _, color in strategies:
        header_cells += f'<th style="color:{color}">{display}</th>'

    # Data rows
    rows = ""
    for metric in metrics:
        row = f'<td class="metric-name">{metric}</td>'
        for display, key, color in strategies:
            val = results[f"stats_{key}"][metric]
            row += f'<td style="color:{color}">{val}</td>'
        rows += f"<tr>{row}</tr>"

    return f"""
<table class="stats-table">
  <thead><tr>{header_cells}</tr></thead>
  <tbody>{rows}</tbody>
</table>"""


def build_metrics_comparison_chart(results: dict) -> go.Figure:
    """
    Three-panel grouped bar chart: Sharpe Ratio, Ann. Return (%), Max Drawdown (%).

    Each panel has a bar per strategy so different scales don't compress the view.
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Sharpe Ratio", "Ann. Return (%)", "Max Drawdown (%)"],
        horizontal_spacing=0.08,
    )

    metrics_cfg = [
        ("Sharpe Ratio",  _parse_float),
        ("Ann. Return",   _parse_pct),
        ("Max Drawdown",  _parse_pct),
    ]

    for col_idx, (metric_key, parser) in enumerate(metrics_cfg, start=1):
        for display, key, color in STRATEGIES:
            stats = results[f"stats_{key}"]
            val   = parser(stats[metric_key])
            fig.add_trace(go.Bar(
                name=display,
                x=[display],
                y=[val],
                marker_color=color,
                showlegend=(col_idx == 1),
                text=[f"{val:.2f}"],
                textposition="outside",
                textfont=dict(color=COLOR_TEXT, size=10),
            ), row=1, col=col_idx)

    fig.update_layout(
        barmode="group",
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_PANEL,
        font=dict(color=COLOR_TEXT, size=11),
        height=380,
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=-0.18,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11),
        ),
        margin=dict(l=40, r=20, t=50, b=80),
    )
    fig.update_xaxes(showticklabels=False, gridcolor=COLOR_GRID)
    fig.update_yaxes(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID)
    return fig


def build_feature_importance_chart(results: dict) -> go.Figure:
    """
    Grouped bar chart of normalized feature importances for all 3 ML models.

    LR: |coefficient| normalized to sum=1
    RF: mean decrease in impurity (normalized)
    XGB: gain-based importance (normalized)

    All values are normalized so the three models are visually comparable.
    """
    importances = results.get("feature_importances", {})
    if not importances:
        fig = go.Figure()
        fig.update_layout(
            title="Feature Importances (unavailable)",
            paper_bgcolor=COLOR_BG, height=300,
        )
        return fig

    ml_colors  = [COLOR_LR, COLOR_RF, COLOR_XGB]
    model_names = list(ML_MODELS.keys())

    fig = go.Figure()
    for name, color in zip(model_names, ml_colors):
        if name not in importances:
            continue
        series = importances[name]
        fig.add_trace(go.Bar(
            name=name,
            x=FEATURE_COLS,
            y=series.values,
            marker_color=color,
            text=[f"{v:.3f}" for v in series.values],
            textposition="outside",
            textfont=dict(size=9, color=COLOR_TEXT),
        ))

    fig.update_layout(
        barmode="group",
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_PANEL,
        font=dict(color=COLOR_TEXT, size=11),
        height=420,
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=1.08,
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(title="Feature", gridcolor=COLOR_GRID),
        yaxis=dict(title="Normalized Importance", gridcolor=COLOR_GRID,
                   zerolinecolor=COLOR_GRID),
        margin=dict(l=50, r=20, t=60, b=60),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ANALYST TAB — CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_sentiment_chart(analyst: dict) -> go.Figure:
    """
    Horizontal bar chart of ML composite sentiment scores (0-100) per ticker.
    Color-coded: green (>70), amber (40-70), red (<40).
    """
    sentiment = analyst["sentiment"].copy()
    sentiment = sentiment.sort_values("sentiment_score", ascending=True)

    colors = []
    for s in sentiment["sentiment_score"]:
        if s >= 70:
            colors.append("#3FB950")
        elif s >= 40:
            colors.append("#F28C28")
        else:
            colors.append("#F85149")

    fig = go.Figure(go.Bar(
        x=sentiment["sentiment_score"],
        y=sentiment.index,
        orientation="h",
        marker_color=colors,
        text=[f"{s:.0f}" for s in sentiment["sentiment_score"]],
        textposition="outside",
        textfont=dict(color=COLOR_TEXT, size=11),
        customdata=sentiment["sentiment_label"].astype(str),
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<br>%{customdata}<extra></extra>",
    ))

    fig.add_vline(x=70, line_dash="dot", line_color="#3FB950", line_width=1,
                  annotation_text="Bullish", annotation_font_color="#3FB950",
                  annotation_position="top right")
    fig.add_vline(x=40, line_dash="dot", line_color="#F28C28", line_width=1,
                  annotation_text="Neutral", annotation_font_color="#F28C28",
                  annotation_position="top right")

    fig.update_layout(
        title=dict(text="<b>ML Composite Sentiment Score</b><br><sup>Rating quality (40%) + Price target upside (40%) + Buy ratio (20%)</sup>",
                   font=dict(size=14, color=COLOR_TEXT), x=0.5, xanchor="center"),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_PANEL,
        font=dict(color=COLOR_TEXT, size=11),
        height=420,
        xaxis=dict(title="Score (0-100)", range=[0, 110], gridcolor=COLOR_GRID),
        yaxis=dict(gridcolor=COLOR_GRID),
        margin=dict(l=60, r=60, t=80, b=40),
    )
    return fig


def build_price_target_chart(analyst: dict) -> go.Figure:
    """
    Horizontal analyst price target range chart.
    Shows low-to-high range bar, mean target diamond, current price circle.
    Color: green if current < mean (upside), red if current > mean (downside).
    """
    snap = analyst["snapshots"].dropna(subset=["target_mean", "current_price"]).copy()
    snap = snap.sort_values("upside_pct", ascending=True)
    tickers = snap.index.tolist()

    fig = go.Figure()

    for ticker in tickers:
        row     = snap.loc[ticker]
        cur     = row["current_price"]
        mean    = row["target_mean"]
        high    = row["target_high"]
        low     = row["target_low"]
        upside  = row["upside_pct"]
        color   = "#3FB950" if upside >= 0 else "#F85149"

        # Range bar: low -> high
        fig.add_trace(go.Scatter(
            x=[low, high], y=[ticker, ticker],
            mode="lines",
            line=dict(color="rgba(139,148,158,0.5)", width=10),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Mean target diamond
        fig.add_trace(go.Scatter(
            x=[mean], y=[ticker],
            mode="markers",
            marker=dict(symbol="diamond", size=14, color=color,
                        line=dict(color=COLOR_TEXT, width=1)),
            name="Mean Target" if ticker == tickers[0] else None,
            showlegend=(ticker == tickers[0]),
            hovertemplate=f"<b>{ticker}</b><br>Mean Target: ${mean:.0f}<br>Upside: {upside:+.1f}%<extra></extra>",
        ))
        # Current price circle
        fig.add_trace(go.Scatter(
            x=[cur], y=[ticker],
            mode="markers",
            marker=dict(symbol="circle", size=10, color=COLOR_UNSCALED,
                        line=dict(color=COLOR_TEXT, width=1)),
            name="Current Price" if ticker == tickers[0] else None,
            showlegend=(ticker == tickers[0]),
            hovertemplate=f"<b>{ticker}</b><br>Current: ${cur:.2f}<extra></extra>",
        ))
        # Upside annotation
        fig.add_annotation(
            x=max(high, cur) * 1.01, y=ticker,
            text=f"<b>{upside:+.1f}%</b>",
            showarrow=False,
            font=dict(color=color, size=10),
            xanchor="left",
        )

    fig.update_layout(
        title=dict(text="<b>Analyst Price Target Ranges</b><br><sup>Line: low-to-high range | Diamond: mean target | Circle: current price</sup>",
                   font=dict(size=14, color=COLOR_TEXT), x=0.5, xanchor="center"),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_PANEL,
        font=dict(color=COLOR_TEXT, size=11),
        height=max(350, len(tickers) * 50),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.06,
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Price ($)", gridcolor=COLOR_GRID),
        yaxis=dict(gridcolor=COLOR_GRID),
        hovermode="y unified",
        margin=dict(l=60, r=80, t=90, b=40),
    )
    return fig


def build_rating_breakdown_chart(analyst: dict) -> go.Figure:
    """
    Stacked horizontal bar: Strong Buy / Buy / Hold / Sell / Strong Sell per ticker.
    Sorted by bullish ratio descending.
    """
    bd = analyst["breakdowns"].copy()
    bd["total"] = bd[["strong_buy", "buy", "hold", "sell", "strong_sell"]].sum(axis=1).clip(lower=1)
    bd = bd.sort_values("buy_pct_now", ascending=True)
    tickers = bd.index.tolist()

    segments = [
        ("Strong Buy", "strong_buy", "#3FB950"),
        ("Buy",        "buy",        "#79C0FF"),
        ("Hold",       "hold",       "#F28C28"),
        ("Sell",       "sell",       "#FF7B72"),
        ("Strong Sell","strong_sell","#F85149"),
    ]

    fig = go.Figure()
    for label, col, color in segments:
        pct = (bd[col].fillna(0) / bd["total"] * 100).round(1)
        fig.add_trace(go.Bar(
            name=label,
            x=pct,
            y=tickers,
            orientation="h",
            marker_color=color,
            text=[f"{v:.0f}%" if v >= 5 else "" for v in pct],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        title=dict(text="<b>Analyst Rating Breakdown</b><br><sup>% of analysts per rating category</sup>",
                   font=dict(size=14, color=COLOR_TEXT), x=0.5, xanchor="center"),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_PANEL,
        font=dict(color=COLOR_TEXT, size=11),
        height=max(350, len(tickers) * 48),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.06,
                    bgcolor="rgba(0,0,0,0)", traceorder="normal"),
        xaxis=dict(title="% of Analysts", range=[0, 100], gridcolor=COLOR_GRID),
        yaxis=dict(gridcolor=COLOR_GRID),
        margin=dict(l=60, r=40, t=90, b=40),
    )
    return fig


def build_revision_drift_chart(analyst: dict) -> go.Figure:
    """
    Two-panel chart:
      Left:  Target price revision trend (avg $ change in last 30 days)
      Right: Rating drift (buy% change over 3 months)
    """
    rev   = analyst["revision_trend"]
    drift = analyst["rating_drift"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Avg Target Revision (Last 30 Days, $)", "Consensus Drift (3-Month Buy% Change)"],
        horizontal_spacing=0.12,
    )

    # Panel 1: Target revision
    if not rev.empty:
        rev_sorted = rev["avg_target_change"].sort_values()
        fig.add_trace(go.Bar(
            x=rev_sorted.values,
            y=rev_sorted.index,
            orientation="h",
            marker_color=["#3FB950" if v >= 0 else "#F85149" for v in rev_sorted.values],
            text=[f"${v:+.1f}" for v in rev_sorted.values],
            textposition="outside",
            textfont=dict(size=10, color=COLOR_TEXT),
            showlegend=False,
            hovertemplate="<b>%{y}</b><br>Avg revision: $%{x:+.1f}<extra></extra>",
        ), row=1, col=1)
        fig.add_vline(x=0, line_color=COLOR_SUBTEXT, line_width=1, row=1, col=1)

    # Panel 2: Rating drift
    if not drift.empty:
        drift_sorted = drift["drift"].sort_values()
        fig.add_trace(go.Bar(
            x=drift_sorted.values,
            y=drift_sorted.index,
            orientation="h",
            marker_color=["#3FB950" if v >= 0 else "#F85149" for v in drift_sorted.values],
            text=[f"{v:+.1f}pp" for v in drift_sorted.values],
            textposition="outside",
            textfont=dict(size=10, color=COLOR_TEXT),
            showlegend=False,
            hovertemplate="<b>%{y}</b><br>Buy% drift: %{x:+.1f}pp<extra></extra>",
        ), row=1, col=2)
        fig.add_vline(x=0, line_color=COLOR_SUBTEXT, line_width=1, row=1, col=2)

    fig.update_layout(
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_PANEL,
        font=dict(color=COLOR_TEXT, size=11),
        height=400,
        margin=dict(l=60, r=80, t=60, b=40),
    )
    fig.update_xaxes(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID)
    fig.update_yaxes(gridcolor=COLOR_GRID)
    return fig


def build_consensus_table_html(analyst: dict) -> str:
    """
    HTML summary table: one row per ticker with consensus metrics + sentiment score.
    Color-coded consensus and upside cells.
    """
    snap      = analyst["snapshots"].copy()
    sentiment = analyst["sentiment"].copy()

    CONSENSUS_COLOR = {
        "strong_buy":  "#3FB950", "buy":  "#79C0FF",
        "hold":        "#F28C28",
        "sell":        "#FF7B72", "strong_sell": "#F85149",
        "n/a":         COLOR_SUBTEXT,
    }

    header = """
<table class="stats-table analyst-table">
  <thead><tr>
    <th>Ticker</th><th>Price</th><th>Consensus</th>
    <th>Score</th><th>Analysts</th>
    <th>Target Low</th><th>Target Mean</th><th>Target High</th>
    <th>Upside %</th><th>Sentiment (ML)</th>
  </tr></thead>
  <tbody>"""

    rows = ""
    for ticker in snap.index:
        r   = snap.loc[ticker]
        s   = sentiment.loc[ticker] if ticker in sentiment.index else {}

        cons_key   = str(r.get("consensus_key", "n/a")).lower()
        cons_color = CONSENSUS_COLOR.get(cons_key, COLOR_SUBTEXT)
        upside     = r.get("upside_pct", float("nan"))
        up_color   = "#3FB950" if upside >= 15 else ("#F28C28" if upside >= 0 else "#F85149")
        sent_score = s.get("sentiment_score", float("nan")) if hasattr(s, "get") else float("nan")
        sent_label = s.get("sentiment_label", "") if hasattr(s, "get") else ""
        sent_color = "#3FB950" if sent_score >= 70 else ("#F28C28" if sent_score >= 40 else "#F85149")

        def _fmt(v, prefix="$", decimals=2):
            try:
                return f"{prefix}{v:.{decimals}f}" if not (v != v) else "—"
            except Exception:
                return "—"

        rows += f"""
<tr>
  <td style="font-weight:600;color:{COLOR_TEXT}">{ticker}</td>
  <td>{_fmt(r.get('current_price'))}</td>
  <td style="color:{cons_color};font-weight:600">{r.get('consensus_key','—').replace('_',' ').title()}</td>
  <td>{_fmt(r.get('consensus_score'), prefix='', decimals=2)}</td>
  <td>{int(r.get('n_analysts', 0)) if r.get('n_analysts') == r.get('n_analysts') else '—'}</td>
  <td>{_fmt(r.get('target_low'))}</td>
  <td>{_fmt(r.get('target_mean'))}</td>
  <td>{_fmt(r.get('target_high'))}</td>
  <td style="color:{up_color};font-weight:600">{f'{upside:+.1f}%' if upside==upside else '—'}</td>
  <td style="color:{sent_color};font-weight:600">{f'{sent_score:.0f} — {sent_label}' if sent_score==sent_score else '—'}</td>
</tr>"""

    return header + rows + "\n  </tbody>\n</table>"


def build_firm_accuracy_table_html(analyst: dict) -> str:
    """HTML leaderboard table of analyst firm directional accuracy."""
    fa = analyst["firm_accuracy"]
    if fa is None or fa.empty:
        return "<p style='color:#8B949E;font-size:0.8rem'>Insufficient historical data for firm accuracy calculation.</p>"

    header = """
<table class="stats-table analyst-table">
  <thead><tr>
    <th>#</th><th>Firm</th><th>Ratings Evaluated</th>
    <th>Correct</th><th>Accuracy</th><th>Avg Fwd Return</th>
  </tr></thead>
  <tbody>"""

    rows = ""
    for i, row in fa.iterrows():
        acc = row["accuracy_pct"]
        acc_color = "#3FB950" if acc >= 60 else ("#F28C28" if acc >= 50 else "#F85149")
        ret = row["avg_return"]
        ret_color = "#3FB950" if ret > 0 else "#F85149"
        rows += f"""
<tr>
  <td style="color:{COLOR_SUBTEXT}">{i+1}</td>
  <td style="font-weight:600;color:{COLOR_TEXT}">{row['Firm']}</td>
  <td>{int(row['n_ratings'])}</td>
  <td>{int(row['n_correct'])}</td>
  <td style="color:{acc_color};font-weight:600">{acc:.1f}%</td>
  <td style="color:{ret_color}">{ret:+.1f}%</td>
</tr>"""

    return header + rows + "\n  </tbody>\n</table>"


def build_upgrades_table_html(analyst: dict) -> str:
    """
    HTML table of recent upgrades/downgrades, sorted newest first.
    Green left border = upgrade, red = downgrade, gray = maintain.
    Includes JS click-to-sort on column headers.
    """
    upg = analyst["upgrades"]
    if upg is None or upg.empty:
        return "<p style='color:#8B949E;font-size:0.8rem'>No recent upgrade/downgrade data.</p>"

    ACTION_COLOR = {"up": "#3FB950", "down": "#F85149"}
    ACTION_LABEL = {"up": "Upgrade", "down": "Downgrade", "main": "Maintain",
                    "reit": "Reiterate", "init": "Initiate"}

    header = """
<table class="stats-table analyst-table sortable" id="upgrades-table">
  <thead><tr>
    <th onclick="sortTable('upgrades-table',0)">Date</th>
    <th onclick="sortTable('upgrades-table',1)">Ticker</th>
    <th onclick="sortTable('upgrades-table',2)">Firm</th>
    <th onclick="sortTable('upgrades-table',3)">Action</th>
    <th onclick="sortTable('upgrades-table',4)">From Grade</th>
    <th onclick="sortTable('upgrades-table',5)">To Grade</th>
    <th onclick="sortTable('upgrades-table',6)">New Target</th>
    <th onclick="sortTable('upgrades-table',7)">Prior Target</th>
    <th onclick="sortTable('upgrades-table',8)">Change ($)</th>
  </tr></thead>
  <tbody>"""

    rows = ""
    for _, row in upg.head(100).iterrows():
        action    = str(row.get("Action", "")).lower()
        color     = ACTION_COLOR.get(action, COLOR_SUBTEXT)
        label     = ACTION_LABEL.get(action, action.title())
        date_str  = str(row.get("date", ""))[:10]
        cur_tgt   = row.get("currentPriceTarget")
        prior_tgt = row.get("priorPriceTarget")
        change    = row.get("target_change")

        def _m(v, prefix="$"):
            try:
                return f"{prefix}{v:.0f}" if v == v else "—"
            except Exception:
                return "—"
        def _c(v):
            try:
                return f"${v:+.0f}" if v == v else "—"
            except Exception:
                return "—"

        chg_color = "#3FB950" if (change == change and change > 0) else ("#F85149" if (change == change and change < 0) else COLOR_SUBTEXT)
        rows += f"""
<tr style="border-left:3px solid {color}">
  <td style="color:{COLOR_SUBTEXT};font-size:0.78rem">{date_str}</td>
  <td style="font-weight:600;color:{COLOR_TEXT}">{row.get('ticker','')}</td>
  <td>{row.get('Firm','')}</td>
  <td style="color:{color};font-weight:600">{label}</td>
  <td style="color:{COLOR_SUBTEXT}">{row.get('FromGrade','—')}</td>
  <td style="color:{COLOR_TEXT}">{row.get('ToGrade','—')}</td>
  <td>{_m(cur_tgt)}</td>
  <td style="color:{COLOR_SUBTEXT}">{_m(prior_tgt)}</td>
  <td style="color:{chg_color};font-weight:600">{_c(change)}</td>
</tr>"""

    return header + rows + "\n  </tbody>\n</table>"


def build_analyst_tab_html(analyst: dict) -> str:
    """
    Assemble all analyst section HTML for Tab 3.
    Returns a complete HTML fragment (no <html>/<body> wrapper).
    """
    import plotly.io as pio

    sentiment_fig  = build_sentiment_chart(analyst)
    target_fig     = build_price_target_chart(analyst)
    breakdown_fig  = build_rating_breakdown_chart(analyst)
    rev_drift_fig  = build_revision_drift_chart(analyst)

    sentiment_html  = sentiment_fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
    target_html     = target_fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
    breakdown_html  = breakdown_fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
    rev_drift_html  = rev_drift_fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

    consensus_tbl  = build_consensus_table_html(analyst)
    accuracy_tbl   = build_firm_accuracy_table_html(analyst)
    upgrades_tbl   = build_upgrades_table_html(analyst)

    return f"""
    <h2>Consensus Overview <span class="badge badge-analyst">Live Data</span></h2>
    <div class="table-container">{consensus_tbl}</div>

    <h2>ML Sentiment Score <span class="badge badge-ml">Composite Signal</span></h2>
    <div class="chart-container">{sentiment_html}</div>

    <h2>Price Target Ranges <span class="badge badge-analyst">Low / Mean / High</span></h2>
    <div class="chart-container">{target_html}</div>

    <h2>Rating Breakdown <span class="badge badge-analyst">Buy vs Hold vs Sell</span></h2>
    <div class="chart-container">{breakdown_html}</div>

    <h2>Revision & Drift Analysis <span class="badge badge-ml">ML Analytics</span></h2>
    <div class="chart-container">{rev_drift_html}</div>

    <h2>Firm Accuracy Leaderboard <span class="badge badge-ml">63-Day Directional</span></h2>
    <div class="table-container">{accuracy_tbl}</div>

    <h2>Recent Upgrades & Downgrades <span class="badge badge-analyst">Last 6 Months</span></h2>
    <div class="table-container">{upgrades_tbl}</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD ASSEMBLER
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard(results: dict, analyst_results: dict = None, output_path: str = "tsmom_dashboard.html"):
    """
    Assemble all charts into a two-tab self-contained HTML dashboard.

    Tab 1 — TSMOM Strategy
        Performance table (2 baselines) + 4-row strategy chart

    Tab 2 — ML Comparison
        Performance table (all 5) + 3-row ML chart + metrics bar + feature importance

    Parameters
    ----------
    results : dict
        Merged output of tsmom.run_pipeline() + ml_engine.run_ml_pipeline().
    output_path : str
        Destination HTML file path.
    """
    TSMOM_STRATEGIES = [s for s in STRATEGIES if s[1] in ("unscaled", "scaled")]

    # ── Figure 1: TSMOM baselines (4 rows) ────────────────────────────────
    fig1 = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=[
            "Cumulative Returns (Log Scale)",
            "Rolling 60-Day Sharpe Ratio",
            "Drawdown Curve",
            f"60-Day Realized Volatility — {', '.join(ASSETS)}",
        ],
        vertical_spacing=0.06,
        row_heights=[0.30, 0.22, 0.22, 0.26],
    )
    build_cumulative_returns_chart(fig1, results, row=1, col=1)
    build_rolling_sharpe_chart(fig1, results, row=2, col=1)
    build_drawdown_chart(fig1, results, row=3, col=1)
    build_rolling_vol_chart(fig1, results, row=4, col=1)

    fig1.update_layout(
        title=dict(
            text=(
                "<b>Volatility-Scaled Time-Series Momentum (TSMOM)</b><br>"
                "<sup>Moskowitz, Ooi & Pedersen (2012)  |  2005-present</sup>"
            ),
            font=dict(size=17, color=COLOR_TEXT), x=0.5, xanchor="center",
        ),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_PANEL,
        font=dict(family="'SF Mono', 'Fira Code', monospace", color=COLOR_TEXT, size=11),
        height=1020,
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=1.01,
            bgcolor="rgba(0,0,0,0)", font=dict(size=12),
        ),
        hovermode="x unified",
        shapes=_stress_shapes(),
        annotations=_stress_annotations(),
        margin=dict(l=60, r=30, t=110, b=40),
    )
    _apply_dark_axes(fig1)

    # ── Figure 2: ML comparison (3 rows) ──────────────────────────────────
    fig2 = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=[
            "Cumulative Returns — All 5 Strategies (Log Scale)",
            "Rolling 60-Day Sharpe — ML Models",
            "Drawdown Curves — All 5 Strategies",
        ],
        vertical_spacing=0.07,
        row_heights=[0.42, 0.25, 0.33],
    )
    build_ml_cumulative_chart(fig2, results, row=1, col=1)
    build_ml_sharpe_chart(fig2, results, row=2, col=1)
    build_ml_drawdown_chart(fig2, results, row=3, col=1)

    fig2.update_layout(
        title=dict(
            text=(
                "<b>ML Signal Comparison vs TSMOM Baselines</b><br>"
                "<sup>Walk-forward  |  3-year burn-in  |  Quarterly retraining  |  "
                "Vol-scaled positions</sup>"
            ),
            font=dict(size=17, color=COLOR_TEXT), x=0.5, xanchor="center",
        ),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_PANEL,
        font=dict(family="'SF Mono', 'Fira Code', monospace", color=COLOR_TEXT, size=11),
        height=900,
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=1.01,
            bgcolor="rgba(0,0,0,0)", font=dict(size=12),
        ),
        hovermode="x unified",
        shapes=_stress_shapes(),
        annotations=_stress_annotations(),
        margin=dict(l=60, r=30, t=110, b=40),
    )
    _apply_dark_axes(fig2)

    # ── Standalone figures ─────────────────────────────────────────────────
    tsmom_stats_html = build_stats_table_html(results, strategies=TSMOM_STRATEGIES)
    full_stats_html  = build_stats_table_html(results, strategies=STRATEGIES)
    metrics_fig      = build_metrics_comparison_chart(results)
    imp_fig          = build_feature_importance_chart(results)

    # ── Analyst Tab 3 ─────────────────────────────────────────────────────
    analyst_tab_html = build_analyst_tab_html(analyst_results) if analyst_results else (
        "<p style='color:#8B949E;padding:40px;text-align:center'>"
        "Analyst data unavailable — run with analyst_results to enable this tab.</p>"
    )

    # ── Serialize (only fig1 carries the Plotly.js bundle) ────────────────
    main_html    = fig1.to_html(full_html=False, include_plotlyjs=True,
                                config={"responsive": True})
    ml_html      = fig2.to_html(full_html=False, include_plotlyjs=False,
                                config={"responsive": True})
    metrics_html = metrics_fig.to_html(full_html=False, include_plotlyjs=False)
    imp_html     = imp_fig.to_html(full_html=False, include_plotlyjs=False)

    # ── Full HTML page (two-tab layout) ───────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>TSMOM + ML Strategy Dashboard</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      background: {COLOR_BG};
      color: {COLOR_TEXT};
      font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
      padding: 0 32px 40px;
    }}

    /* ── Page header ── */
    .page-header {{
      padding: 28px 0 0;
      margin-bottom: 20px;
    }}
    .page-header h1 {{
      font-size: 1.25rem;
      font-weight: 700;
      color: {COLOR_TEXT};
      letter-spacing: 0.02em;
    }}
    .page-header p {{
      font-size: 0.78rem;
      color: {COLOR_SUBTEXT};
      margin-top: 4px;
    }}

    /* ── Tab bar ── */
    .tab-bar {{
      display: flex;
      gap: 0;
      border-bottom: 1px solid {COLOR_GRID};
      margin-bottom: 28px;
    }}
    .tab-btn {{
      background: none;
      border: none;
      border-bottom: 2px solid transparent;
      color: {COLOR_SUBTEXT};
      font-family: inherit;
      font-size: 0.82rem;
      font-weight: 500;
      padding: 12px 22px;
      cursor: pointer;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin-bottom: -1px;
      transition: color 0.15s, border-color 0.15s;
    }}
    .tab-btn:hover {{ color: {COLOR_TEXT}; }}
    .tab-btn.active {{
      color: {COLOR_UNSCALED};
      border-bottom-color: {COLOR_UNSCALED};
    }}
    .tab-btn.ml-tab.active {{
      color: {COLOR_LR};
      border-bottom-color: {COLOR_LR};
    }}
    .tab-btn.analyst-tab.active {{
      color: {COLOR_ANALYST};
      border-bottom-color: {COLOR_ANALYST};
    }}

    /* ── Tab panels ── */
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}

    /* ── Section headers inside tabs ── */
    h2 {{
      font-size: 0.78rem;
      font-weight: 600;
      color: {COLOR_SUBTEXT};
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin: 28px 0 10px;
      border-bottom: 1px solid {COLOR_GRID};
      padding-bottom: 6px;
    }}
    h2:first-child {{ margin-top: 0; }}

    /* ── Badges ── */
    .badge {{
      display: inline-block;
      font-size: 0.68rem;
      padding: 1px 7px;
      border-radius: 4px;
      margin-left: 8px;
      letter-spacing: 0.03em;
      vertical-align: middle;
      font-weight: 400;
    }}
    .badge-ml      {{ background: rgba(63,185,80,0.15);   color: {COLOR_LR}; }}
    .badge-base    {{ background: rgba(76,155,232,0.15);  color: {COLOR_UNSCALED}; }}
    .badge-analyst {{ background: rgba(163,113,247,0.15); color: {COLOR_ANALYST}; }}

    /* ── Analyst table extras ── */
    .analyst-table th {{ cursor: pointer; user-select: none; }}
    .analyst-table th:hover {{ color: {COLOR_TEXT}; }}

    .chart-container {{ width: 100%; }}
    .table-container  {{ width: 100%; margin-bottom: 4px; overflow-x: auto; }}

    /* ── Stats table ── */
    .stats-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
    }}
    .stats-table th, .stats-table td {{
      padding: 9px 16px;
      text-align: center;
      border: 1px solid {COLOR_GRID};
    }}
    .stats-table thead tr {{
      background: {COLOR_PANEL};
    }}
    .stats-table thead th {{
      font-weight: 600;
      letter-spacing: 0.04em;
      font-size: 0.78rem;
    }}
    .stats-table tbody tr:nth-child(even) {{
      background: rgba(255,255,255,0.02);
    }}
    .stats-table td.metric-name {{
      color: {COLOR_SUBTEXT};
      text-align: left;
      font-size: 0.78rem;
      letter-spacing: 0.04em;
    }}

    footer {{
      margin-top: 40px;
      padding-top: 16px;
      border-top: 1px solid {COLOR_GRID};
      font-size: 0.70rem;
      color: {COLOR_SUBTEXT};
      text-align: center;
      line-height: 1.7;
    }}
  </style>
</head>
<body>

  <div class="page-header">
    <h1>TSMOM + ML Strategy Dashboard</h1>
    <p>SPY, QQQ, GLD, TLT, EEM, IWM, NVDA, GOOG &nbsp;|&nbsp; 2005-present &nbsp;|&nbsp; Vol-scaled positions</p>
  </div>

  <!-- Tab bar -->
  <div class="tab-bar">
    <button class="tab-btn active"      id="btn-tsmom"    onclick="switchTab('tsmom')">TSMOM Strategy</button>
    <button class="tab-btn ml-tab"      id="btn-ml"       onclick="switchTab('ml')">ML Comparison</button>
    <button class="tab-btn analyst-tab" id="btn-analyst"  onclick="switchTab('analyst')">Analyst Ratings</button>
  </div>

  <!-- Tab 1: TSMOM -->
  <div id="tab-tsmom" class="tab-panel active">
    <h2>Performance Summary <span class="badge badge-base">Baselines</span></h2>
    <div class="table-container">{tsmom_stats_html}</div>

    <h2>Strategy Charts <span class="badge badge-base">Unscaled vs Vol-Scaled</span></h2>
    <div class="chart-container">{main_html}</div>
  </div>

  <!-- Tab 2: ML Comparison -->
  <div id="tab-ml" class="tab-panel">
    <h2>Performance Summary <span class="badge badge-ml">All 5 Strategies</span></h2>
    <div class="table-container">{full_stats_html}</div>

    <h2>Cumulative Returns & Drawdowns <span class="badge badge-ml">Walk-Forward</span></h2>
    <div class="chart-container">{ml_html}</div>

    <h2>Metrics Comparison <span class="badge badge-base">All Strategies</span></h2>
    <div class="chart-container">{metrics_html}</div>

    <h2>Feature Importance <span class="badge badge-ml">LR vs RF vs XGBoost</span></h2>
    <div class="chart-container">{imp_html}</div>
  </div>

  <!-- Tab 3: Analyst Ratings -->
  <div id="tab-analyst" class="tab-panel">
    {analyst_tab_html}
  </div>

  <footer>
    TSMOM: Moskowitz, T., Ooi, Y.H., &amp; Pedersen, L.H. (2012). &quot;Time Series Momentum.&quot;
    <i>Journal of Financial Economics</i>, 104(2), 228-250.<br>
    ML: Walk-forward validation &nbsp;|&nbsp; 3-year burn-in &nbsp;|&nbsp;
    Quarterly retraining &nbsp;|&nbsp; No lookahead bias<br>
    Data: Yahoo Finance (yfinance) &nbsp;|&nbsp; Built with Plotly, scikit-learn, XGBoost
  </footer>

  <script>
    function switchTab(id) {{
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.getElementById('tab-' + id).classList.add('active');
      document.getElementById('btn-' + id).classList.add('active');
      requestAnimationFrame(() => {{
        const panel = document.getElementById('tab-' + id);
        panel.querySelectorAll('.plotly-graph-div').forEach(div => {{
          if (window.Plotly) Plotly.relayout(div, {{autosize: true}});
        }});
      }});
    }}

    // Click-to-sort for analyst tables
    function sortTable(tableId, col) {{
      const table = document.getElementById(tableId);
      if (!table) return;
      const tbody = table.tBodies[0];
      const rows  = Array.from(tbody.rows);
      const asc   = table.dataset.sortCol == col && table.dataset.sortDir !== 'asc';
      rows.sort((a, b) => {{
        const va = a.cells[col]?.innerText.replace(/[$%+,]/g,'') || '';
        const vb = b.cells[col]?.innerText.replace(/[$%+,]/g,'') || '';
        const na = parseFloat(va), nb = parseFloat(vb);
        const cmp = !isNaN(na) && !isNaN(nb) ? na - nb : va.localeCompare(vb);
        return asc ? cmp : -cmp;
      }});
      rows.forEach(r => tbody.appendChild(r));
      table.dataset.sortCol = col;
      table.dataset.sortDir = asc ? 'asc' : 'desc';
    }}
  </script>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[Dashboard] Saved: {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results         = run_pipeline()
    ml_results      = run_ml_pipeline(results["log_returns"])
    results.update(ml_results)
    analyst_results = run_analyst_pipeline(WATCHLIST)
    build_dashboard(results, analyst_results=analyst_results, output_path="index.html")
    print("\nOpen index.html in your browser to view the dashboard.")
