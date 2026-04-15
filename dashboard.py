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

# ─────────────────────────────────────────────────────────────────────────────
# COLOR SCHEME
# ─────────────────────────────────────────────────────────────────────────────

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
# DASHBOARD ASSEMBLER
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard(results: dict, output_path: str = "tsmom_dashboard.html"):
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
    .badge-ml   {{ background: rgba(63,185,80,0.15);  color: {COLOR_LR}; }}
    .badge-base {{ background: rgba(76,155,232,0.15); color: {COLOR_UNSCALED}; }}

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
    <button class="tab-btn active"   id="btn-tsmom" onclick="switchTab('tsmom')">TSMOM Strategy</button>
    <button class="tab-btn ml-tab"   id="btn-ml"    onclick="switchTab('ml')">ML Comparison</button>
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

      // Plotly charts inside hidden divs need a resize signal when revealed.
      requestAnimationFrame(() => {{
        const panel = document.getElementById('tab-' + id);
        panel.querySelectorAll('.plotly-graph-div').forEach(div => {{
          if (window.Plotly) Plotly.relayout(div, {{autosize: true}});
        }});
      }});
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
    results    = run_pipeline()
    ml_results = run_ml_pipeline(results["log_returns"])
    results.update(ml_results)
    build_dashboard(results, output_path="index.html")
    print("\nOpen index.html in your browser to view the dashboard.")
