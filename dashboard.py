"""
TSMOM Dashboard Generator
=========================
Generates a self-contained HTML dashboard using Plotly, visualizing
cumulative returns, rolling Sharpe, drawdowns, and realized volatility
for both the unscaled and vol-scaled TSMOM strategies.

Run:
    python dashboard.py
This will produce 'tsmom_dashboard.html' in the current directory.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tsmom import run_pipeline, STRESS_PERIODS, ASSETS

# ─────────────────────────────────────────────────────────────────────────────
# COLOR SCHEME
# ─────────────────────────────────────────────────────────────────────────────

COLOR_UNSCALED = "#4C9BE8"   # Blue  — unscaled TSMOM
COLOR_SCALED   = "#F28C28"   # Amber — vol-scaled TSMOM
COLOR_STRESS   = "rgba(255, 80, 80, 0.10)"
COLOR_BG       = "#0D1117"
COLOR_PANEL    = "#161B22"
COLOR_GRID     = "#21262D"
COLOR_TEXT     = "#E6EDF3"
COLOR_SUBTEXT  = "#8B949E"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _stress_shapes(x_min=None, x_max=None) -> list[dict]:
    """
    Build Plotly shape dicts for each stress period (vertical shaded bands).

    Parameters
    ----------
    x_min, x_max : ignored (Plotly handles axis-relative coords with xref='x')

    Returns
    -------
    list of dict
        Plotly layout shapes for shaded stress regions.
    """
    shapes = []
    for _, start, end in STRESS_PERIODS:
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=start,
                x1=end,
                y0=0,
                y1=1,
                fillcolor=COLOR_STRESS,
                line_width=0,
                layer="below",
            )
        )
    return shapes


def _stress_annotations(row: int, col: int) -> list[dict]:
    """
    Build text annotations labeling each stress period on a subplot.

    Parameters
    ----------
    row, col : int
        Subplot position (1-indexed) for annotation targeting.

    Returns
    -------
    list of dict
        Plotly annotation dicts.
    """
    annotations = []
    for label, start, end in STRESS_PERIODS:
        mid = str(
            (
                np.datetime64(start) + (np.datetime64(end) - np.datetime64(start)) / 2
            )
        )[:10]
        annotations.append(
            dict(
                x=mid,
                y=1.01,
                xref=f"x{row if row > 1 else ''}",
                yref="paper",
                text=f"<b>{label}</b>",
                showarrow=False,
                font=dict(size=9, color="rgba(255,100,100,0.8)"),
                xanchor="center",
            )
        )
    return annotations


def _common_trace_args(name: str, color: str) -> dict:
    """Return shared trace kwargs for consistent styling."""
    return dict(name=name, line=dict(color=color, width=1.8))


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_cumulative_returns_chart(fig, results: dict, row: int, col: int):
    """
    Add cumulative return traces (log scale) to the figure.

    Both strategies plotted on the same axes. Log scale preserves
    proportional distance and avoids visual distortion for long horizons.
    """
    cum_u = results["cum_unscaled"]
    cum_s = results["cum_scaled"]

    fig.add_trace(
        go.Scatter(
            x=cum_u.index, y=cum_u.values,
            **_common_trace_args("Unscaled TSMOM", COLOR_UNSCALED),
        ),
        row=row, col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=cum_s.index, y=cum_s.values,
            **_common_trace_args("Vol-Scaled TSMOM", COLOR_SCALED),
        ),
        row=row, col=col,
    )

    fig.update_yaxes(
        title_text="Cumulative Return (log scale)",
        type="log",
        row=row, col=col,
    )


def build_rolling_sharpe_chart(fig, results: dict, row: int, col: int):
    """
    Add rolling 60-day Sharpe ratio traces for both strategies.

    Includes a dashed zero-line for quick visual reference.
    """
    sharpe_u = results["sharpe_unscaled"]
    sharpe_s = results["sharpe_scaled"]

    fig.add_hline(y=0, line_dash="dot", line_color=COLOR_SUBTEXT, line_width=1, row=row, col=col)
    fig.add_trace(
        go.Scatter(
            x=sharpe_u.index, y=sharpe_u.values,
            **_common_trace_args("Unscaled TSMOM", COLOR_UNSCALED),
            showlegend=False,
        ),
        row=row, col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=sharpe_s.index, y=sharpe_s.values,
            **_common_trace_args("Vol-Scaled TSMOM", COLOR_SCALED),
            showlegend=False,
        ),
        row=row, col=col,
    )

    fig.update_yaxes(title_text="Rolling 60-Day Sharpe", row=row, col=col)


def build_drawdown_chart(fig, results: dict, row: int, col: int):
    """
    Add underwater (drawdown) curves for both strategies.

    Filled area charts emphasize the duration and severity of drawdowns.
    """
    dd_u = results["dd_unscaled"]
    dd_s = results["dd_scaled"]

    fig.add_trace(
        go.Scatter(
            x=dd_u.index, y=dd_u.values * 100,
            fill="tozeroy",
            fillcolor=f"rgba(76,155,232,0.15)",
            **_common_trace_args("Unscaled TSMOM", COLOR_UNSCALED),
            showlegend=False,
        ),
        row=row, col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=dd_s.index, y=dd_s.values * 100,
            fill="tozeroy",
            fillcolor=f"rgba(242,140,40,0.15)",
            **_common_trace_args("Vol-Scaled TSMOM", COLOR_SCALED),
            showlegend=False,
        ),
        row=row, col=col,
    )

    fig.update_yaxes(title_text="Drawdown (%)", row=row, col=col)


def build_rolling_vol_chart(fig, results: dict, row: int, col: int):
    """
    Add per-asset 60-day rolling realized volatility traces.

    Visualizes how each asset's vol fluctuates over time — the key
    input that drives position sizing in the vol-scaled strategy.
    """
    realized_vol = results["realized_vol"]

    # Use a muted palette for multiple assets
    asset_colors = [
        "#58A6FF", "#F78166", "#3FB950", "#D2A8FF",
        "#FFA657", "#79C0FF", "#56D364",
    ]

    for i, col_name in enumerate(realized_vol.columns):
        color = asset_colors[i % len(asset_colors)]
        fig.add_trace(
            go.Scatter(
                x=realized_vol.index,
                y=(realized_vol[col_name] * 100).values,
                name=col_name,
                line=dict(color=color, width=1.2),
                opacity=0.8,
            ),
            row=row, col=col,
        )

    fig.update_yaxes(title_text="Realized Vol (%, ann.)", row=row, col=col)


def build_stats_table(results: dict) -> go.Figure:
    """
    Build a standalone Plotly table figure with summary statistics.

    Returns
    -------
    go.Figure
        A Plotly figure containing only the stats table.
    """
    stats_u = results["stats_unscaled"]
    stats_s = results["stats_scaled"]

    metrics = ["Ann. Return", "Ann. Vol", "Sharpe Ratio", "Max Drawdown", "Calmar Ratio"]
    vals_u = [stats_u[m] for m in metrics]
    vals_s = [stats_s[m] for m in metrics]

    fig = go.Figure(
        go.Table(
            header=dict(
                values=["<b>Metric</b>", "<b>Unscaled TSMOM</b>", "<b>Vol-Scaled TSMOM</b>"],
                fill_color=COLOR_PANEL,
                font=dict(color=COLOR_TEXT, size=13),
                align="center",
                line_color=COLOR_GRID,
                height=36,
            ),
            cells=dict(
                values=[metrics, vals_u, vals_s],
                fill_color=COLOR_BG,
                font=dict(color=[COLOR_SUBTEXT, COLOR_UNSCALED, COLOR_SCALED], size=12),
                align="center",
                line_color=COLOR_GRID,
                height=32,
            ),
        )
    )

    fig.update_layout(
        paper_bgcolor=COLOR_BG,
        margin=dict(l=0, r=0, t=0, b=0),
        height=230,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD ASSEMBLER
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard(results: dict, output_path: str = "tsmom_dashboard.html"):
    """
    Assemble all charts into a single self-contained HTML dashboard.

    Layout (4 rows × 1 col):
        Row 1: Cumulative returns (log scale)
        Row 2: Rolling 60-day Sharpe ratio
        Row 3: Drawdown curves
        Row 4: Per-asset realized volatility

    Stress-period shaded bands are applied to rows 1–3.

    Parameters
    ----------
    results : dict
        Output from tsmom.run_pipeline().
    output_path : str
        File path for the output HTML file.
    """
    subplot_titles = [
        "Cumulative Returns (Log Scale)",
        "Rolling 60-Day Sharpe Ratio",
        "Drawdown Curve",
        f"60-Day Realized Volatility — {', '.join(ASSETS)}",
    ]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.06,
        row_heights=[0.30, 0.22, 0.22, 0.26],
    )

    # ── Add chart layers ───────────────────────────────────────────────────
    build_cumulative_returns_chart(fig, results, row=1, col=1)
    build_rolling_sharpe_chart(fig, results, row=2, col=1)
    build_drawdown_chart(fig, results, row=3, col=1)
    build_rolling_vol_chart(fig, results, row=4, col=1)

    # ── Stress-period shading (rows 1–3 share x-axis via shapes) ──────────
    # Plotly shapes on shared axes apply to all linked subplots automatically
    stress_shapes = []
    for _, start, end in STRESS_PERIODS:
        stress_shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=start,
                x1=end,
                y0=0,
                y1=1,
                fillcolor=COLOR_STRESS,
                line_width=0,
                layer="below",
            )
        )

    # ── Stress annotations on cumulative returns chart ─────────────────────
    annotations = []
    for label, start, end in STRESS_PERIODS:
        start_dt = np.datetime64(start)
        end_dt = np.datetime64(end)
        mid = str(start_dt + (end_dt - start_dt) // 2)[:10]
        annotations.append(
            dict(
                x=mid,
                y=0.985,
                xref="x",
                yref="paper",
                text=f"<i>{label}</i>",
                showarrow=False,
                font=dict(size=9, color="rgba(255,120,120,0.9)"),
                xanchor="center",
            )
        )

    # ── Global layout ─────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                "<b>Volatility-Scaled Time-Series Momentum (TSMOM)</b><br>"
                "<sup>Moskowitz, Ooi & Pedersen (2012) · SPY, QQQ, GLD, TLT, EEM, IWM · 2005–present</sup>"
            ),
            font=dict(size=18, color=COLOR_TEXT),
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_PANEL,
        font=dict(family="'SF Mono', 'Fira Code', monospace", color=COLOR_TEXT, size=11),
        height=1050,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.01,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        ),
        hovermode="x unified",
        shapes=stress_shapes,
        annotations=annotations,
        margin=dict(l=60, r=30, t=110, b=40),
    )

    fig.update_xaxes(
        gridcolor=COLOR_GRID,
        zerolinecolor=COLOR_GRID,
        showspikes=True,
        spikecolor=COLOR_SUBTEXT,
        spikethickness=1,
    )
    fig.update_yaxes(
        gridcolor=COLOR_GRID,
        zerolinecolor=COLOR_GRID,
    )

    # ── Build summary stats table as separate figure ───────────────────────
    stats_fig = build_stats_table(results)

    # ── Write self-contained HTML ──────────────────────────────────────────
    main_html = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
    stats_html = stats_fig.to_html(full_html=False, include_plotlyjs=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>TSMOM Strategy Dashboard</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: {COLOR_BG};
      color: {COLOR_TEXT};
      font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
      padding: 24px 32px;
    }}
    h2 {{
      font-size: 1.1rem;
      font-weight: 600;
      color: {COLOR_SUBTEXT};
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin: 28px 0 10px;
      border-bottom: 1px solid {COLOR_GRID};
      padding-bottom: 6px;
    }}
    .chart-container {{ width: 100%; }}
    .table-container {{
      width: 100%;
      max-width: 760px;
      margin: 0 auto;
    }}
    footer {{
      margin-top: 36px;
      padding-top: 16px;
      border-top: 1px solid {COLOR_GRID};
      font-size: 0.75rem;
      color: {COLOR_SUBTEXT};
      text-align: center;
    }}
  </style>
</head>
<body>
  <h2>Performance Summary</h2>
  <div class="table-container">{stats_html}</div>

  <h2>Strategy Charts</h2>
  <div class="chart-container">{main_html}</div>

  <footer>
    Strategy based on Moskowitz, T., Ooi, Y.H., &amp; Pedersen, L.H. (2012).
    &quot;Time Series Momentum.&quot; <i>Journal of Financial Economics</i>, 104(2), 228–250.
    &nbsp;|&nbsp; Data via Yahoo Finance (yfinance) &nbsp;|&nbsp; Built with Plotly
  </footer>
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
    results = run_pipeline()
    build_dashboard(results, output_path="tsmom_dashboard.html")
    print("\nOpen tsmom_dashboard.html in your browser to view the dashboard.")
