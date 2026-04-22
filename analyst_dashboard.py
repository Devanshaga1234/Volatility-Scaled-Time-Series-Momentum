"""
Standalone Analyst Ratings Dashboard
=====================================
Fetches Wall Street analyst data for WATCHLIST and builds a self-contained
HTML dashboard: analyst_dashboard.html

Also re-exported as Tab 3 inside tsmom_dashboard (via dashboard.py).

Run:
    python analyst_dashboard.py
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

from analyst_data import run_analyst_pipeline, WATCHLIST
from dashboard import (
    build_sentiment_chart,
    build_price_target_chart,
    build_rating_breakdown_chart,
    build_revision_drift_chart,
    build_consensus_table_html,
    build_firm_accuracy_table_html,
    build_upgrades_table_html,
    COLOR_BG, COLOR_PANEL, COLOR_GRID, COLOR_TEXT, COLOR_SUBTEXT,
    COLOR_UNSCALED, COLOR_LR, COLOR_ANALYST,
)


def build_analyst_dashboard(analyst: dict, output_path: str = "analyst_dashboard.html"):
    """
    Assemble all analyst charts and tables into a standalone HTML file.

    Sections
    --------
    1. Consensus Overview Table (HTML)
    2. ML Sentiment Score Chart
    3. Price Target Range Chart
    4. Rating Breakdown Chart
    5. Revision & Drift Analysis Charts
    6. Firm Accuracy Leaderboard (HTML)
    7. Recent Upgrades / Downgrades Feed (HTML, sortable)
    """
    today = date.today().strftime("%B %d, %Y")

    # ── Build Plotly figures ───────────────────────────────────────────────
    sentiment_fig  = build_sentiment_chart(analyst)
    target_fig     = build_price_target_chart(analyst)
    breakdown_fig  = build_rating_breakdown_chart(analyst)
    rev_drift_fig  = build_revision_drift_chart(analyst)

    # ── Serialize (first figure carries Plotly.js) ────────────────────────
    sentiment_html  = sentiment_fig.to_html(full_html=False, include_plotlyjs=True,
                                            config={"responsive": True})
    target_html     = target_fig.to_html(full_html=False, include_plotlyjs=False,
                                         config={"responsive": True})
    breakdown_html  = breakdown_fig.to_html(full_html=False, include_plotlyjs=False,
                                            config={"responsive": True})
    rev_drift_html  = rev_drift_fig.to_html(full_html=False, include_plotlyjs=False,
                                            config={"responsive": True})

    # ── HTML tables ───────────────────────────────────────────────────────
    consensus_tbl = build_consensus_table_html(analyst)
    accuracy_tbl  = build_firm_accuracy_table_html(analyst)
    upgrades_tbl  = build_upgrades_table_html(analyst)

    tickers_str = ", ".join(WATCHLIST)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Analyst Ratings Dashboard</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: {COLOR_BG};
      color: {COLOR_TEXT};
      font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
      padding: 24px 32px 48px;
    }}

    .page-header {{
      padding-bottom: 20px;
      border-bottom: 1px solid {COLOR_GRID};
      margin-bottom: 28px;
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

    h2 {{
      font-size: 0.78rem;
      font-weight: 600;
      color: {COLOR_SUBTEXT};
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin: 32px 0 10px;
      border-bottom: 1px solid {COLOR_GRID};
      padding-bottom: 6px;
    }}
    h2:first-child {{ margin-top: 0; }}

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
    .badge-analyst {{ background: rgba(163,113,247,0.15); color: {COLOR_ANALYST}; }}

    .chart-container {{ width: 100%; }}
    .table-container  {{ width: 100%; margin-bottom: 4px; overflow-x: auto; }}

    /* ── Shared table style ── */
    .stats-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
    }}
    .stats-table th, .stats-table td {{
      padding: 9px 14px;
      text-align: center;
      border: 1px solid {COLOR_GRID};
    }}
    .stats-table thead tr {{ background: {COLOR_PANEL}; }}
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
    }}
    .analyst-table th {{
      cursor: pointer;
      user-select: none;
    }}
    .analyst-table th:hover {{ color: {COLOR_TEXT}; }}

    footer {{
      margin-top: 48px;
      padding-top: 16px;
      border-top: 1px solid {COLOR_GRID};
      font-size: 0.70rem;
      color: {COLOR_SUBTEXT};
      text-align: center;
      line-height: 1.8;
    }}
  </style>
</head>
<body>

  <div class="page-header">
    <h1>Analyst Ratings Dashboard</h1>
    <p>{tickers_str} &nbsp;|&nbsp; {today} &nbsp;|&nbsp; Data: Yahoo Finance</p>
  </div>

  <h2>Consensus Overview <span class="badge badge-analyst">Live Data</span></h2>
  <div class="table-container">{consensus_tbl}</div>

  <h2>ML Sentiment Score <span class="badge badge-ml">Composite Signal</span></h2>
  <div class="chart-container">{sentiment_html}</div>

  <h2>Price Target Ranges <span class="badge badge-analyst">Low / Mean / High vs Current</span></h2>
  <div class="chart-container">{target_html}</div>

  <h2>Rating Breakdown <span class="badge badge-analyst">Buy vs Hold vs Sell</span></h2>
  <div class="chart-container">{breakdown_html}</div>

  <h2>Revision & Drift Analysis <span class="badge badge-ml">Target Revisions + Consensus Drift</span></h2>
  <div class="chart-container">{rev_drift_html}</div>

  <h2>Firm Accuracy Leaderboard <span class="badge badge-ml">63-Day Directional Accuracy</span></h2>
  <div class="table-container">{accuracy_tbl}</div>

  <h2>Recent Upgrades & Downgrades <span class="badge badge-analyst">Last 6 Months</span></h2>
  <div class="table-container">{upgrades_tbl}</div>

  <footer>
    Data: Yahoo Finance (yfinance) &nbsp;|&nbsp;
    ML Sentiment = Rating quality (40%) + Price target upside (40%) + Buy ratio (20%)<br>
    Firm accuracy evaluated over 63-day forward window &nbsp;|&nbsp;
    Last updated: {today}
  </footer>

  <script>
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

    print(f"[Analyst Dashboard] Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    analyst = run_analyst_pipeline(WATCHLIST)
    build_analyst_dashboard(analyst, output_path="analyst_dashboard.html")
    print("\nOpen analyst_dashboard.html in your browser.")
