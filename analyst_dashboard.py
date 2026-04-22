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
      color: {COLOR_ANALYST};
      border-bottom-color: {COLOR_ANALYST};
    }}
    .tab-btn.lookup-tab.active {{
      color: {COLOR_LR};
      border-bottom-color: {COLOR_LR};
    }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}

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

    /* ── Ticker Lookup tab ── */
    .lookup-box {{
      display: flex;
      gap: 12px;
      align-items: center;
      margin-bottom: 32px;
    }}
    .lookup-input {{
      background: {COLOR_PANEL};
      border: 1px solid {COLOR_GRID};
      border-radius: 6px;
      color: {COLOR_TEXT};
      font-family: inherit;
      font-size: 1rem;
      font-weight: 700;
      padding: 10px 16px;
      width: 160px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      outline: none;
      transition: border-color 0.15s;
    }}
    .lookup-input:focus {{ border-color: {COLOR_LR}; }}
    .lookup-btn {{
      background: {COLOR_LR};
      border: none;
      border-radius: 6px;
      color: #0D1117;
      cursor: pointer;
      font-family: inherit;
      font-size: 0.82rem;
      font-weight: 700;
      letter-spacing: 0.06em;
      padding: 10px 22px;
      transition: opacity 0.15s;
    }}
    .lookup-btn:hover {{ opacity: 0.85; }}
    .lookup-btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}

    .lookup-spinner {{
      color: {COLOR_SUBTEXT};
      font-size: 0.8rem;
      display: none;
    }}
    .lookup-error {{
      color: #F85149;
      font-size: 0.82rem;
      display: none;
      margin-top: 8px;
    }}

    /* ── Result card ── */
    .result-card {{
      display: none;
      background: {COLOR_PANEL};
      border: 1px solid {COLOR_GRID};
      border-radius: 10px;
      padding: 28px 32px;
      max-width: 900px;
    }}
    .result-header {{
      display: flex;
      align-items: flex-start;
      gap: 24px;
      margin-bottom: 24px;
      flex-wrap: wrap;
    }}
    .result-ticker {{
      font-size: 2rem;
      font-weight: 800;
      color: {COLOR_TEXT};
      letter-spacing: 0.04em;
      line-height: 1;
    }}
    .result-name {{
      font-size: 0.8rem;
      color: {COLOR_SUBTEXT};
      margin-top: 4px;
    }}
    .result-price {{
      font-size: 1.6rem;
      font-weight: 700;
      color: {COLOR_TEXT};
    }}
    .result-change {{
      font-size: 0.85rem;
      margin-top: 4px;
    }}
    .rating-pill {{
      display: inline-block;
      border-radius: 8px;
      padding: 8px 20px;
      font-size: 1rem;
      font-weight: 800;
      letter-spacing: 0.05em;
      margin-top: 4px;
    }}
    .sentiment-bar-wrap {{
      margin: 20px 0 28px;
    }}
    .sentiment-bar-label {{
      display: flex;
      justify-content: space-between;
      font-size: 0.74rem;
      color: {COLOR_SUBTEXT};
      margin-bottom: 6px;
    }}
    .sentiment-bar-track {{
      background: {COLOR_GRID};
      border-radius: 6px;
      height: 10px;
      width: 100%;
      position: relative;
    }}
    .sentiment-bar-fill {{
      border-radius: 6px;
      height: 10px;
      transition: width 0.6s ease;
    }}
    .sentiment-score-num {{
      font-size: 2.2rem;
      font-weight: 800;
      line-height: 1;
      margin-top: 6px;
    }}
    .sentiment-score-label {{
      font-size: 0.8rem;
      color: {COLOR_SUBTEXT};
      margin-top: 2px;
    }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 14px;
      margin-bottom: 28px;
    }}
    .metric-card {{
      background: {COLOR_BG};
      border: 1px solid {COLOR_GRID};
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .metric-card .label {{
      font-size: 0.68rem;
      color: {COLOR_SUBTEXT};
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }}
    .metric-card .value {{
      font-size: 1.1rem;
      font-weight: 700;
      color: {COLOR_TEXT};
    }}
    .explanation-section {{
      border-top: 1px solid {COLOR_GRID};
      padding-top: 20px;
    }}
    .explanation-section h3 {{
      font-size: 0.72rem;
      font-weight: 600;
      color: {COLOR_SUBTEXT};
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-bottom: 12px;
    }}
    .explanation-item {{
      display: flex;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 12px;
      font-size: 0.82rem;
      line-height: 1.6;
    }}
    .explanation-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      margin-top: 6px;
      flex-shrink: 0;
    }}
    .rating-breakdown-row {{
      display: flex;
      gap: 8px;
      align-items: center;
      margin-bottom: 10px;
      flex-wrap: wrap;
    }}
    .rb-label {{
      font-size: 0.75rem;
      width: 90px;
      color: {COLOR_SUBTEXT};
      text-align: right;
    }}
    .rb-bar-track {{
      flex: 1;
      background: {COLOR_GRID};
      border-radius: 4px;
      height: 8px;
      min-width: 80px;
    }}
    .rb-bar-fill {{
      border-radius: 4px;
      height: 8px;
    }}
    .rb-count {{
      font-size: 0.75rem;
      color: {COLOR_SUBTEXT};
      width: 30px;
    }}

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

  <!-- Tab bar -->
  <div class="tab-bar">
    <button class="tab-btn active"     id="btn-overview" onclick="switchTab('overview')">Overview</button>
    <button class="tab-btn lookup-tab" id="btn-lookup"   onclick="switchTab('lookup')">Ticker Lookup</button>
  </div>

  <!-- Tab 1: Overview (existing content) -->
  <div id="tab-overview" class="tab-panel active">
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
  </div>

  <!-- Tab 2: Ticker Lookup -->
  <div id="tab-lookup" class="tab-panel">
    <h2>Ticker Lookup <span class="badge badge-ml">Live Rating & Analysis</span></h2>

    <div class="lookup-box">
      <input id="ticker-input" class="lookup-input" type="text"
             placeholder="e.g. AAPL" maxlength="10"
             onkeydown="if(event.key==='Enter') doLookup()" />
      <button class="lookup-btn" id="lookup-btn" onclick="doLookup()">Analyze</button>
      <span class="lookup-spinner" id="lookup-spinner">Fetching data...</span>
    </div>
    <div class="lookup-error" id="lookup-error"></div>
    <div class="result-card" id="result-card"></div>
  </div>

  <footer>
    Data: Yahoo Finance (yfinance) &nbsp;|&nbsp;
    ML Sentiment = Rating quality (40%) + Price target upside (40%) + Buy ratio (20%)<br>
    Firm accuracy evaluated over 63-day forward window &nbsp;|&nbsp;
    Last updated: {today}
  </footer>

  <script>
    /* ── Tab switching ── */
    function switchTab(id) {{
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.getElementById('tab-' + id).classList.add('active');
      document.getElementById('btn-' + id).classList.add('active');
    }}

    /* ── Sort tables ── */
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

    /* ── Ticker Lookup ── */
    const PROXY = 'https://query2.finance.yahoo.com/v10/finance/quoteSummary/';
    const MODULES = 'financialData,summaryDetail,quoteType,price,recommendationTrend';

    async function doLookup() {{
      const raw    = document.getElementById('ticker-input').value.trim().toUpperCase();
      const errEl  = document.getElementById('lookup-error');
      const spin   = document.getElementById('lookup-spinner');
      const card   = document.getElementById('result-card');
      const btn    = document.getElementById('lookup-btn');

      errEl.style.display = 'none';
      card.style.display  = 'none';
      if (!raw) return;

      spin.style.display = 'inline';
      btn.disabled = true;

      try {{
        const url  = `${{PROXY}}${{raw}}?modules=${{MODULES}}&corsDomain=finance.yahoo.com`;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${{resp.status}}`);
        const json = await resp.json();
        const res  = json?.quoteSummary?.result?.[0];
        if (!res) throw new Error('No data returned. Check the ticker symbol.');
        renderResult(raw, res);
      }} catch(e) {{
        errEl.textContent = `Could not fetch data for "${{raw}}": ${{e.message}}`;
        errEl.style.display = 'block';
      }} finally {{
        spin.style.display = 'none';
        btn.disabled = false;
      }}
    }}

    function renderResult(ticker, res) {{
      const fin   = res.financialData  || {{}};
      const price = res.price          || {{}};
      const qt    = res.quoteType      || {{}};
      const trend = res.recommendationTrend?.trend || [];

      /* ── Raw values ── */
      const companyName   = qt.longName || qt.shortName || ticker;
      const currentPrice  = price.regularMarketPrice?.raw  ?? fin.currentPrice?.raw  ?? null;
      const change        = price.regularMarketChange?.raw  ?? null;
      const changePct     = price.regularMarketChangePercent?.raw ?? null;
      const targetMean    = fin.targetMeanPrice?.raw   ?? null;
      const targetHigh    = fin.targetHighPrice?.raw   ?? null;
      const targetLow     = fin.targetLowPrice?.raw    ?? null;
      const targetMedian  = fin.targetMedianPrice?.raw ?? null;
      const recKey        = fin.recommendationKey      ?? 'n/a';
      const recScore      = fin.recommendationMean?.raw ?? null;
      const nAnalysts     = fin.numberOfAnalystOpinions?.raw ?? 0;

      /* ── Rating breakdown from trend[0] (current month) ── */
      const curr = trend.find(t => t.period === '0m') || {{}};
      const sb   = curr.strongBuy   || 0;
      const b    = curr.buy         || 0;
      const h    = curr.hold        || 0;
      const s    = curr.sell        || 0;
      const ss   = curr.strongSell  || 0;
      const tot  = sb + b + h + s + ss || 1;

      /* ── ML Sentiment Score (mirrors Python logic) ── */
      const upside   = (currentPrice && targetMean) ? (targetMean - currentPrice) / currentPrice * 100 : null;
      const c1 = recScore != null ? (5 - Math.min(5, Math.max(1, recScore))) / 4 * 40 : 0;
      const uClipped = upside != null ? Math.min(100, Math.max(-50, upside)) : 0;
      const c2 = (uClipped + 50) / 150 * 40;
      const c3 = (sb + b) / tot * 20;
      const score = Math.min(100, Math.max(0, c1 + c2 + c3));

      /* ── Derived labels ── */
      const scoreColor = score >= 70 ? '#3FB950' : score >= 40 ? '#F28C28' : '#F85149';
      const scoreLabel = score >= 80 ? 'Strong Bullish'
                       : score >= 60 ? 'Bullish'
                       : score >= 40 ? 'Neutral'
                       : score >= 20 ? 'Bearish' : 'Strong Bearish';

      const RATING_COLORS = {{
        'strong_buy': '#3FB950', 'buy': '#79C0FF',
        'hold': '#F28C28', 'sell': '#FF7B72', 'strong_sell': '#F85149'
      }};
      const ratingColor = RATING_COLORS[recKey] || '#8B949E';
      const ratingLabel = recKey.replace('_',' ').replace(/\\b\\w/g, c => c.toUpperCase());

      const upColor  = upside == null ? '#8B949E' : upside >= 0 ? '#3FB950' : '#F85149';
      const chgColor = (change ?? 0) >= 0 ? '#3FB950' : '#F85149';

      /* ── Explanation bullets ── */
      const bullets = generateExplanation({{
        score, scoreLabel, recKey, recScore, nAnalysts,
        upside, targetMean, targetHigh, targetLow,
        sb, b, h, s, ss, tot,
        c1, c2, c3,
      }});

      /* ── Rating breakdown bars ── */
      const breakdown = [
        ['Strong Buy',  sb,  '#3FB950'],
        ['Buy',         b,   '#79C0FF'],
        ['Hold',        h,   '#F28C28'],
        ['Sell',        s,   '#FF7B72'],
        ['Strong Sell', ss,  '#F85149'],
      ].map(([lbl, count, col]) => `
        <div class="rating-breakdown-row">
          <span class="rb-label">${{lbl}}</span>
          <div class="rb-bar-track">
            <div class="rb-bar-fill" style="width:${{(count/tot*100).toFixed(0)}}%;background:${{col}}"></div>
          </div>
          <span class="rb-count">${{count}}</span>
        </div>`).join('');

      const fmt  = (v, d=2, pre='$') => v != null ? `${{pre}}${{v.toFixed(d)}}` : '—';
      const fmtp = (v) => v != null ? `${{v>=0?'+':''}}${{v.toFixed(1)}}%` : '—';

      document.getElementById('result-card').innerHTML = `
        <div class="result-header">
          <div>
            <div class="result-ticker">${{ticker}}</div>
            <div class="result-name">${{companyName}}</div>
            <div style="margin-top:10px">
              <span class="rating-pill" style="background:${{ratingColor}}22;color:${{ratingColor}};border:1px solid ${{ratingColor}}44">
                ${{ratingLabel}}
              </span>
            </div>
          </div>
          <div>
            <div class="result-price">${{fmt(currentPrice)}}</div>
            <div class="result-change" style="color:${{chgColor}}">
              ${{fmt(change, 2, change>=0?'+$':'$')}} (${{fmtp(changePct ? changePct*100 : null)}}) today
            </div>
          </div>
          <div>
            <div class="sentiment-score-num" style="color:${{scoreColor}}">${{score.toFixed(0)}}<span style="font-size:1rem;font-weight:400;color:#8B949E">/100</span></div>
            <div class="sentiment-score-label">ML Sentiment Score</div>
            <div style="font-size:0.78rem;color:${{scoreColor}};margin-top:2px;font-weight:600">${{scoreLabel}}</div>
          </div>
        </div>

        <div class="sentiment-bar-wrap">
          <div class="sentiment-bar-label">
            <span>Strong Bearish</span><span>Neutral</span><span>Strong Bullish</span>
          </div>
          <div class="sentiment-bar-track">
            <div class="sentiment-bar-fill" style="width:${{score.toFixed(0)}}%;background:${{scoreColor}}"></div>
          </div>
        </div>

        <div class="metrics-grid">
          <div class="metric-card">
            <div class="label">Current Price</div>
            <div class="value">${{fmt(currentPrice)}}</div>
          </div>
          <div class="metric-card">
            <div class="label">Target Mean</div>
            <div class="value" style="color:${{upColor}}">${{fmt(targetMean)}}</div>
          </div>
          <div class="metric-card">
            <div class="label">Upside to Mean</div>
            <div class="value" style="color:${{upColor}}">${{fmtp(upside)}}</div>
          </div>
          <div class="metric-card">
            <div class="label">Target High</div>
            <div class="value">${{fmt(targetHigh)}}</div>
          </div>
          <div class="metric-card">
            <div class="label">Target Low</div>
            <div class="value">${{fmt(targetLow)}}</div>
          </div>
          <div class="metric-card">
            <div class="label">Consensus Score</div>
            <div class="value">${{recScore != null ? recScore.toFixed(2) : '—'}} <span style="font-size:0.7rem;color:#8B949E">(1=SB, 5=SS)</span></div>
          </div>
          <div class="metric-card">
            <div class="label">Analysts</div>
            <div class="value">${{nAnalysts}}</div>
          </div>
          <div class="metric-card">
            <div class="label">Buy Ratio</div>
            <div class="value" style="color:#3FB950">${{((sb+b)/tot*100).toFixed(0)}}%</div>
          </div>
        </div>

        <div style="margin-bottom:28px">
          <div style="font-size:0.72rem;color:#8B949E;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:12px">Rating Breakdown</div>
          ${{breakdown}}
        </div>

        <div class="explanation-section">
          <h3>Analysis & Explanation</h3>
          ${{bullets}}
        </div>`;

      document.getElementById('result-card').style.display = 'block';
    }}

    function generateExplanation(d) {{
      const items = [];

      /* Overall rating */
      const ratingMap = {{
        'strong_buy':  ['#3FB950', 'Strong Buy — Wall Street has high conviction this stock will outperform.'],
        'buy':         ['#79C0FF', 'Buy — analysts expect above-average returns vs the market.'],
        'hold':        ['#F28C28', 'Hold — analysts see limited upside or uncertainty; neither a clear buy nor sell.'],
        'sell':        ['#FF7B72', 'Sell — analysts expect the stock to underperform; consider reducing exposure.'],
        'strong_sell': ['#F85149', 'Strong Sell — analysts have strong negative conviction; significant downside risk flagged.'],
      }};
      const [rc, rt] = ratingMap[d.recKey] || ['#8B949E', 'No consensus available.'];
      items.push([rc, rt]);

      /* Score breakdown */
      items.push(['#A371F7',
        `ML Sentiment Score of <strong>${{d.score.toFixed(0)}}/100</strong> built from three signals: ` +
        `rating quality (${{d.c1.toFixed(1)}} pts), ` +
        `price target upside (${{d.c2.toFixed(1)}} pts), ` +
        `and buy ratio (${{d.c3.toFixed(1)}} pts).`
      ]);

      /* Upside */
      if (d.upside != null) {{
        const dir   = d.upside >= 0 ? 'upside' : 'downside';
        const color = d.upside >= 0 ? '#3FB950' : '#F85149';
        items.push([color,
          `The average analyst price target of <strong>$${{(d.targetMean||0).toFixed(2)}}</strong> implies ` +
          `<strong>${{Math.abs(d.upside).toFixed(1)}}% ${{dir}}</strong> from the current price. ` +
          (d.upside >= 20 ? 'This is a large implied upside — analysts see significant room to run.' :
           d.upside >= 0  ? 'Modest upside; the stock may be approaching fair value in the analyst consensus.' :
                            'Analysts\' mean target is below the current price — the stock may be overvalued relative to consensus.')
        ]);
      }}

      /* Target range */
      if (d.targetHigh && d.targetLow) {{
        const spread = d.targetHigh - d.targetLow;
        const spreadPct = d.targetMean ? (spread / d.targetMean * 100) : 0;
        items.push(['#8B949E',
          `Analyst price target range: <strong>$${{(d.targetLow||0).toFixed(2)}} — $${{(d.targetHigh||0).toFixed(2)}}</strong> ` +
          `(spread: ${{spreadPct.toFixed(0)}}% of mean target). ` +
          (spreadPct > 50 ? 'The wide spread indicates significant disagreement among analysts about the stock\'s fair value.' :
           spreadPct > 20 ? 'Moderate spread — some divergence in analyst views.' :
                            'Tight spread — analysts are broadly aligned on valuation.')
        ]);
      }}

      /* Buy ratio */
      const buyPct = (d.sb + d.b) / d.tot * 100;
      const holdPct = d.h / d.tot * 100;
      items.push([buyPct >= 70 ? '#3FB950' : buyPct >= 50 ? '#F28C28' : '#F85149',
        `<strong>${{buyPct.toFixed(0)}}%</strong> of ${{d.nAnalysts}} analysts rate this a Buy or Strong Buy ` +
        `(${{d.sb}} Strong Buy, ${{d.b}} Buy, ${{d.h}} Hold, ${{d.s+d.ss}} Sell). ` +
        (buyPct >= 80 ? 'Near-unanimous bullish consensus.' :
         buyPct >= 60 ? 'Clear majority bullish, though some caution remains.' :
         buyPct >= 40 ? 'Divided — significant Hold and Sell ratings temper the bullish case.' :
                        'Bearish majority — most analysts do not recommend buying at current levels.')
      ]);

      /* Analyst count */
      items.push(['#8B949E',
        `Coverage by <strong>${{d.nAnalysts}} analysts</strong>. ` +
        (d.nAnalysts >= 30 ? 'Heavily covered — this is a major stock with broad institutional focus.' :
         d.nAnalysts >= 10 ? 'Well covered — sufficient analyst diversity for reliable consensus.' :
         d.nAnalysts >= 3  ? 'Lightly covered — treat consensus with more caution; a single analyst change can move the rating significantly.' :
                             'Very few analysts — consensus may not be representative.')
      ]);

      return items.map(([color, text]) => `
        <div class="explanation-item">
          <div class="explanation-dot" style="background:${{color}}"></div>
          <div style="color:#E6EDF3">${{text}}</div>
        </div>`).join('');
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
