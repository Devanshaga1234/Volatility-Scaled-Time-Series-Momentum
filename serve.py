"""
Local development server for all quant dashboards.

Routes:
    /                     -> landing page (links to both dashboards)
    /analyst              -> analyst_dashboard.html
    /tsmom                -> index.html  (TSMOM + ML dashboard)
    /api/lookup?ticker=X  -> yfinance proxy (used by Ticker Lookup tab)

Usage:
    python serve.py
Then open:  http://localhost:8050
"""

import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

import yfinance as yf
from portfolio_builder import get_portfolio_summary

PORT = 8050
BASE  = os.path.dirname(os.path.abspath(__file__))
ANALYST_FILE = os.path.join(BASE, "analyst_dashboard.html")
TSMOM_FILE   = os.path.join(BASE, "index.html")

LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Quant Dashboards</title>
  <style>
    body { background:#0D1117; color:#E6EDF3; font-family:'SF Mono',monospace;
           display:flex; flex-direction:column; align-items:center;
           justify-content:center; min-height:100vh; margin:0; gap:32px; }
    h1   { font-size:1.1rem; letter-spacing:0.1em; color:#8B949E; text-transform:uppercase; }
    .cards { display:flex; gap:24px; flex-wrap:wrap; justify-content:center; }
    .card  { background:#161B22; border:1px solid #21262D; border-radius:10px;
             padding:28px 36px; text-align:center; text-decoration:none;
             color:#E6EDF3; transition:border-color 0.15s; min-width:200px; }
    .card:hover { border-color:#58A6FF; }
    .card .title { font-size:1rem; font-weight:700; margin-bottom:6px; }
    .card .sub   { font-size:0.75rem; color:#8B949E; }
    .dot-blue   { color:#58A6FF; }
    .dot-purple { color:#A371F7; }
  </style>
</head>
<body>
  <h1>Quant Dashboards &nbsp;<span style="color:#21262D">|</span>&nbsp; localhost:8050</h1>
  <div class="cards">
    <a class="card" href="/tsmom">
      <div class="title"><span class="dot-blue">&#9632;</span> TSMOM + ML</div>
      <div class="sub">Strategy backtests &amp; ML signal comparison</div>
    </a>
    <a class="card" href="/analyst">
      <div class="title"><span class="dot-purple">&#9632;</span> Analyst Ratings</div>
      <div class="sub">Wall Street consensus &amp; ticker lookup</div>
    </a>
  </div>
</body>
</html>"""


def _ticker_data(symbol: str) -> dict:
    t = yf.Ticker(symbol)
    info = t.info

    # Recommendation trend
    try:
        recs = t.recommendations
        if recs is not None and not recs.empty:
            latest = recs.iloc[-1]
            trend_entry = {
                "period": "0m",
                "strongBuy":  int(latest.get("strongBuy",  0)),
                "buy":        int(latest.get("buy",        0)),
                "hold":       int(latest.get("hold",       0)),
                "sell":       int(latest.get("sell",       0)),
                "strongSell": int(latest.get("strongSell", 0)),
            }
            trend = [trend_entry]
        else:
            trend = []
    except Exception:
        trend = []

    # Build a response that matches the structure renderResult() expects
    return {
        "financialData": {
            "currentPrice":             {"raw": info.get("currentPrice")},
            "targetMeanPrice":          {"raw": info.get("targetMeanPrice")},
            "targetHighPrice":          {"raw": info.get("targetHighPrice")},
            "targetLowPrice":           {"raw": info.get("targetLowPrice")},
            "targetMedianPrice":        {"raw": info.get("targetMedianPrice")},
            "recommendationKey":        info.get("recommendationKey", "n/a"),
            "recommendationMean":       {"raw": info.get("recommendationMean")},
            "numberOfAnalystOpinions":  {"raw": info.get("numberOfAnalystOpinions", 0)},
        },
        "price": {
            "regularMarketPrice":          {"raw": info.get("currentPrice")},
            "regularMarketChange":         {"raw": info.get("regularMarketChange")},
            "regularMarketChangePercent":  {"raw": info.get("regularMarketChangePercent")},
        },
        "quoteType": {
            "longName":  info.get("longName", symbol),
            "shortName": info.get("shortName", symbol),
        },
        "recommendationTrend": {
            "trend": trend,
        },
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} {fmt % args}")

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, filepath: str, missing_hint: str = "File not found."):
        try:
            with open(filepath, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            msg = missing_hint.encode()
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def _send_landing(self):
        body = LANDING_HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path.rstrip("/") or "/"

        if path == "/":
            self._send_landing()

        elif path in ("/analyst", "/analyst_dashboard.html"):
            self._send_file(ANALYST_FILE,
                            "analyst_dashboard.html not found. Run: python analyst_dashboard.py")

        elif path in ("/tsmom", "/index.html"):
            self._send_file(TSMOM_FILE,
                            "index.html not found. Run: python dashboard.py")

        elif path == "/api/portfolio":
            params  = urllib.parse.parse_qs(parsed.query)
            deposit = float((params.get("deposit") or ["30"])[0])
            try:
                data = get_portfolio_summary(deposit=deposit)
                self._send_json(data)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)

        elif path == "/api/lookup":
            params = urllib.parse.parse_qs(parsed.query)
            symbol = (params.get("ticker") or [""])[0].upper().strip()
            if not symbol:
                self._send_json({"error": "ticker parameter required"}, 400)
                return
            try:
                data = _ticker_data(symbol)
                self._send_json({"ok": True, "data": data})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)

        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    server = HTTPServer(("localhost", PORT), Handler)
    print(f"Quant Dashboards running at  http://localhost:{PORT}")
    print(f"  TSMOM + ML  ->  http://localhost:{PORT}/tsmom")
    print(f"  Analyst     ->  http://localhost:{PORT}/analyst")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
