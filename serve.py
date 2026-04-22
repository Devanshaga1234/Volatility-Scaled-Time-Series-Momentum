"""
Local development server for the Analyst Dashboard.

Serves analyst_dashboard.html and proxies ticker lookups through yfinance
so the browser never hits Yahoo Finance directly (no CORS issues).

Usage:
    python serve.py
Then open:  http://localhost:8050
"""

import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

import yfinance as yf

PORT = 8050
HTML_FILE = os.path.join(os.path.dirname(__file__), "analyst_dashboard.html")


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

    def _send_html(self):
        try:
            with open(HTML_FILE, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"analyst_dashboard.html not found. Run: python analyst_dashboard.py")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path.rstrip("/") or "/"

        if path in ("/", "/analyst_dashboard.html"):
            self._send_html()

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
    print(f"Analyst Dashboard running at  http://localhost:{PORT}")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
