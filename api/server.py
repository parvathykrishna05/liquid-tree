"""
Liquid Tree AI - Web Application & API Server
Serves the monitoring dashboard and exposes REST endpoints for telemetry,
predictive modeling, and microscopy cell detection.
Zero extra dependencies: built on Python standard library http.server.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

PROJECT_ROOT = Path("C:/4algae").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from api.service import LiquidTreeService


class LiquidTreeHTTPHandler(SimpleHTTPRequestHandler):
    service = LiquidTreeService(PROJECT_ROOT)

    def __init__(self, *args, **kwargs):
        dashboard_dir = str(PROJECT_ROOT / "dashboard")
        super().__init__(*args, directory=dashboard_dir, **kwargs)

    def _send_json_response(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            self._send_json_response(self.service.get_reactor_overview())
        elif path == "/api/telemetry":
            self._send_json_response(self.service.get_sensor_telemetry())
        elif path == "/api/vision/samples":
            self._send_json_response({"samples": self.service.get_vision_validation_samples()})
        elif path == "/api/alerts":
            self._send_json_response({"alerts": self.service.get_system_alerts()})
        elif path == "/api/recommendations":
            self._send_json_response(self.service.get_operational_recommendations())
        elif path == "/api/history":
            self._send_json_response(self.service.get_historical_trends())
        elif path.startswith("/api/"):
            self._send_json_response({"error": f"Unknown API endpoint: {path}"}, status=404)
        else:
            # Fallback to serving static files from dashboard/
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            payload = json.loads(post_body)
        except json.JSONDecodeError:
            payload = {}

        if path == "/api/predict":
            current_density = float(payload.get("current_density", 18.4))
            current_ph = float(payload.get("current_ph", 7.42))
            light = float(payload.get("light", 240.0))
            co2_ppm = float(payload.get("co2_ppm", 412.0))
            result = self.service.predict_next_day_growth(current_density, current_ph, light, co2_ppm)
            self._send_json_response(result)

        elif path == "/api/vision/detect":
            sample_id = payload.get("sample_id", "distinct_108")
            conf_threshold = float(payload.get("conf_threshold", 0.25))
            result = self.service.run_vision_detection(sample_id, conf_threshold)
            self._send_json_response(result)

        else:
            self._send_json_response({"error": f"Unknown POST endpoint: {path}"}, status=404)


def run_server(port: int = 8050):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, LiquidTreeHTTPHandler)
    print("=" * 70)
    print(f"Liquid Tree AI Server running at http://127.0.0.1:{port}/")
    print(f"Dashboard UI: http://127.0.0.1:{port}/index.html")
    print(f"REST API root: http://127.0.0.1:{port}/api/")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Liquid Tree server.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Liquid Tree AI Server")
    parser.add_argument("--port", type=int, default=8050, help="Server port (default: 8050)")
    args = parser.parse_args()
    run_server(args.port)
