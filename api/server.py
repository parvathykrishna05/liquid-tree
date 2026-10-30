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
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
        elif path == "/api/vision/sample-image":
            sample_id = parse_qs(parsed.query).get("sample_id", ["distinct_108"])[0]
            if not sample_id.replace("_", "").isalnum():
                self._send_json_response({"error": "Invalid sample id."}, status=400)
                return
            image_path = self.service.val_images_dir / f"{sample_id}.png"
            if not image_path.is_file():
                self._send_json_response({"error": "Microscopy sample image is unavailable."}, status=404)
                return
            import base64
            image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
            self._send_json_response({"image_data_uri": f"data:image/png;base64,{image_data}"})
        elif path == "/api/alerts":
            self._send_json_response({"alerts": self.service.get_system_alerts()})
        elif path == "/api/recommendations":
            self._send_json_response(self.service.get_operational_recommendations())
        elif path == "/api/crash-risk":
            status = self.service.get_reactor_overview()
            telemetry = self.service.get_sensor_telemetry()["sensors"]
            risk = self.service.assess_culture_crash_risk(
                current_density=status["current_cell_density_million_ml"],
                current_ph=telemetry["ph"]["value"],
                light=telemetry["incident_ppfd_light"]["value"],
                co2_ppm=telemetry["co2_inflow_concentration"]["value"],
                culture_day=status["culture_day"],
            )
            self._send_json_response(risk)
        elif path == "/api/automation/actuate":
            telemetry = self.service.get_sensor_telemetry()["sensors"]
            status = self.service.get_reactor_overview()
            actuation = self.service.execute_closed_loop_actuation(
                current_ph=telemetry["ph"]["value"],
                current_density=status["current_cell_density_million_ml"],
                current_light=telemetry["incident_ppfd_light"]["value"],
                co2_ppm=telemetry["co2_inflow_concentration"]["value"],
                mode="AUTO_CLOSED_LOOP",
            )
            self._send_json_response(actuation)
        elif path == "/api/mobile/zones":
            dispatch = self.service.simulate_mobile_liquid_tree_dispatch(target_zone="boardroom_a")
            self._send_json_response(dispatch)
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
            culture_day = int(payload.get("culture_day", 8))
            result = self.service.predict_next_day_growth(current_density, current_ph, light, co2_ppm, culture_day)
            self._send_json_response(result)

        elif path == "/api/crash-risk":
            current_density = float(payload.get("current_density", 18.4))
            current_ph = float(payload.get("current_ph", 7.42))
            light = float(payload.get("light", 240.0))
            co2_ppm = float(payload.get("co2_ppm", 412.0))
            culture_day = int(payload.get("culture_day", 8))
            result = self.service.assess_culture_crash_risk(current_density, current_ph, light, co2_ppm, culture_day)
            self._send_json_response(result)

        elif path == "/api/automation/actuate":
            current_ph = float(payload.get("current_ph", 7.42))
            current_density = float(payload.get("current_density", 18.4))
            current_light = float(payload.get("current_light", 240.0))
            co2_ppm = float(payload.get("co2_ppm", 412.0))
            mode = payload.get("mode", "AUTO_CLOSED_LOOP")
            result = self.service.execute_closed_loop_actuation(current_ph, current_density, current_light, co2_ppm, mode)
            self._send_json_response(result)

        elif path == "/api/mobile/dispatch":
            target_zone = payload.get("target_zone", "boardroom_a")
            custom_co2 = payload.get("custom_co2_ppm", None)
            if custom_co2 is not None:
                custom_co2 = float(custom_co2)
            result = self.service.simulate_mobile_liquid_tree_dispatch(target_zone, custom_co2)
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
    httpd = ThreadingHTTPServer(server_address, LiquidTreeHTTPHandler)
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
