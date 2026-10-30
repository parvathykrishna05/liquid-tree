"""
Liquid Tree AI - Flask API for Railway/cloud deployment.
Wraps LiquidTreeService with Flask so it can run as a proper WSGI server.
"""

from __future__ import annotations

import os
import json
import base64
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── import core service ───────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.service import LiquidTreeService

app = Flask(__name__)
CORS(app)  # Allow all origins (Vercel frontend)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
service = LiquidTreeService(PROJECT_ROOT)


# ── Health Check ──────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "4Algae Liquid Tree API"})


# ── GET endpoints ─────────────────────────────────────────────────────
@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify(service.get_reactor_overview())


@app.route("/api/telemetry", methods=["GET"])
def get_telemetry():
    return jsonify(service.get_sensor_telemetry())


@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    return jsonify({"alerts": service.get_system_alerts()})


@app.route("/api/recommendations", methods=["GET"])
def get_recommendations():
    return jsonify(service.get_operational_recommendations())


@app.route("/api/history", methods=["GET"])
def get_history():
    return jsonify(service.get_historical_trends())


@app.route("/api/vision/samples", methods=["GET"])
def get_vision_samples():
    return jsonify({"samples": service.get_vision_validation_samples()})


@app.route("/api/vision/sample-image", methods=["GET"])
def get_vision_sample_image():
    sample_id = request.args.get("sample_id", "distinct_108")
    if not sample_id.replace("_", "").isalnum():
        return jsonify({"error": "Invalid sample id."}), 400

    image_path = service.val_images_dir / f"{sample_id}.png"
    if not image_path.is_file():
        return jsonify({"error": "Microscopy sample image is unavailable."}), 404

    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return jsonify({"image_data_uri": f"data:image/png;base64,{image_data}"})


@app.route("/api/crash-risk", methods=["GET", "POST"])
def crash_risk():
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
    else:
        data = {}

    overview  = service.get_reactor_overview()
    telemetry = service.get_sensor_telemetry()["sensors"]

    result = service.assess_culture_crash_risk(
        current_density=float(data.get("current_density", overview["current_cell_density_million_ml"])),
        current_ph=float(data.get("current_ph", telemetry["ph"]["value"])),
        light=float(data.get("light", telemetry["incident_ppfd_light"]["value"])),
        co2_ppm=float(data.get("co2_ppm", telemetry["co2_inflow_concentration"]["value"])),
        culture_day=int(data.get("culture_day", overview["culture_day"])),
    )
    return jsonify(result)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True) or {}
    result = service.predict_next_day_growth(
        current_density=float(data.get("current_density", 18.4)),
        current_ph=float(data.get("current_ph", 7.42)),
        light=float(data.get("light", 240.0)),
        co2_ppm=float(data.get("co2_ppm", 412.0)),
        culture_day=int(data.get("culture_day", 8)),
    )
    return jsonify(result)


@app.route("/api/automation/actuate", methods=["GET", "POST"])
def actuate():
    data = request.get_json(force=True, silent=True) or {} if request.method == "POST" else {}
    telemetry = service.get_sensor_telemetry()["sensors"]
    overview  = service.get_reactor_overview()
    result = service.execute_closed_loop_actuation(
        current_ph=float(data.get("current_ph", telemetry["ph"]["value"])),
        current_density=float(data.get("current_density", overview["current_cell_density_million_ml"])),
        current_light=float(data.get("current_light", telemetry["incident_ppfd_light"]["value"])),
        co2_ppm=float(data.get("co2_ppm", telemetry["co2_inflow_concentration"]["value"])),
        mode=data.get("mode", "AUTO_CLOSED_LOOP"),
    )
    return jsonify(result)


@app.route("/api/mobile/zones", methods=["GET"])
def mobile_zones():
    result = service.simulate_mobile_liquid_tree_dispatch(target_zone="boardroom_a")
    return jsonify(result)


@app.route("/api/mobile/dispatch", methods=["POST"])
def mobile_dispatch():
    data = request.get_json(force=True, silent=True) or {}
    target_zone = data.get("target_zone", "boardroom_a")
    custom_co2 = data.get("custom_co2_ppm")
    if custom_co2 is not None:
        custom_co2 = float(custom_co2)
    result = service.simulate_mobile_liquid_tree_dispatch(target_zone, custom_co2)
    return jsonify(result)


@app.route("/api/vision/detect", methods=["POST"])
def vision_detect():
    data = request.get_json(force=True, silent=True) or {}
    result = service.run_vision_detection(
        sample_id=data.get("sample_id", "distinct_108"),
        conf_threshold=float(data.get("conf_threshold", 0.25)),
    )
    return jsonify(result)


# ── Run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
