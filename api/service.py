"""
Liquid Tree AI - Core Backend Service & Data Pipeline
Provides modular business logic for simulated telemetry, experimental growth estimates,
microscopy cell detection (Phase 9C YOLOv8n), diagnostics, and operational recommendations.
"""

from __future__ import annotations

import os
import json
import base64
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path("C:/4algae").resolve()


class LiquidTreeService:
    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = project_root
        self.checkpoint_id = "phase9c_yolo_train_20epochs_best"
        self.checkpoint_path = project_root / "models" / "phase9c_yolo" / "train_20epochs" / "weights" / "best.pt"
        self.val_images_dir = project_root / "data" / "processed" / "cidac" / "phase9c_yolo" / "images" / "val"
        self.val_labels_dir = project_root / "data" / "processed" / "cidac" / "phase9c_yolo" / "labels" / "val"
        self._yolo_model = None

    def _vision_checkpoint_metadata(self) -> Dict[str, str]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_path": self.checkpoint_path.relative_to(self.project_root).as_posix(),
        }

    def get_reactor_overview(self) -> Dict[str, Any]:
        """Returns physical and biological status of the Liquid Tree bioreactor."""
        return {
            "reactor_id": "LT-BELGRADE-01",
            "reactor_type": "Urban Liquid Tree Photobioreactor (Column PBR)",
            "microalgae_strain": "Chlorella vulgaris (UTEX 2714 equivalent)",
            "working_volume_liters": 600.0,
            "operational_status": "OPTIMAL_GROWTH",
            "culture_day": 8,
            "current_cell_density_million_ml": 18.4,
            "estimated_dry_biomass_g_l": 0.92,  # ~0.05 g/L per 10^6 cells/mL
            "daily_co2_sequestered_kg": 1.15,
            "ambient_mode": "URBAN_DAYLIGHT_CYCLE",
            "telemetry_source_disclaimer": "SIMULATED TELEMETRY: Illustrative values; not live hardware readings.",
        }

    def get_sensor_telemetry(self) -> Dict[str, Any]:
        """
        Returns simulated sensor telemetry with explicit source metadata.
        """
        return {
            "source": "SIMULATED",
            "source_details": "Simulated parametric stream; no live sensor or hardware connection",
            "timestamp": "2026-10-09T20:15:00Z",
            "sensors": {
                "ph": {
                    "value": 7.42,
                    "unit": "pH",
                    "status": "NORMAL",
                    "optimal_range": [7.0, 7.8],
                    "critical_thresholds": [6.5, 8.5],
                    "is_simulated": True,
                },
                "temperature": {
                    "value": 23.6,
                    "unit": "°C",
                    "status": "NORMAL",
                    "optimal_range": [20.0, 26.0],
                    "critical_thresholds": [15.0, 32.0],
                    "is_simulated": True,
                },
                "dissolved_oxygen": {
                    "value": 8.7,
                    "unit": "mg/L",
                    "status": "NORMAL",
                    "optimal_range": [6.0, 10.0],
                    "critical_thresholds": [4.0, 14.0],
                    "is_simulated": True,
                },
                "co2_inflow_concentration": {
                    "value": 412.0,
                    "unit": "ppm",
                    "status": "NORMAL",
                    "optimal_range": [380.0, 1200.0],
                    "is_simulated": True,
                },
                "optical_density_od680": {
                    "value": 1.28,
                    "unit": "AU",
                    "status": "NORMAL",
                    "optical_cell_density_equiv_million_ml": 18.2,
                    "is_simulated": True,
                },
                "incident_ppfd_light": {
                    "value": 240.0,
                    "unit": "μmol·m⁻²·s⁻¹",
                    "status": "NORMAL",
                    "photoperiod": "16L:8D",
                    "is_simulated": True,
                },
            },
        }

    def predict_next_day_growth(self, current_density: float, current_ph: float, light: float, co2_ppm: float) -> Dict[str, Any]:
        """
        Computes an experimental formula-based next-day estimate.
        This hand-coded formula is not a trained or validated ML model.
        """
        # Experimental logistic-style estimate; its parameters are not a validated trained model.
        # N_{t+1} = N_t + mu * N_t * (1 - N_t / K)
        # Optimal pH = 7.4; optimal light = 300 umol/m2/s; CO2 saturation ~800 ppm
        ph_factor = np.exp(-0.5 * ((current_ph - 7.4) / 0.8) ** 2)
        light_factor = min(light / 250.0, 1.2) * np.exp(-max(0.0, light - 350.0) / 400.0)
        co2_factor = 0.8 + 0.3 * (min(co2_ppm, 1000.0) / 1000.0)

        # Base specific growth rate mu (day^-1)
        k_carrying_capacity = 32.0  # million cells/mL
        mu_max = 0.38
        mu_effective = mu_max * ph_factor * light_factor * co2_factor

        density_increment = mu_effective * current_density * max(0.0, (1.0 - current_density / k_carrying_capacity))
        predicted_next_day_density = round(float(current_density + density_increment), 2)
        doubling_time_days = round(float(np.log(2.0) / max(mu_effective, 0.01)), 2)

        return {
            "model_type": "Experimental formula-based growth estimate (not a trained model)",
            "inputs": {
                "current_cell_density": current_density,
                "current_ph": current_ph,
                "light_ppfd": light,
                "co2_ppm": co2_ppm,
            },
            "predictions": {
                "predicted_next_day_density_million_ml": predicted_next_day_density,
                "predicted_density_delta": round(predicted_next_day_density - current_density, 2),
                "predicted_specific_growth_rate_mu_day": round(float(mu_effective), 3),
                "doubling_time_days": doubling_time_days,
                "culture_phase": "EXPONENTIAL" if current_density < 20.0 else "DECELERATION/STATIONARY",
            },
            "disclaimer": "Experimental estimate from a hand-coded formula; not validated as a predictive model.",
        }

    def get_vision_validation_samples(self) -> List[Dict[str, Any]]:
        """Returns available validation microscopy samples for interactive inspection."""
        samples = [
            {"id": "distinct_108", "filename": "distinct_108.png", "subset": "distinct", "label": "Distinct Cells (Medium Density)"},
            {"id": "distinct_110", "filename": "distinct_110.png", "subset": "distinct", "label": "Distinct Cells (Low Density)"},
            {"id": "distinct_114", "filename": "distinct_114.png", "subset": "distinct", "label": "Distinct Cells (Dispersed)"},
            {"id": "distinct_130", "filename": "distinct_130.png", "subset": "distinct", "label": "Distinct Cells (Single Cells)"},
            {"id": "clusters_107", "filename": "clusters_107.png", "subset": "clusters", "label": "Cell Clusters & Aggregates"},
            {"id": "clusters_119", "filename": "clusters_119.png", "subset": "clusters", "label": "High-Density Cluster Sample"},
            {"id": "clusters_200", "filename": "clusters_200.png", "subset": "clusters", "label": "Cell Aggregation with Debris"},
        ]
        return samples

    def run_vision_detection(self, sample_id: str = "distinct_108", conf_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Executes YOLOv8n cell detection on the requested validation image.
        Uses Phase 9C best.pt checkpoint.
        Strictly excludes Class 1 from cell counts.
        """
        img_name = f"{sample_id}.png"
        img_path = self.val_images_dir / img_name
        lbl_path = self.val_labels_dir / f"{sample_id}.txt"

        if not img_path.exists():
            return {
                "status": "UNAVAILABLE",
                "error": f"Sample image not found: {img_name}",
                **self._vision_checkpoint_metadata(),
            }

        # Read ground truth labels for comparison
        gt_class0 = 0
        gt_class1 = 0
        gt_boxes = []
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:])
                        gt_boxes.append({"class_id": cls_id, "box": [xc, yc, w, h]})
                        if cls_id == 0:
                            gt_class0 += 1
                        else:
                            gt_class1 += 1

        try:
            if not self.checkpoint_path.exists() and self._yolo_model is None:
                return {
                    "status": "UNAVAILABLE",
                    "error": "YOLO weights are unavailable; no predictions were produced.",
                    **self._vision_checkpoint_metadata(),
                }

            if self._yolo_model is None:
                from ultralytics import YOLO
                self._yolo_model = YOLO(str(self.checkpoint_path))

            results = self._yolo_model.predict(
                source=str(img_path),
                conf=conf_threshold,
                imgsz=640,
                device="cpu",
                verbose=False,
            )[0]

            detections = []
            pred_class0 = 0
            pred_class1 = 0
            for b in results.boxes:
                cls_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                xc = ((x1 + x2) / 2.0) / 640.0
                yc = ((y1 + y2) / 2.0) / 640.0
                w = (x2 - x1) / 640.0
                h = (y2 - y1) / 640.0

                detections.append({
                    "class_id": cls_id,
                    "class_name": "chlorella_cell" if cls_id == 0 else "cluster_noise",
                    "confidence": round(conf, 3),
                    "box": [round(xc, 4), round(yc, 4), round(w, 4), round(h, 4)],
                })
                if cls_id == 0:
                    pred_class0 += 1
                else:
                    pred_class1 += 1

            # Read only after successful inference so image I/O failures are not reported as success.
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"Inference exception: {e}")
            return {
                "status": "ERROR",
                "error": f"Microscopy inference unavailable: {e}",
                **self._vision_checkpoint_metadata(),
            }

        return {
            "status": "SUCCESS",
            **self._vision_checkpoint_metadata(),
            "proof_of_concept_notice": "MICROSCOPY PROOF OF CONCEPT: YOLOv8n detector trained on Phase 9B CIDACC dataset. Class 1 clusters/debris strictly excluded from cell count.",
            "sample_id": sample_id,
            "image_data_uri": f"data:image/png;base64,{img_b64}",
            "conf_threshold": conf_threshold,
            "counts": {
                "predicted_individual_cells_class0": pred_class0,
                "predicted_clusters_debris_class1": pred_class1,
                "ground_truth_cells_class0": gt_class0,
                "ground_truth_clusters_class1": gt_class1,
                "counting_error": pred_class0 - gt_class0,
                "vision_derived_density_million_ml": None,
                "density_message": "Calibration unavailable: cell counts have not been experimentally calibrated to culture density.",
            },
            "detections": detections,
        }

    def get_system_alerts(self, sensor_density: float | None = None, vision_density: float | None = None) -> List[Dict[str, Any]]:
        """Returns simulated operating alerts and compares counts only when both are supplied."""
        alerts = []

        # Check sensor-vision agreement
        if sensor_density is not None and vision_density is not None and sensor_density > 0:
            discrepancy_pct = abs(sensor_density - vision_density) / sensor_density * 100.0
            if discrepancy_pct > 25.0:
                alerts.append({
                    "severity": "WARNING",
                    "category": "SENSOR_VISION_DISAGREEMENT",
                    "message": f"Optical density sensor ({sensor_density:.1f}M cells/mL) diverges from microscopy cell count ({vision_density:.1f}M cells/mL) by {discrepancy_pct:.1f}%. Possible wall biofouling or cellular clumping.",
                    "action_required": "Inspect bioreactor optical path and perform manual hemocytometer verification."
                })
            else:
                alerts.append({
                    "severity": "INFO",
                    "category": "SENSOR_VISION_AGREEMENT",
                    "message": f"Sensor OD ({sensor_density:.1f}M) and Microscopy Vision ({vision_density:.1f}M) are in close agreement (delta: {discrepancy_pct:.1f}%).",
                    "action_required": "None."
                })

        alerts.append({
            "severity": "NORMAL",
            "category": "PH_HOMEOSTASIS",
            "message": "Simulated pH value 7.42 is within the demonstration range (7.0 - 7.8). No physical valve is connected or controlled.",
            "action_required": "None."
        })
        alerts.append({
            "severity": "NORMAL",
            "category": "TEMPERATURE_CONTROL",
            "message": "Simulated temperature value 23.6°C is within the demonstration range. No physical heat exchanger is connected or controlled.",
            "action_required": "None."
        })

        return alerts

    def get_operational_recommendations(self, current_density: float = 18.4) -> Dict[str, Any]:
        """Provides operational harvest and sparging recommendations (clearly tagged experimental)."""
        harvest_threshold = 24.0  # million cells/mL
        near_harvest = current_density >= 18.0

        return {
            "status": "EXPERIMENTAL_OPTIMIZATION_RECOMMENDATION",
            "disclaimer": "EXPERIMENTAL PROTOCOL: Operating recommendations are algorithmic heuristics and require field agronomic validation before autonomous execution.",
            "harvest_decision": {
                "recommend_harvest": bool(current_density >= harvest_threshold),
                "readiness_percentage": round(min(current_density / harvest_threshold * 100.0, 100.0), 1),
                "harvest_type": "PARTIAL_VOLUME_EXCHANGE" if near_harvest else "ACCUMULATION",
                "recommended_exchange_volume_liters": 180.0 if near_harvest else 0.0,
                "rationale": (
                    f"Current density ({current_density}M cells/mL) is {round(current_density / harvest_threshold * 100.0, 1)}% of maximum carrying capacity. "
                    "Performing a 30% dilution harvest (180 L) resets culture to the steep exponential growth phase, maximizing weekly CO2 capture."
                ) if near_harvest else "Culture is in active accumulation phase; maintain growth.",
            },
            "sparging_recommendation": {
                "co2_enrichment_ratio": "0.04% - 0.08% v/v pulse",
                "duty_cycle": "15 min ON / 45 min OFF",
                "target_ph": 7.4,
            },
            "lighting_recommendation": {
                "photoperiod_schedule": "16 hours illumination (06:00 - 22:00) / 8 hours dark respiration",
                "night_mode": "INACTIVE",
            }
        }

    def get_historical_trends(self) -> Dict[str, Any]:
        """Returns hard-coded illustrative replay data, not measured reactor history."""
        days = list(range(1, 15))
        # Fixed demonstration arrays for dashboard rendering; not experimental observations.
        density_curve = [1.2, 1.8, 2.7, 4.1, 6.2, 9.4, 13.8, 18.4, 21.8, 24.5, 26.2, 27.1, 27.8, 28.2]
        ph_curve = [7.25, 7.30, 7.35, 7.38, 7.40, 7.42, 7.41, 7.42, 7.45, 7.48, 7.50, 7.52, 7.51, 7.53]
        temp_curve = [22.1, 22.8, 23.4, 23.1, 23.9, 24.2, 23.8, 23.6, 24.1, 23.5, 23.7, 24.0, 23.8, 23.9]
        co2_capture_kg = [0.15, 0.22, 0.34, 0.51, 0.78, 1.05, 1.18, 1.15, 1.08, 0.95, 0.82, 0.71, 0.65, 0.58]

        return {
            "source": "SIMULATED_ILLUSTRATIVE_REPLAY",
            "reference": "Hard-coded illustrative 14-day trajectory; not measured reactor data.",
            "days": days,
            "metrics": {
                "cell_density_million_ml": density_curve,
                "ph": ph_curve,
                "temperature_c": temp_curve,
                "daily_co2_kg": co2_capture_kg,
            }
        }
