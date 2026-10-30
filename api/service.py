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

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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

    def assess_culture_crash_risk(
        self,
        current_density: float,
        current_ph: float,
        light: float,
        co2_ppm: float,
        culture_day: int = 8,
    ) -> Dict[str, Any]:
        """
        Calculates an early-warning Culture Crash Risk Index (0 - 100%) based on
        Phase 7 empirical biological dynamics (23.7% negative growth transitions)
        and environmental stress factors (self-shading, pH drift, carbon starvation).
        """
        stress_breakdown = {}

        # 1. Density Saturation & Self-Shading Stress (Beer-Lambert light attenuation)
        if current_density > 22.0:
            shading_penalty = min(40.0, ((current_density - 22.0) / 10.0) * 40.0)
            shading_status = "CRITICAL_MUTUAL_SHADING" if current_density > 26.0 else "ELEVATED_SHADING"
        else:
            shading_penalty = max(0.0, (current_density / 22.0) * 8.0)
            shading_status = "ADEQUATE_LIGHT_PENETRATION"
        stress_breakdown["self_shading_stress"] = {
            "score": round(float(shading_penalty), 1),
            "status": shading_status,
            "description": "Inner reactor light attenuation due to dense cell packing.",
        }

        # 2. pH Homeostasis Stress (Optimal: 7.0 - 7.6)
        if current_ph < 6.8:
            ph_penalty = min(30.0, ((6.8 - current_ph) / 1.0) * 30.0)
            ph_status = "ACID_STRESS"
        elif current_ph > 7.8:
            ph_penalty = min(35.0, ((current_ph - 7.8) / 1.0) * 35.0)
            ph_status = "ALKALINE_DRIFT_NUTRIENT_PRECIPITATION"
        else:
            ph_penalty = 0.0
            ph_status = "OPTIMAL_PH_WINDOW"
        stress_breakdown["ph_stress"] = {
            "score": round(float(ph_penalty), 1),
            "status": ph_status,
            "description": "Deviation from physiological optimal pH (7.2 - 7.6).",
        }

        # 3. Carbon Inflow Sufficiency vs Biomass Demand
        required_co2 = 380.0 + (current_density * 20.0)
        if co2_ppm < required_co2:
            co2_deficit = (required_co2 - co2_ppm) / max(required_co2, 1.0)
            co2_penalty = min(30.0, co2_deficit * 30.0)
            co2_status = "CARBON_LIMITATION"
        else:
            co2_penalty = 0.0
            co2_status = "CARBON_SATURATED"
        stress_breakdown["carbon_starvation_stress"] = {
            "score": round(float(co2_penalty), 1),
            "status": co2_status,
            "description": f"CO2 supply ({co2_ppm:.0f} ppm) vs biomass demand ({required_co2:.0f} ppm equivalent).",
        }

        # 4. Light Irradiance Extremes
        if light < 100.0:
            light_penalty = min(20.0, ((100.0 - light) / 80.0) * 20.0)
            light_status = "LIGHT_STARVATION"
        elif light > 450.0:
            light_penalty = min(20.0, ((light - 450.0) / 150.0) * 20.0)
            light_status = "PHOTO_INHIBITION"
        else:
            light_penalty = 0.0
            light_status = "PHOTOSYNTHETICALLY_ACTIVE"
        stress_breakdown["irradiance_stress"] = {
            "score": round(float(light_penalty), 1),
            "status": light_status,
            "description": "Photon flux density relative to saturation kinetics.",
        }

        # 5. Culture Age / Senescence
        if culture_day > 12:
            age_penalty = min(15.0, (culture_day - 12) * 3.0)
            age_status = "SENESCENCE_RISK"
        else:
            age_penalty = 0.0
            age_status = "VIGOROUS_BATCH"
        stress_breakdown["culture_age_stress"] = {
            "score": round(float(age_penalty), 1),
            "status": age_status,
            "description": f"Batch age (Day {culture_day}) relative to stationary phase onset.",
        }

        total_risk_score = min(100.0, sum(s["score"] for s in stress_breakdown.values()))

        if total_risk_score >= 60.0:
            risk_level = "CRITICAL_WARNING"
            advisory = "CRITICAL: Imminent culture crash risk within 24-48 hours. Execute emergency 30-40% partial dilution harvest and adjust CO2 sparging to restore optimal pH."
            stability_window_hours = round(max(6.0, 48.0 - (total_risk_score - 60.0) * 0.8), 1)
        elif total_risk_score >= 30.0:
            risk_level = "ELEVATED_STRESS"
            advisory = "ELEVATED STRESS: Growth deceleration detected. Culture entering dense mutual-shading or carbon-limited regime. Plan dilution harvest within 48-72 hours."
            stability_window_hours = round(max(48.0, 96.0 - (total_risk_score - 30.0) * 1.5), 1)
        else:
            risk_level = "LOW_RISK_NOMINAL"
            advisory = "NOMINAL: Culture is in healthy exponential growth. Photobioreactor conditions are balanced."
            stability_window_hours = 120.0

        return {
            "crash_risk_index_pct": round(float(total_risk_score), 1),
            "risk_level": risk_level,
            "early_warning_advisory": advisory,
            "predicted_stability_window_hours": stability_window_hours,
            "stress_factor_breakdown": stress_breakdown,
            "empirical_basis": "Derived from Phase 7 growth dynamics audit identifying 23.7% negative growth transitions in stagnant cultures.",
        }

    def predict_next_day_growth(self, current_density: float, current_ph: float, light: float, co2_ppm: float, culture_day: int = 8) -> Dict[str, Any]:
        """
        Computes an experimental formula-based next-day estimate coupled with
        the Early-Warning Culture Crash Detector.
        """
        # Experimental logistic-style estimate; its parameters are not a validated trained model.
        # N_{t+1} = N_t + mu * N_t * (1 - N_t / K)
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

        # Biophysical Monod-Lambert kinetics (Grey-Box Baseline comparison)
        k_s_co2 = 180.0  # ppm half-saturation constant
        mu_monod = 0.38 * (co2_ppm / (k_s_co2 + co2_ppm))
        # Beer-Lambert self-shading photon attenuation: I_eff = I_0 * exp(-k_ext * N_t)
        k_ext = 0.038
        i_eff = light * np.exp(-k_ext * current_density)
        k_light = 120.0
        light_kinetics = i_eff / (k_light + i_eff)
        mu_biophysical = mu_monod * light_kinetics * ph_factor
        n_analytical_next = round(float(current_density + (mu_biophysical * current_density)), 2)
        grey_box_residual = round(float(predicted_next_day_density - n_analytical_next), 2)

        crash_risk = self.assess_culture_crash_risk(current_density, current_ph, light, co2_ppm, culture_day)

        return {
            "model_type": "Experimental formula-based growth estimate coupled with Early-Warning Crash Detector",
            "inputs": {
                "current_cell_density": current_density,
                "current_ph": current_ph,
                "light_ppfd": light,
                "co2_ppm": co2_ppm,
                "culture_day": culture_day,
            },
            "predictions": {
                "predicted_next_day_density_million_ml": predicted_next_day_density,
                "predicted_density_delta": round(predicted_next_day_density - current_density, 2),
                "predicted_specific_growth_rate_mu_day": round(float(mu_effective), 3),
                "doubling_time_days": doubling_time_days,
                "culture_phase": "EXPONENTIAL" if current_density < 20.0 else "DECELERATION/STATIONARY",
            },
            "biophysical_grey_box": {
                "monod_specific_growth_rate": round(float(mu_monod), 3),
                "effective_photon_flux_ppfd": round(float(i_eff), 1),
                "analytical_first_principles_density": n_analytical_next,
                "grey_box_residual": grey_box_residual,
                "physics_basis": "Monod substrate saturation coupled with Beer-Lambert mutual shading exponential decay.",
            },
            "crash_risk_assessment": crash_risk,
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
            total_cell_biovolume_um3 = 0.0
            total_cell_dry_mass_pg = 0.0
            cell_diameters = []

            for b in results.boxes:
                cls_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                xc = ((x1 + x2) / 2.0) / 640.0
                yc = ((y1 + y2) / 2.0) / 640.0
                w = (x2 - x1) / 640.0
                h = (y2 - y1) / 640.0

                det_entry = {
                    "class_id": cls_id,
                    "class_name": "chlorella_cell" if cls_id == 0 else "cluster_noise",
                    "confidence": round(conf, 3),
                    "box": [round(xc, 4), round(yc, 4), round(w, 4), round(h, 4)],
                }

                if cls_id == 0:
                    pred_class0 += 1
                    # Geometric volume estimation (prolate spheroid model from bounding box)
                    # 640x640 hemocytometer brightfield crop scaling: 0.3125 um/pixel
                    scale_um_per_px = 0.3125
                    w_um = max(0.1, (x2 - x1) * scale_um_per_px)
                    h_um = max(0.1, (y2 - y1) * scale_um_per_px)
                    major_axis_um = max(w_um, h_um)
                    minor_axis_um = min(w_um, h_um)
                    vol_um3 = (4.0 / 3.0) * np.pi * (major_axis_um / 2.0) * ((minor_axis_um / 2.0) ** 2)
                    eq_diam_um = 2.0 * ((3.0 * vol_um3) / (4.0 * np.pi)) ** (1.0 / 3.0)
                    dry_mass_pg = vol_um3 * 0.2835  # 1.05 g/cm^3 wet density * 27% dry matter fraction

                    det_entry["geometric_biovolume"] = {
                        "major_axis_um": round(float(major_axis_um), 2),
                        "minor_axis_um": round(float(minor_axis_um), 2),
                        "equivalent_diameter_um": round(float(eq_diam_um), 2),
                        "biovolume_um3": round(float(vol_um3), 1),
                        "est_dry_mass_pg": round(float(dry_mass_pg), 2),
                    }
                    total_cell_biovolume_um3 += vol_um3
                    total_cell_dry_mass_pg += dry_mass_pg
                    cell_diameters.append(eq_diam_um)
                else:
                    pred_class1 += 1

                detections.append(det_entry)

            # Read only after successful inference so image I/O failures are not reported as success.
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            # Uncertainty Quantification & Active Learning Triage
            cell_confs = [d["confidence"] for d in detections if d["class_id"] == 0]
            mean_conf = round(float(np.mean(cell_confs)), 3) if cell_confs else 0.0
            borderline_cells = sum(1 for c in cell_confs if c < 0.55)
            borderline_ratio = (borderline_cells / max(pred_class0, 1)) if pred_class0 > 0 else 0.0
            cluster_ratio = (pred_class1 / max(pred_class0 + pred_class1, 1))

            optical_score = max(10.0, min(100.0, 100.0 - (borderline_ratio * 40.0) - (cluster_ratio * 35.0)))
            if optical_score >= 80.0:
                triage_decision = "AUTONOMOUS_COUNT_VERIFIED"
                triage_advisory = "High detection confidence with minimal debris interference. Frame suitable for autonomous counting."
            elif optical_score >= 60.0:
                triage_decision = "MODERATE_UNCERTAINTY"
                triage_advisory = "Moderate boundary ambiguity or clumping present. Reliable for trend tracking."
            else:
                triage_decision = "FLAGGED_FOR_ACTIVE_LEARNING_REVIEW"
                triage_advisory = "High ambiguity or clumping interference. Frame flagged for human-in-the-loop inspection."
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
            "biomass_estimation": {
                "total_biovolume_um3": round(float(total_cell_biovolume_um3), 1),
                "mean_cell_diameter_um": round(float(np.mean(cell_diameters)), 2) if cell_diameters else 0.0,
                "mean_cell_volume_um3": round(float(total_cell_biovolume_um3 / max(pred_class0, 1)), 1) if pred_class0 > 0 else 0.0,
                "total_estimated_cell_dry_mass_pg": round(float(total_cell_dry_mass_pg), 2),
                "optical_scale_um_per_px": 0.3125,
                "biophysical_basis": "Prolate spheroid integration from YOLO bounding boxes; 27% dry matter fraction (0.283 pg/um^3).",
            },
            "uncertainty_and_triage": {
                "mean_detection_confidence": mean_conf,
                "borderline_detection_count": borderline_cells,
                "optical_quality_score": round(float(optical_score), 1),
                "triage_decision": triage_decision,
                "triage_advisory": triage_advisory,
                "cluster_interference_pct": round(float(cluster_ratio * 100.0), 1),
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

    def execute_closed_loop_actuation(
        self,
        current_ph: float = 7.42,
        current_density: float = 18.4,
        current_light: float = 240.0,
        co2_ppm: float = 412.0,
        mode: str = "AUTO",
    ) -> Dict[str, Any]:
        """
        Closed-Loop Cyber-Physical Actuation Engine (AI-MPC).
        Simulates automated valve, sparger, LED driver, and harvest cycles.
        """
        # 1. Automated Solenoid Valve CO2 Sparging Control (Predictive pH Regulation)
        if current_ph > 7.55:
            pulse_sec = round(min(15.0, max(2.5, (current_ph - 7.4) * 24.0 + (current_density * 0.15))), 1)
            valve_state = "PULSING_OPEN"
            gas_flow_slpm = 2.5
            predicted_stabilization_min = round(3.5 + (current_ph - 7.4) * 5.0, 1)
            target_stabilized_ph = 7.40
            action_desc = f"Alkaline drift detected ({current_ph:.2f}). Actuating CO2 solenoid valve for {pulse_sec}s pulse at {gas_flow_slpm} SLPM."
        elif current_ph < 6.80:
            pulse_sec = 0.0
            valve_state = "EMERGENCY_LOCKOUT"
            gas_flow_slpm = 0.0
            predicted_stabilization_min = 0.0
            target_stabilized_ph = current_ph
            action_desc = f"Acidic shock threshold breached ({current_ph:.2f}). Gas inflow locked closed; auto-buffering alert."
        else:
            pulse_sec = 0.0
            valve_state = "CLOSED_STANDBY"
            gas_flow_slpm = 0.0
            predicted_stabilization_min = 0.0
            target_stabilized_ph = current_ph
            action_desc = f"pH is nominal ({current_ph:.2f}). Sparging valve on idle standby."

        sparging_actuator = {
            "valve_id": "SV-CO2-01",
            "valve_state": valve_state,
            "pulse_duration_seconds": pulse_sec,
            "gas_flow_slpm": gas_flow_slpm,
            "target_stabilized_ph": target_stabilized_ph,
            "predicted_stabilization_minutes": predicted_stabilization_min,
            "control_algorithm": "Predictive Dissolution Kinetics (AI-MPC)",
            "action_summary": action_desc,
        }

        # 2. Automated Adaptive LED Driver (Beer-Lambert Depth Compensation)
        k_ext = 0.038
        interior_photon_penetration_pct = round(float(np.exp(-k_ext * current_density) * 100.0), 1)

        if current_density < 10.0:
            target_ppfd = 180.0
            dimming_pct = 60.0
            spectrum = "3:1 Blue:Red (Vegetative Inoculation)"
            power_saving_pct = 40.0
        elif current_density < 20.0:
            target_ppfd = 260.0
            dimming_pct = 85.0
            spectrum = "4:1 Red:Blue (Active Photosynthesis & Rubisco)"
            power_saving_pct = 15.0
        else:
            target_ppfd = 340.0
            dimming_pct = 100.0
            spectrum = "5:1 Red:Blue (Dense Core Deep Penetration)"
            power_saving_pct = 0.0

        lighting_actuator = {
            "driver_id": "LED-PWM-DIM-01",
            "target_ppfd_umol": target_ppfd,
            "driver_output_power_pct": dimming_pct,
            "calculated_interior_core_penetration_pct": interior_photon_penetration_pct,
            "optimized_photonic_spectrum": spectrum,
            "dynamic_power_saving_pct": power_saving_pct,
            "control_algorithm": "Beer-Lambert Optical Extinction Modulation",
        }

        # 3. Automated Dilution Harvesting Cycle
        harvest_threshold = 24.0
        if current_density >= harvest_threshold:
            harvest_status = "CYCLE_TRIGGERED"
            drain_valve = "ACTUATING_OPEN"
            drain_volume = 180.0
            refill_volume = 180.0
            reset_density = round(current_density * 0.70, 1)
            harvested_biomass_kg = round(drain_volume * (current_density * 0.05) / 1000.0 * 20.0, 2)
            harvest_desc = f"Density threshold ({current_density:.1f}M >= {harvest_threshold}M) reached. Triggering 180L auto-drain and sterile media replenishment."
        else:
            harvest_status = "ACCUMULATION_STANDBY"
            drain_valve = "CLOSED"
            drain_volume = 0.0
            refill_volume = 0.0
            reset_density = current_density
            harvested_biomass_kg = 0.0
            harvest_desc = f"Culture accumulating ({current_density:.1f}M / {harvest_threshold}M). Auto-harvest valve closed."

        harvest_actuator = {
            "valve_drain_id": "V-DRAIN-01",
            "valve_refill_id": "V-REFILL-02",
            "cycle_status": harvest_status,
            "drain_valve_state": drain_valve,
            "drain_volume_liters": drain_volume,
            "refill_volume_liters": refill_volume,
            "predicted_reset_density_million_ml": reset_density,
            "estimated_harvested_slurry_kg": harvested_biomass_kg,
            "action_summary": harvest_desc,
        }

        return {
            "system_mode": mode,
            "cyber_physical_state": "NOMINAL_CLOSED_LOOP",
            "timestamp": "2026-10-30T01:05:00Z",
            "actuators": {
                "co2_sparging_valve": sparging_actuator,
                "adaptive_led_lighting": lighting_actuator,
                "automated_harvest_valves": harvest_actuator,
            },
            "disclaimer": "CYBER-PHYSICAL DIGITAL TWIN: Real-time autonomous actuator command stream. Connectable to physical relay board (Modbus/MQTT/GPIO).",
        }

    def simulate_mobile_liquid_tree_dispatch(
        self,
        target_zone: str = "boardroom_a",
        custom_co2_ppm: float | None = None,
    ) -> Dict[str, Any]:
        """
        Simulates an Autonomous Mobile Robot (AMR) Liquid Tree pod (60L Chlorella column)
        dispatched across corporate building indoor air quality (IAQ) zones.
        """
        zones = {
            "boardroom_a": {
                "name": "Executive Boardroom A",
                "occupancy": 8,
                "room_volume_m3": 75.0,
                "default_co2_ppm": 1650.0,
                "priority": "HIGH_URGENT",
                "air_quality_status": "POOR_FATIGUE_INDUCING",
                "transit_distance_meters": 35.0,
            },
            "conference_room_b": {
                "name": "Team Conference Room B",
                "occupancy": 3,
                "room_volume_m3": 50.0,
                "default_co2_ppm": 1050.0,
                "priority": "MEDIUM",
                "air_quality_status": "MODERATE_ELEVATED",
                "transit_distance_meters": 18.0,
            },
            "open_workspace": {
                "name": "Engineering Open Desks",
                "occupancy": 22,
                "room_volume_m3": 280.0,
                "default_co2_ppm": 640.0,
                "priority": "LOW",
                "air_quality_status": "HEALTHY",
                "transit_distance_meters": 45.0,
            },
            "docking_station": {
                "name": "Autonomous Mother Base Dock",
                "occupancy": 0,
                "room_volume_m3": 25.0,
                "default_co2_ppm": 420.0,
                "priority": "CHARGING_MAINTENANCE",
                "air_quality_status": "CLEAN",
                "transit_distance_meters": 0.0,
            },
        }

        selected_zone = zones.get(target_zone, zones["boardroom_a"])
        start_co2 = custom_co2_ppm if custom_co2_ppm is not None else selected_zone["default_co2_ppm"]
        room_vol_m3 = selected_zone["room_volume_m3"]

        robot_specs = {
            "robot_id": "AMR-LT-01 (AeroTree Pod)",
            "algae_working_volume_liters": 60.0,
            "strain": "Chlorella vulgaris (UTEX 2714)",
            "chassis_type": "Mecanum Omnidirectional Mobile Base",
            "navigation_stack": "2D/3D LiDAR SLAM + Ultrasonic Obstacle Avoidance",
            "air_displacement_blower_cfm": 85.0,
            "battery_level_pct": 82.0,
            "docking_state": "UNDOCKED_MISSION_ACTIVE" if target_zone != "docking_station" else "DOCKED_CHARGING",
        }

        target_co2 = 750.0
        delta_co2_ppm = max(0.0, start_co2 - target_co2)

        hourly_co2_fixation_grams = 18.2
        co2_mass_to_remove_grams = (delta_co2_ppm / 1000.0) * 1.8 * room_vol_m3
        scrub_time_minutes = round(max(5.0, min(120.0, (co2_mass_to_remove_grams / max(hourly_co2_fixation_grams, 1.0)) * 60.0)), 1)
        est_o2_produced_grams = round(co2_mass_to_remove_grams * (32.0 / 44.0), 1)
        transit_time_min = round(selected_zone["transit_distance_meters"] / 0.8 / 60.0, 1)

        mission = {
            "dispatch_zone": selected_zone["name"],
            "zone_key": target_zone,
            "priority": selected_zone["priority"],
            "initial_co2_ppm": start_co2,
            "target_co2_ppm": target_co2 if target_zone != "docking_station" else 420.0,
            "transit_time_minutes": transit_time_min,
            "estimated_scrub_time_minutes": scrub_time_minutes if target_zone != "docking_station" else 0.0,
            "estimated_pure_o2_released_grams": est_o2_produced_grams if target_zone != "docking_station" else 0.0,
            "estimated_battery_draw_pct": round(min(30.0, 4.0 + (scrub_time_minutes * 0.25)), 1),
            "mission_status": "MISSION_DISPATCHED" if target_zone != "docking_station" else "RETURNING_TO_BASE",
            "dispatch_advisory": (
                f"Elevated CO2 detected in {selected_zone['name']} ({start_co2:.0f} ppm). "
                f"AMR Liquid Tree dispatched. ETA: {transit_time_min} min. "
                f"Active Monod scrubbing will remediate IAQ to {target_co2:.0f} ppm in ~{scrub_time_minutes} min, generating {est_o2_produced_grams}g of fresh O2."
            ) if target_zone != "docking_station" else "Robot returning to Autonomous Base Dock for inductive charging and media top-up.",
        }

        return {
            "robot": robot_specs,
            "mission": mission,
            "all_zones": zones,
        }
