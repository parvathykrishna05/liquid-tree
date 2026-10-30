import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from api.service import LiquidTreeService


class MicroscopyIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.service = LiquidTreeService(root)
        self.service.val_images_dir.mkdir(parents=True)
        self.service.val_labels_dir.mkdir(parents=True)
        self.service.checkpoint_path.parent.mkdir(parents=True)
        self.image_path = self.service.val_images_dir / "sample.png"
        self.image_path.write_bytes(b"image bytes for mocked inference")
        self.label_path = self.service.val_labels_dir / "sample.txt"
        self.label_path.write_text("0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_selected_checkpoint_path_is_configured_without_loading_model(self):
        expected = (
            Path(self.temp_dir.name)
            / "models"
            / "phase9c_yolo"
            / "train_20epochs"
            / "weights"
            / "best.pt"
        )
        self.assertEqual(self.service.checkpoint_path, expected)
        self.assertEqual(self.service.checkpoint_id, "phase9c_yolo_train_20epochs_best")

    def test_default_alerts_do_not_claim_sensor_vision_agreement_without_measurements(self):
        alerts = self.service.get_system_alerts()

        self.assertFalse(any(alert["category"].startswith("SENSOR_VISION_") for alert in alerts))
        messages = " ".join(alert["message"] for alert in alerts).lower()
        self.assertIn("simulated", messages)
        self.assertNotIn("valve idle", messages)
        self.assertNotIn("heat exchange active", messages)

    def test_missing_weights_never_turn_annotations_into_predictions(self):
        result = self.service.run_vision_detection("sample")

        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertEqual(result["checkpoint_path"], "models/phase9c_yolo/train_20epochs/weights/best.pt")
        self.assertNotIn("detections", result)
        self.assertNotIn("counts", result)
        self.assertNotEqual(result["status"], "SUCCESS")

    def test_inference_exception_never_turns_annotations_into_predictions(self):
        self.service.checkpoint_path.touch()
        model = Mock()
        model.predict.side_effect = RuntimeError("decode failed")
        self.service._yolo_model = model

        result = self.service.run_vision_detection("sample")

        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["checkpoint_id"], "phase9c_yolo_train_20epochs_best")
        self.assertIn("decode failed", result["error"])
        self.assertNotIn("detections", result)
        self.assertNotIn("counts", result)

    def test_successful_empty_inference_has_zero_predictions_and_no_density_calibration(self):
        self.service.checkpoint_path.touch()
        self.service._yolo_model = Mock()
        self.service._yolo_model.predict.return_value = [Mock(boxes=[])]

        result = self.service.run_vision_detection("sample")

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["checkpoint_path"], "models/phase9c_yolo/train_20epochs/weights/best.pt")
        self.assertEqual(result["counts"]["predicted_individual_cells_class0"], 0)
        self.assertEqual(result["counts"]["predicted_clusters_debris_class1"], 0)
        self.assertEqual(result["counts"]["ground_truth_cells_class0"], 1)
        self.assertEqual(result["detections"], [])
        self.assertIsNone(result["counts"]["vision_derived_density_million_ml"])
        self.assertIn("Calibration unavailable", result["counts"]["density_message"])

    def test_successful_predictions_remain_separate_from_ground_truth(self):
        self.service.checkpoint_path.touch()

        def box(cls_id, confidence, xyxy):
            item = lambda value: Mock(item=lambda: value)
            coords = Mock(tolist=lambda: xyxy)
            return SimpleNamespace(cls=[item(cls_id)], conf=[item(confidence)], xyxy=[coords])

        self.service._yolo_model = Mock()
        self.service._yolo_model.predict.return_value = [
            Mock(boxes=[box(0, 0.91, [100, 100, 200, 200]), box(1, 0.82, [250, 250, 300, 300])])
        ]

        result = self.service.run_vision_detection("sample")

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["counts"]["predicted_individual_cells_class0"], 1)
        self.assertEqual(result["counts"]["predicted_clusters_debris_class1"], 1)
        self.assertEqual(result["counts"]["ground_truth_cells_class0"], 1)
        self.assertEqual(len(result["detections"]), 2)
        self.assertIsNone(result["counts"]["vision_derived_density_million_ml"])
        self.assertIn("biomass_estimation", result)
        self.assertGreater(result["biomass_estimation"]["total_biovolume_um3"], 0.0)
        self.assertGreater(result["biomass_estimation"]["total_estimated_cell_dry_mass_pg"], 0.0)

    def test_biomass_geometric_volume_estimator_zero_when_empty(self):
        self.service.checkpoint_path.touch()
        self.service._yolo_model = Mock()
        self.service._yolo_model.predict.return_value = [Mock(boxes=[])]

        result = self.service.run_vision_detection("sample")

        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("biomass_estimation", result)
        self.assertEqual(result["biomass_estimation"]["total_biovolume_um3"], 0.0)
        self.assertEqual(result["biomass_estimation"]["mean_cell_diameter_um"], 0.0)
        self.assertEqual(result["biomass_estimation"]["total_estimated_cell_dry_mass_pg"], 0.0)

    def test_culture_crash_risk_assessment_nominal_and_critical(self):
        # Nominal healthy conditions
        nominal = self.service.assess_culture_crash_risk(
            current_density=12.0,
            current_ph=7.4,
            light=250.0,
            co2_ppm=600.0,
            culture_day=5,
        )
        self.assertEqual(nominal["risk_level"], "LOW_RISK_NOMINAL")
        self.assertLess(nominal["crash_risk_index_pct"], 30.0)
        self.assertIn("stress_factor_breakdown", nominal)

        # Critical crash conditions: severe self-shading, high pH, carbon limitation, old batch
        critical = self.service.assess_culture_crash_risk(
            current_density=29.0,
            current_ph=8.6,
            light=50.0,
            co2_ppm=320.0,
            culture_day=15,
        )
        self.assertEqual(critical["risk_level"], "CRITICAL_WARNING")
        self.assertGreaterEqual(critical["crash_risk_index_pct"], 60.0)
        self.assertIn("emergency", critical["early_warning_advisory"].lower())
        self.assertLessEqual(critical["predicted_stability_window_hours"], 48.0)

    def test_biophysical_grey_box_kinetics_computed_in_prediction(self):
        pred = self.service.predict_next_day_growth(
            current_density=18.4,
            current_ph=7.42,
            light=240.0,
            co2_ppm=412.0,
            culture_day=8,
        )
        self.assertIn("biophysical_grey_box", pred)
        grey_box = pred["biophysical_grey_box"]
        self.assertGreater(grey_box["monod_specific_growth_rate"], 0.0)
        self.assertGreater(grey_box["effective_photon_flux_ppfd"], 0.0)
        self.assertIn("analytical_first_principles_density", grey_box)
        self.assertIn("grey_box_residual", grey_box)

    def test_uncertainty_and_active_learning_triage_in_vision(self):
        self.service.checkpoint_path.touch()

        def box(cls_id, confidence, xyxy):
            item = lambda value: Mock(item=lambda: value)
            coords = Mock(tolist=lambda: xyxy)
            return SimpleNamespace(cls=[item(cls_id)], conf=[item(confidence)], xyxy=[coords])

        self.service._yolo_model = Mock()
        self.service._yolo_model.predict.return_value = [
            Mock(boxes=[box(0, 0.92, [100, 100, 150, 150]), box(0, 0.48, [200, 200, 240, 240])])
        ]

        result = self.service.run_vision_detection("sample")

        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("uncertainty_and_triage", result)
        triage = result["uncertainty_and_triage"]
        self.assertEqual(triage["borderline_detection_count"], 1)
        self.assertGreaterEqual(triage["optical_quality_score"], 0.0)
        self.assertIn("triage_decision", triage)

    def test_closed_loop_actuation_sparging_and_harvest(self):
        # 1. Nominal case
        act_nom = self.service.execute_closed_loop_actuation(current_ph=7.42, current_density=18.4)
        self.assertEqual(act_nom["actuators"]["co2_sparging_valve"]["valve_state"], "CLOSED_STANDBY")
        self.assertEqual(act_nom["actuators"]["automated_harvest_valves"]["cycle_status"], "ACCUMULATION_STANDBY")

        # 2. Alkaline drift triggers CO2 pulse
        act_alk = self.service.execute_closed_loop_actuation(current_ph=7.82, current_density=18.4)
        self.assertEqual(act_alk["actuators"]["co2_sparging_valve"]["valve_state"], "PULSING_OPEN")
        self.assertGreater(act_alk["actuators"]["co2_sparging_valve"]["pulse_duration_seconds"], 0.0)

        # 3. High density triggers auto-harvest
        act_harv = self.service.execute_closed_loop_actuation(current_ph=7.40, current_density=26.5)
        self.assertEqual(act_harv["actuators"]["automated_harvest_valves"]["cycle_status"], "CYCLE_TRIGGERED")
        self.assertEqual(act_harv["actuators"]["automated_harvest_valves"]["drain_valve_state"], "ACTUATING_OPEN")
        self.assertGreater(act_harv["actuators"]["automated_harvest_valves"]["drain_volume_liters"], 0.0)

    def test_mobile_liquid_tree_dispatch_simulation(self):
        dispatch = self.service.simulate_mobile_liquid_tree_dispatch("boardroom_a")
        self.assertIn("robot", dispatch)
        self.assertIn("mission", dispatch)
        self.assertEqual(dispatch["mission"]["zone_key"], "boardroom_a")
        self.assertGreater(dispatch["mission"]["estimated_scrub_time_minutes"], 0.0)
        self.assertGreater(dispatch["mission"]["estimated_pure_o2_released_grams"], 0.0)
        self.assertEqual(dispatch["robot"]["algae_working_volume_liters"], 60.0)


if __name__ == "__main__":
    unittest.main()
