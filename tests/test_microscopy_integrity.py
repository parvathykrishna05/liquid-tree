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


if __name__ == "__main__":
    unittest.main()
