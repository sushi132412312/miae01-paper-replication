"""离线边界检查；这里的样例均是人工数据，不是模型成绩。"""
import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common import PROJECT_ROOT, read_json
from src.data import fixed_subset, manifest_for, validate_real_config
from src.evaluation import export_predictions, recompute_accuracy, summarize_seed_results


class DataSelectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"sample_id": f"train:{i}", "label": i % 2, "sentence": f"人工例子 {i}"}
                     for i in range(30)]

    def test_fixed_subset_is_repeatable_and_ignores_global_rng(self):
        first = fixed_subset(self.rows, 8, 2026)
        random.seed(44)
        for _ in range(30):
            random.random()
        self.assertEqual(first, fixed_subset(self.rows, 8, 2026))

    def test_fixed_subset_does_not_change_global_rng(self):
        before = random.getstate()
        fixed_subset(self.rows, 8, 2026)
        self.assertEqual(before, random.getstate())

    def test_bad_counts_and_duplicate_ids_are_rejected(self):
        for count in (0, -1, 31, True):
            with self.assertRaises(ValueError):
                fixed_subset(self.rows, count, 2026)
        with self.assertRaises(ValueError):
            fixed_subset([self.rows[0], self.rows[0]], 1, 2026)

    def test_manifest_detects_text_or_label_change(self):
        original = manifest_for(self.rows, "train", 2026)
        altered = copy.deepcopy(self.rows)
        altered[0]["sentence"] = "实际内容变化"
        self.assertNotEqual(original["selection_sha256"], manifest_for(altered, "train", 2026)["selection_sha256"])
        self.assertEqual(original["count"], 30)
        self.assertEqual(original["label_counts"], {"0": 15, "1": 15})

    def test_pending_real_training_is_rejected_before_output_creation(self):
        from src.train import run
        config = read_json(PROJECT_ROOT / "configs/pilot.pending.json")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            with patch("src.train.build_model", side_effect=AssertionError("不应加载模型")):
                with self.assertRaises(PermissionError):
                    run(config, "ft", output)
            self.assertFalse(output.exists())

    def test_real_config_rejects_floating_revisions(self):
        config = read_json(PROJECT_ROOT / "configs/pilot.pending.json")
        config["status"] = "approved_engineering_pilot"
        config["model_revision"] = "main"
        with self.assertRaises(ValueError):
            validate_real_config(config)


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "predictions.csv"
        self.manifest = {"samples": [{"sample_id": f"validation:{i}", "label": label}
                                     for i, label in enumerate([1, 0, 0])]}
        self.rows = [{"run_id": "artificial-example", "method": "ft", "seed": 42,
                      "checkpoint": "artificial", "sample_id": f"validation:{i}",
                      "gold_label": label, "predicted_label": prediction}
                     for i, (label, prediction) in enumerate(zip([1, 0, 0], [1, 0, 1]))]

    def check_rows(self, rows):
        export_predictions(self.path, rows)
        return recompute_accuracy(self.path, self.manifest)

    def test_accuracy_recomputed_from_ids_not_row_order(self):
        result = self.check_rows(list(reversed(self.rows)))
        self.assertEqual(result["correct"], 2)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)

    def test_missing_record_rejected(self):
        with self.assertRaises(ValueError):
            self.check_rows(self.rows[:-1])

    def test_duplicate_record_rejected(self):
        with self.assertRaises(ValueError):
            self.check_rows(self.rows + [self.rows[0]])

    def test_wrong_gold_label_rejected(self):
        self.rows[0]["gold_label"] = 0
        with self.assertRaises(ValueError):
            self.check_rows(self.rows)

    def test_invalid_prediction_rejected(self):
        self.rows[0]["predicted_label"] = -1
        with self.assertRaises(ValueError):
            self.check_rows(self.rows)

    def test_mixed_checkpoint_rejected(self):
        self.rows[0]["checkpoint"] = "another-model"
        with self.assertRaises(ValueError):
            self.check_rows(self.rows)

    def test_unexpected_run_rejected(self):
        export_predictions(self.path, self.rows)
        with self.assertRaises(ValueError):
            recompute_accuracy(self.path, self.manifest, {"method": "lora"})


class SeedStatisticsTests(unittest.TestCase):
    def results(self):
        return [{"seed": seed, "accuracy_percent": value, "method": "ft",
                 "experiment_kind": "artificial-example", "protocol_sha256": "same",
                 "validation_selection_sha256": "same"}
                for seed, value in zip((42, 43, 44), (80., 82., 84.))]

    def test_sample_standard_deviation(self):
        result = summarize_seed_results(self.results())
        self.assertEqual(result["mean_percent"], 82.)
        self.assertEqual(result["sample_sd_pp"], 2.)

    def test_one_seed_does_not_report_zero_sd(self):
        result = summarize_seed_results(self.results()[:1])
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["sample_sd_pp"])

    def test_duplicate_seed_rejected(self):
        results = self.results()
        results[1]["seed"] = 42
        with self.assertRaises(ValueError):
            summarize_seed_results(results)

    def test_mixed_protocol_or_validation_rejected(self):
        for key in ("protocol_sha256", "validation_selection_sha256", "experiment_kind"):
            results = self.results()
            results[0][key] = "different"
            with self.assertRaises(ValueError):
                summarize_seed_results(results)

    def test_non_finite_score_rejected(self):
        results = self.results()
        results[0]["accuracy_percent"] = float("nan")
        with self.assertRaises(ValueError):
            summarize_seed_results(results)


if __name__ == "__main__":
    unittest.main()
