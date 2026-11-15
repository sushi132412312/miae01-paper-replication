"""Real execution of existing modules; no mocks or missing production replacements.

Temporary artificial inputs are fixtures, not fabricated training results.
The evaluation target is the EXISTING review copy, not missing src.evaluation.
"""
import copy
import csv
import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from third_party import loralib
from tools import mac_audit_support as support

ROOT = Path(__file__).resolve().parents[2]
PACKET = Path(os.environ.get("WEEK02_HANDOFF_DIR", ROOT.parent / "handoff_utf8/第2周任务_20261115"))
OFFICIAL = Path(os.environ.get("WEEK02_OFFICIAL_DIR", "/Users/xuan/Documents/LoRA-official"))
spec = importlib.util.spec_from_file_location("existing_evaluation_review", ROOT / "records/week02/recount_verification/verify_evaluation.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class LinearRealTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_forward_matches_explicit_low_rank_equation(self):
        layer = loralib.Linear(5, 3, r=2, lora_alpha=6, merge_weights=False)
        with torch.no_grad():
            layer.lora_B.fill_(0.2)
        x = torch.randn(2, 4, 5)
        expected = F.linear(x, layer.weight, layer.bias) + (x @ layer.lora_A.T @ layer.lora_B.T) * 3
        torch.testing.assert_close(layer(x), expected)

    def test_no_bias_and_batch_shape(self):
        layer = loralib.Linear(5, 3, r=2, bias=False, merge_weights=False)
        self.assertIsNone(layer.bias)
        self.assertEqual(layer(torch.randn(2, 4, 5)).shape, (2, 4, 3))

    def test_transposed_storage_preserves_forward(self):
        layer = loralib.Linear(5, 3, r=2, fan_in_fan_out=True, merge_weights=False)
        self.assertEqual(layer.weight.shape, (5, 3))
        x = torch.randn(2, 5)
        torch.testing.assert_close(layer(x), F.linear(x, layer.weight.T, layer.bias))

    def test_zero_rank_disables_adapter_in_official_layer(self):
        layer = loralib.Linear(5, 3, r=0)
        self.assertFalse(hasattr(layer, "lora_A"))
        self.assertTrue(layer.weight.requires_grad)
        x = torch.randn(2, 5)
        torch.testing.assert_close(layer(x), F.linear(x, layer.weight, layer.bias))

    def test_first_step_gradients_and_frozen_weight(self):
        layer = loralib.Linear(5, 3, r=2, lora_alpha=2, merge_weights=False)
        old = layer.weight.detach().clone()
        opt = torch.optim.SGD((p for p in layer.parameters() if p.requires_grad), lr=0.1)
        layer(torch.ones(2, 5)).square().sum().backward()
        self.assertIsNone(layer.weight.grad)
        self.assertEqual(torch.count_nonzero(layer.lora_A.grad).item(), 0)
        self.assertGreater(torch.count_nonzero(layer.lora_B.grad).item(), 0)
        opt.step()
        self.assertTrue(torch.equal(layer.weight, old))
        self.assertGreater(torch.count_nonzero(layer.lora_B).item(), 0)

    def test_second_step_can_update_A(self):
        layer = loralib.Linear(5, 3, r=2, lora_alpha=2, merge_weights=False)
        opt = torch.optim.SGD((p for p in layer.parameters() if p.requires_grad), lr=0.1)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            layer(torch.ones(2, 5)).square().sum().backward()
            opt.step()
        self.assertGreater(torch.count_nonzero(layer.lora_A.grad).item(), 0)

    def test_merge_unmerge_preserves_output(self):
        layer = loralib.Linear(5, 3, r=2, lora_alpha=2, merge_weights=True)
        with torch.no_grad():
            layer.lora_B.fill_(0.2)
        x = torch.randn(2, 5)
        before = layer.weight.detach().clone()
        output = layer(x).detach()
        layer.eval()
        self.assertTrue(layer.merged)
        torch.testing.assert_close(output, layer(x))
        layer.eval()
        layer.train()
        self.assertFalse(layer.merged)
        torch.testing.assert_close(before, layer.weight)
        torch.testing.assert_close(output, layer(x))

    def test_wrong_input_dimension_raises(self):
        with self.assertRaises(RuntimeError):
            loralib.Linear(5, 3, r=2)(torch.ones(2, 4))

    def test_invalid_dropout_rejected_by_torch(self):
        with self.assertRaises(ValueError):
            loralib.Linear(5, 3, r=2, lora_dropout=1.5)


class OfficialUtilitiesRealTests(unittest.TestCase):
    def model(self):
        return nn.Sequential(loralib.Linear(5, 3, r=2, merge_weights=False), nn.Linear(3, 2))

    def test_bias_none_freezes_non_adapter_including_head(self):
        model = self.model()
        loralib.mark_only_lora_as_trainable(model)
        self.assertTrue(all(p.requires_grad == ("lora_" in n) for n, p in model.named_parameters()))

    def test_all_bias_mode(self):
        model = self.model()
        loralib.mark_only_lora_as_trainable(model, bias="all")
        self.assertTrue(all(p.requires_grad == ("lora_" in n or "bias" in n) for n, p in model.named_parameters()))

    def test_lora_only_bias_mode(self):
        model = self.model()
        loralib.mark_only_lora_as_trainable(model, bias="lora_only")
        self.assertTrue(model[0].bias.requires_grad)
        self.assertFalse(model[1].bias.requires_grad)

    def test_state_dict_bias_modes(self):
        model = self.model()
        self.assertEqual(set(loralib.lora_state_dict(model)), {"0.lora_A", "0.lora_B"})
        self.assertEqual(set(loralib.lora_state_dict(model, bias="all")), {"0.lora_A", "0.lora_B", "0.bias", "1.bias"})
        self.assertEqual(set(loralib.lora_state_dict(model, bias="lora_only")), {"0.lora_A", "0.lora_B", "0.bias"})

    def test_invalid_bias_flags_raise(self):
        for function in (loralib.mark_only_lora_as_trainable, loralib.lora_state_dict):
            with self.subTest(function=function.__name__), self.assertRaises(NotImplementedError):
                function(self.model(), bias="invalid")

    def test_adapter_state_roundtrip(self):
        first, second = self.model(), self.model()
        with torch.no_grad():
            first[0].lora_B.fill_(0.5)
        second.load_state_dict(loralib.lora_state_dict(first), strict=False)
        self.assertTrue(torch.equal(first[0].lora_A, second[0].lora_A))
        self.assertTrue(torch.equal(first[0].lora_B, second[0].lora_B))


class OtherLayersSmokeRealTests(unittest.TestCase):
    def test_embedding_initial_zero_update(self):
        layer = loralib.Embedding(9, 4, r=2, merge_weights=False)
        self.assertEqual(torch.count_nonzero(layer.lora_A).item(), 0)
        self.assertGreater(torch.count_nonzero(layer.lora_B).item(), 0)
        indices = torch.tensor([[1, 2]])
        torch.testing.assert_close(layer(indices), F.embedding(indices, layer.weight))

    def test_merged_linear_disabled_middle_group(self):
        layer = loralib.MergedLinear(4, 6, r=2, enable_lora=[True, False, True], merge_weights=False)
        with torch.no_grad():
            layer.lora_B.fill_(0.1)
        delta = layer.merge_AB()
        self.assertEqual(torch.count_nonzero(delta[2:4]).item(), 0)
        x = torch.randn(2, 4)
        torch.testing.assert_close(layer(x), F.linear(x, layer.weight + delta * layer.scaling, layer.bias))

    def test_merged_linear_invalid_partition(self):
        with self.assertRaises(AssertionError):
            loralib.MergedLinear(4, 5, r=2, enable_lora=[True, False])

    def test_convolution_family_kernel_one_smoke(self):
        for dimension, cls in ((1, loralib.Conv1d), (2, loralib.Conv2d), (3, loralib.Conv3d)):
            with self.subTest(dimension=dimension):
                layer = cls(2, 3, kernel_size=1, r=2, merge_weights=False)
                x = torch.randn((1, 2) + (3,) * dimension)
                torch.testing.assert_close(layer(x), layer.conv(x))
                self.assertFalse(layer.conv.weight.requires_grad)


class AuditSupportRealTests(unittest.TestCase):
    def test_seed_controls_python_numpy_torch(self):
        def sample():
            return random.random(), np.random.random(), torch.rand(2)
        support.seed_everything(71)
        first = sample()
        support.seed_everything(71)
        second = sample()
        self.assertEqual(first[:2], second[:2])
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_tensor_hash_order_and_grad_are_irrelevant(self):
        a, b = torch.ones(2, requires_grad=True), torch.zeros(3)
        self.assertEqual(support.tensor_hash([("a", a), ("b", b)]), support.tensor_hash([("b", b), ("a", a.detach())]))

    def test_hash_detects_name_shape_dtype_value(self):
        original = support.tensor_hash([("a", torch.ones(2))])
        for name, tensor in [("b", torch.ones(2)), ("a", torch.ones(1, 2)), ("a", torch.ones(2, dtype=torch.float64)), ("a", torch.zeros(2))]:
            with self.subTest(name=name, shape=list(tensor.shape), dtype=str(tensor.dtype)):
                self.assertNotEqual(original, support.tensor_hash([(name, tensor)]))

    def test_save_json_roundtrip_nested_unicode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "中文/记录.json"
            support.save_json(path, {"核验": [1, True, None]})
            self.assertEqual(json.loads(path.read_text()), {"核验": [1, True, None]})

    def test_save_json_rejects_nan_and_unsupported_object(self):
        with tempfile.TemporaryDirectory() as directory:
            for value, error in [(float("nan"), ValueError), (object(), TypeError)]:
                with self.subTest(error=error.__name__), self.assertRaises(error):
                    support.save_json(Path(directory) / "invalid.json", value)

    def test_execution_context_is_real_and_not_student_claim(self):
        context = support.execution_context()
        self.assertFalse(context["student_independent_work"])
        self.assertEqual(len(context["git_head_at_execution"]), 40)
        self.assertEqual(context["source_sha256"]["src/models.py"], hashlib.sha256((ROOT / "src/models.py").read_bytes()).hexdigest())


class ExistingReviewEvaluationRealTests(unittest.TestCase):
    """Runs the existing review copy, without asserting identity to missing source."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "predictions.csv"
        evaluation.write_predictions(self.path, dict(evaluation.LABELS))
        self.manifest = evaluation.manifest()

    def mutate(self, operation):
        with self.path.open(newline="") as file:
            rows = list(csv.DictReader(file))
        rows = operation(rows)
        with self.path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=evaluation.FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_recount_by_ids_after_reorder_and_manifest_path(self):
        self.mutate(lambda rows: list(reversed(rows)))
        manifest = self.path.with_suffix(".json")
        manifest.write_text(json.dumps(self.manifest))
        result = evaluation.recompute_accuracy(self.path, manifest)
        self.assertEqual((result["correct"], result["count"], result["accuracy"]), (6, 6, 1))

    def test_duplicate_row_rejected(self):
        self.mutate(lambda rows: rows + [rows[-1]])
        with self.assertRaisesRegex(ValueError, "重复"):
            evaluation.recompute_accuracy(self.path, self.manifest)

    def test_missing_row_rejected(self):
        self.mutate(lambda rows: rows[:-1])
        with self.assertRaisesRegex(ValueError, "缺失"):
            evaluation.recompute_accuracy(self.path, self.manifest)

    def test_unknown_id_rejected(self):
        self.mutate(lambda rows: [dict(rows[0], sample_id="unknown")] + rows[1:])
        with self.assertRaisesRegex(ValueError, "未知"):
            evaluation.recompute_accuracy(self.path, self.manifest)

    def test_wrong_gold_label_rejected(self):
        self.mutate(lambda rows: [dict(rows[0], gold_label="0")] + rows[1:])
        with self.assertRaisesRegex(ValueError, "金标签"):
            evaluation.recompute_accuracy(self.path, self.manifest)

    def test_invalid_prediction_rejected(self):
        self.mutate(lambda rows: [dict(rows[0], predicted_label="2")] + rows[1:])
        with self.assertRaisesRegex(ValueError, "0 或 1"):
            evaluation.recompute_accuracy(self.path, self.manifest)

    def test_mixed_checkpoint_rejected(self):
        self.mutate(lambda rows: [dict(rows[0], checkpoint="other")] + rows[1:])
        with self.assertRaisesRegex(ValueError, "混入"):
            evaluation.recompute_accuracy(self.path, self.manifest)

    def test_expected_run_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "预期"):
            evaluation.recompute_accuracy(self.path, self.manifest, {"seed": 999})

    def test_bad_header_rejected(self):
        self.path.write_text("id,pred\n1,0\n")
        with self.assertRaisesRegex(ValueError, "表头"):
            evaluation.recompute_accuracy(self.path, self.manifest)

    def test_manifest_duplicate_and_invalid_label_rejected(self):
        for samples in [[{"sample_id": "x", "label": 0}] * 2, [{"sample_id": "x", "label": True}], []]:
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                evaluation.recompute_accuracy(self.path, {"samples": samples})

    def seed_results(self):
        base = {"method": "ft", "experiment_kind": "synthetic", "protocol_sha256": "p", "validation_selection_sha256": "v"}
        return [dict(base, seed=s, accuracy_percent=v) for s, v in [(42, 80), (43, 82), (44, 84)]]

    def test_three_seeds_use_sample_sd(self):
        result = evaluation.summarize_seed_results(self.seed_results())
        self.assertEqual((result["mean_percent"], result["sample_sd_pp"]), (82, 2))

    def test_one_seed_incomplete_and_no_zero_sd(self):
        result = evaluation.summarize_seed_results(self.seed_results()[:1])
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["sample_sd_pp"])

    def test_duplicate_seed_rejected(self):
        results = self.seed_results()
        results[1]["seed"] = 42
        with self.assertRaisesRegex(ValueError, "同一个种子"):
            evaluation.summarize_seed_results(results)

    def test_mixed_protocol_rejected(self):
        results = self.seed_results()
        results[1]["protocol_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "protocol"):
            evaluation.summarize_seed_results(results)

    def test_nonfinite_and_out_of_range_rejected(self):
        for value in [float("nan"), float("inf"), -1, 101]:
            results = self.seed_results()
            results[0]["accuracy_percent"] = value
            with self.subTest(value=str(value)), self.assertRaises(ValueError):
                evaluation.summarize_seed_results(results)

    def test_empty_seed_results_rejected(self):
        with self.assertRaises(ValueError):
            evaluation.summarize_seed_results([])


class ExistingToolCLIRealTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def handoff(self, packet):
        output = self.directory / "audit.json"
        result = subprocess.run([sys.executable, "tools/check_handoff_mac.py", "--handoff", str(packet), "--output", str(output)],
                                cwd=ROOT, capture_output=True, text=True, timeout=60)
        return result, json.loads(output.read_text()) if output.exists() else None

    def test_actual_handoff_passes_hashes_and_reports_missing_sources(self):
        result, record = self.handoff(PACKET)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(record["packet_entries"], 43)
        self.assertTrue(record["packet_all_passed"])
        self.assertFalse(record["integration_complete"])

    def test_real_file_corruption_returns_nonzero(self):
        packet = self.directory / "packet"
        shutil.copytree(PACKET, packet)
        (packet / "03_本人复核清单.md").write_text("corrupted fixture")
        result, record = self.handoff(packet)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(record["packet_all_passed"])

    def test_manifest_path_escape_is_rejected(self):
        packet = self.directory / "packet"
        packet.mkdir()
        (packet / "交付文件校验.json").write_text(json.dumps({"../outside.txt": "0" * 64}))
        result, record = self.handoff(packet)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsafe manifest path", result.stderr)
        self.assertIsNone(record)

    def test_missing_handoff_is_rejected(self):
        result, _ = self.handoff(self.directory / "absent")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FileNotFoundError", result.stderr)

    def test_historical_recount_failure_returns_nonzero(self):
        """Regression: a failed recount must produce a failing process exit code."""
        packet = self.directory / "packet"
        shutil.copytree(PACKET, packet)
        relative = "02_助手执行结果/合成LoRA自检/metrics.json"
        metrics = json.loads((packet / relative).read_text())
        metrics["accuracy"] = 0.9
        (packet / relative).write_text(json.dumps(metrics))
        manifest_path = packet / "交付文件校验.json"
        manifest = json.loads(manifest_path.read_text())
        manifest[relative] = hashlib.sha256((packet / relative).read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        result, record = self.handoff(packet)
        self.assertFalse(record["historical_prediction_recount"]["passed"])
        self.assertEqual(result.returncode, 1)

    def rewrite_hash(self, packet, relative):
        path = packet / "交付文件校验.json"
        manifest = json.loads(path.read_text())
        manifest[relative] = hashlib.sha256((packet / relative).read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest))

    def test_empty_predictions_report_failure_without_division_error(self):
        packet = self.directory / "packet"
        shutil.copytree(PACKET, packet)
        relative = "02_助手执行结果/合成LoRA自检/predictions.csv"
        path = packet / relative
        path.write_text(path.read_text().splitlines()[0] + "\n")
        self.rewrite_hash(packet, relative)
        result, record = self.handoff(packet)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(record["historical_prediction_recount"]["passed"])
        self.assertIsNone(record["historical_prediction_recount"]["accuracy"])

    def test_nonbinary_prediction_rejected_even_if_accuracy_unchanged(self):
        packet = self.directory / "packet"
        shutil.copytree(PACKET, packet)
        relative = "02_助手执行结果/合成LoRA自检/predictions.csv"
        path = packet / relative
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        wrong = next(row for row in rows if row["gold_label"] != row["predicted_label"])
        wrong["predicted_label"] = "3"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        self.rewrite_hash(packet, relative)
        result, record = self.handoff(packet)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(record["historical_prediction_recount"]["passed"])

    def test_week01_audit_on_actual_official_sources(self):
        for part in ("data", "lora", "statistics"):
            with self.subTest(part=part):
                output = self.directory / (part + ".json")
                result = subprocess.run([sys.executable, "tools/week01_readonly_audit.py", "--repo", str(OFFICIAL),
                    "--part", part, "--actor", "AI assistant continuation check; not student independent work", "--output", str(output)],
                    cwd=ROOT, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                record = json.loads(output.read_text())
                self.assertFalse(record["是否训练模型"])
                self.assertEqual(len(record["源码SHA256"]), 4)

    def test_mac_shell_entry_supports_help(self):
        result = subprocess.run(["sh", "tools/run_week02_mac.sh", "--help"], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output", result.stdout)

    def test_required_environment_output_argument(self):
        result = subprocess.run([sys.executable, "tools/check_environment.py"], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--output", result.stderr)

    def test_mac_shell_fails_when_project_interpreter_missing(self):
        tools = self.directory / "empty-project/tools"
        tools.mkdir(parents=True)
        script = tools / "run_week02_mac.sh"
        shutil.copyfile(ROOT / "tools/run_week02_mac.sh", script)
        result = subprocess.run(["sh", str(script)], cwd=self.directory, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1)
        self.assertIn("独立环境", result.stderr)
