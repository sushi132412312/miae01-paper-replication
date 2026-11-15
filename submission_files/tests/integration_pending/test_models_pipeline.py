"""CPU 小 RoBERTa 结构检查，始终不下载权重或 SST-2。"""
import tempfile
import unittest
from pathlib import Path

import torch

from src.common import PROJECT_ROOT, read_json
from src.data import batch_order, synthetic_data
from src.evaluation import recompute_accuracy
from src.models import build_model, parameter_report
from src.train import run


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(PROJECT_ROOT / "configs/selfcheck.json")

    def test_paired_backbone_head_and_rng_state(self):
        ft = build_model("ft", 42, self.config, synthetic=True)
        rng_after_ft = torch.get_rng_state().clone()
        lora = build_model("lora", 42, self.config, synthetic=True)
        ft_report, lora_report = parameter_report(ft), parameter_report(lora)
        self.assertEqual(ft_report["common_backbone_sha256"], lora_report["common_backbone_sha256"])
        self.assertEqual(ft_report["classifier_sha256"], lora_report["classifier_sha256"])
        self.assertTrue(torch.equal(rng_after_ft, torch.get_rng_state()))

    def test_zero_update_keeps_initial_eval_logits(self):
        ft = build_model("ft", 42, self.config, synthetic=True).eval()
        lora = build_model("lora", 42, self.config, synthetic=True).eval()
        records, _ = synthetic_data(self.config)
        inputs = torch.tensor([r["input_ids"] for r in records["validation"][:2]])
        with torch.no_grad():
            self.assertTrue(torch.allclose(ft(inputs).logits, lora(inputs).logits, atol=1e-7, rtol=0))

    def test_only_qv_lora_and_classifier_trainable(self):
        model = build_model("lora", 42, self.config, synthetic=True)
        report = parameter_report(model)
        self.assertEqual(len(report["lora_module_names"]), 4)
        self.assertTrue(all(name.endswith((".query", ".value")) for name in report["lora_module_names"]))
        self.assertTrue(all(".lora_" in name or name.startswith("classifier.")
                            for name in report["trainable_parameter_names"]))
        self.assertEqual(report["lora_parameters"], 2 * 2 * (32 * 4 + 4 * 32))
        self.assertEqual(report["trainable_parameters"], report["lora_parameters"] + report["classifier_parameters"])

    def test_ft_all_parameters_trainable(self):
        report = parameter_report(build_model("ft", 42, self.config, synthetic=True))
        self.assertEqual(report["total_parameters"], report["trainable_parameters"])
        self.assertEqual(report["lora_parameters"], 0)

    def test_one_step_freezes_backbone_and_can_update_lora(self):
        model = build_model("lora", 42, self.config, synthetic=True)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        records, _ = synthetic_data(self.config)
        inputs = torch.tensor([r["input_ids"] for r in records["train"][:2]])
        labels = torch.tensor([r["label"] for r in records["train"][:2]])
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
        model.train()
        model(inputs, labels=labels).loss.backward()
        self.assertTrue(all(torch.count_nonzero(p.grad) == 0 for name, p in model.named_parameters() if ".lora_A" in name))
        optimizer.step()
        changed = [name for name, p in model.named_parameters() if not torch.equal(before[name], p.detach())]
        self.assertTrue(any(".lora_B" in name for name in changed))
        self.assertTrue(any(name.startswith("classifier.") for name in changed))
        self.assertTrue(all(p.requires_grad or torch.equal(before[name], p.detach()) for name, p in model.named_parameters()))

    def test_training_seed_changes_order_not_data(self):
        first_data, first_manifest = synthetic_data(self.config)
        other = dict(self.config, train_seed=43)
        other_data, other_manifest = synthetic_data(other)
        self.assertEqual(first_data, other_data)
        self.assertEqual(first_manifest, other_manifest)
        self.assertNotEqual(batch_order(20, 42), batch_order(20, 43))
        self.assertEqual(batch_order(20, 42), batch_order(20, 42))


class PipelineTests(unittest.TestCase):
    def test_synthetic_ft_lora_export_and_recount(self):
        config = read_json(PROJECT_ROOT / "configs/selfcheck.json")
        with tempfile.TemporaryDirectory() as directory:
            for method in ("ft", "lora"):
                output = Path(directory) / method
                metrics = run(config, method, output, synthetic=True)
                counted = recompute_accuracy(output / "predictions.csv", output / "validation_manifest.json")
                self.assertEqual(metrics["accuracy"], counted["accuracy"])
                self.assertEqual(counted["count"], 6)
                self.assertFalse(metrics["paper_reproduction_result"])
                self.assertTrue(read_json(output / "parameter_check.json")["frozen_weights_unchanged"])

    def test_synthetic_run_is_reproducible(self):
        config = read_json(PROJECT_ROOT / "configs/selfcheck.json")
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first", Path(directory) / "second"
            run(config, "lora", first, synthetic=True)
            run(config, "lora", second, synthetic=True)
            self.assertEqual(read_json(first / "training_steps.json"), read_json(second / "training_steps.json"))
            self.assertEqual(read_json(first / "metrics.json")["checkpoint"], read_json(second / "metrics.json")["checkpoint"])


if __name__ == "__main__":
    unittest.main()
