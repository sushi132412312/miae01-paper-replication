"""Standalone module acceptance. Real imports and tensors; no mocks or downloads."""
import unittest

import torch

from src import models
from src.model_support import seed_everything, tensor_hash
from tools import check_lora_mac as core


class LoRADirectTests(core.MacModelDiagnostics):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        core.MODELS = models


class ModelContractTests(unittest.TestCase):
    def test_invalid_rank_rejected(self):
        for rank in (0, -1, True, 4.0, "4", None):
            with self.subTest(rank=rank), self.assertRaisesRegex(ValueError, "lora_r"):
                models.build_model("lora", 42, dict(core.CONFIG, lora_r=rank), synthetic=True)

    def test_invalid_alpha_rejected(self):
        for alpha in (0, -1, True, "8", None, float("nan"), float("inf"), 10**1000):
            with self.subTest(alpha_type=type(alpha).__name__), self.assertRaisesRegex(ValueError, "lora_alpha"):
                models.build_model("lora", 42, dict(core.CONFIG, lora_alpha=alpha), synthetic=True)

    def test_invalid_seed_rejected(self):
        for seed in (-1, True, "42", 2**32):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "seed"):
                models.build_model("lora", seed, core.CONFIG, synthetic=True)

    def test_config_requires_mapping(self):
        for config in (None, [], "config"):
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "config"):
                models.build_model("lora", 42, config, synthetic=True)

    def test_synthetic_flag_requires_bool(self):
        with self.assertRaisesRegex(ValueError, "synthetic"):
            models.build_model("lora", 42, core.CONFIG, synthetic="true")

    def test_rejected_request_does_not_advance_cpu_rng(self):
        before = torch.get_rng_state().clone()
        with self.assertRaises(ValueError):
            models.build_model("lora", 42, dict(core.CONFIG, lora_alpha=float("nan")), synthetic=True)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_valid_config_not_mutated(self):
        config = dict(core.CONFIG)
        models.build_model("lora", 42, config, synthetic=True)
        self.assertEqual(config, core.CONFIG)

    def test_missing_real_data_module_reports_actionable_error(self):
        with self.assertRaisesRegex(RuntimeError, "src.data"):
            models.build_model("lora", 42, core.CONFIG, synthetic=False)

    def test_tensor_hash_scalar_and_bfloat16(self):
        self.assertEqual(len(tensor_hash([("scalar", torch.tensor(1.0))])), 64)
        self.assertEqual(len(tensor_hash([("bf16", torch.ones(2, dtype=torch.bfloat16))])), 64)

    def test_tensor_hash_duplicate_name_rejected(self):
        with self.assertRaisesRegex(ValueError, "重复"):
            tensor_hash([("x", torch.ones(2)), ("x", torch.zeros(2))])

    def test_seed_upper_boundary_repeats(self):
        seed_everything(2**32 - 1)
        first = torch.rand(3)
        seed_everything(2**32 - 1)
        self.assertTrue(torch.equal(first, torch.rand(3)))

    def test_rank_changes_parameter_count(self):
        for rank in (1, 4, 8):
            with self.subTest(rank=rank):
                report = models.parameter_report(models.build_model("lora", 42, dict(core.CONFIG, lora_r=rank), synthetic=True))
                self.assertEqual(report["lora_parameters"], 2 * 2 * (32 * rank + rank * 32))
                self.assertEqual(report["trainable_parameters"], report["lora_parameters"] + 1122)
