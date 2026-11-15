"""Eight real checks importing the standalone LoRA module directly.

No missing-module substitution, pretrained model, dataset or network is used.
These checks remain separate from the original 26 shared integration tests.
"""
import argparse
import io
import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import mac_audit_support as support


def load_models():
    from src import models
    return models


DETAILS = {}
CONFIG = {"status": "synthetic_selfcheck", "lora_r": 4, "lora_alpha": 8}
MODELS = None


def batch():
    import torch
    # Artificial tokens (no padding); identical inputs for all paired runs.
    return torch.tensor([[0, 4, 5, 6, 7, 2], [0, 8, 9, 10, 11, 2]]), torch.tensor([0, 1])


class MacModelDiagnostics(unittest.TestCase):
    def test_01_qv_injection_and_key(self):
        from third_party.loralib import Linear
        model = MODELS.build_model("lora", 42, CONFIG, synthetic=True)
        for layer in model.roberta.encoder.layer:
            self.assertIsInstance(layer.attention.self.query, Linear)
            self.assertIsInstance(layer.attention.self.value, Linear)
            self.assertNotIsInstance(layer.attention.self.key, Linear)
            self.assertFalse(layer.attention.self.query.merge_weights)
        DETAILS["qv_module_names"] = MODELS.parameter_report(model)["lora_module_names"]

    def test_02_initialization(self):
        import torch
        model = MODELS.build_model("lora", 42, CONFIG, synthetic=True)
        checks = []
        for name, module in model.named_modules():
            if not hasattr(module, "lora_A"):
                continue
            bound = 1 / math.sqrt(module.in_features)
            self.assertGreater(torch.count_nonzero(module.lora_A).item(), 0)
            self.assertLessEqual(module.lora_A.abs().max().item(), bound)
            self.assertEqual(torch.count_nonzero(module.lora_B).item(), 0)
            self.assertEqual(torch.count_nonzero(module.lora_B @ module.lora_A).item(), 0)
            self.assertEqual(module.scaling, 2)
            checks.append({"module": name, "A_abs_max": module.lora_A.abs().max().item(),
                           "kaiming_uniform_bound": bound, "B_nonzero": 0, "initial_BA_nonzero": 0})
        DETAILS["initialization"] = checks

    def test_03_parameter_freezing_and_head(self):
        model = MODELS.build_model("lora", 42, CONFIG, synthetic=True)
        report = MODELS.parameter_report(model)
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, ".lora_" in name or name.startswith("classifier."))
        self.assertEqual(report["lora_parameters"], 2 * 2 * (32 * 4 + 4 * 32))
        self.assertEqual(report["classifier_parameters"], (32 * 32 + 32) + (32 * 2 + 2))
        self.assertEqual(report["trainable_parameters"], 2146)
        self.assertEqual(report["frozen_parameters"], 22368)
        self.assertEqual(report["total_parameters"], 24514)
        DETAILS["lora_parameter_report"] = report

    def test_04_ft_all_trainable(self):
        report = MODELS.parameter_report(MODELS.build_model("ft", 42, CONFIG, synthetic=True))
        self.assertEqual(report["trainable_parameters"], report["total_parameters"])
        self.assertEqual(report["total_parameters"], 23490)
        self.assertEqual(report["lora_parameters"], 0)
        self.assertEqual(report["frozen_parameters"], 0)
        DETAILS["ft_parameter_report"] = report

    def test_05_shared_weights_rng_and_logits(self):
        import torch
        ft = MODELS.build_model("ft", 42, CONFIG, synthetic=True).eval()
        state = torch.get_rng_state().clone()
        lora = MODELS.build_model("lora", 42, CONFIG, synthetic=True).eval()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        ft_dict = dict(ft.named_parameters())
        for name, parameter in lora.named_parameters():
            if ".lora_" not in name:
                self.assertTrue(torch.equal(parameter, ft_dict[name]), name)
        inputs, _ = batch()
        with torch.no_grad():
            ft_logits, lora_logits = ft(inputs).logits, lora(inputs).logits
        self.assertTrue(torch.allclose(ft_logits, lora_logits, atol=1e-7, rtol=0))
        DETAILS["initial_logits_max_abs_difference"] = (ft_logits - lora_logits).abs().max().item()
        DETAILS["paired_shared_parameters_equal"] = True
        DETAILS["cpu_rng_state_preserved_by_injection"] = True

    def test_06_optimizer_and_frozen_weights(self):
        import torch
        inputs, labels = batch()
        for method in ("ft", "lora"):
            model = MODELS.build_model(method, 42, CONFIG, synthetic=True).train()
            before = {name: p.detach().clone() for name, p in model.named_parameters()}
            initial = MODELS.parameter_report(model)
            optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
            loss = model(inputs, labels=labels).loss
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            if method == "lora":
                for name, parameter in model.named_parameters():
                    if ".lora_A" in name:
                        self.assertIsNotNone(parameter.grad)
                        self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)
                    if not parameter.requires_grad:
                        self.assertIsNone(parameter.grad)
            optimizer.step()
            changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p.detach())]
            final = MODELS.parameter_report(model)
            if method == "lora":
                self.assertTrue(any(".lora_B" in n for n in changed))
                self.assertTrue(any(n.startswith("classifier.") for n in changed))
                self.assertTrue(all(p.requires_grad or torch.equal(before[n], p.detach()) for n, p in model.named_parameters()))
                self.assertEqual(initial["frozen_parameters_sha256"], final["frozen_parameters_sha256"])
            else:
                self.assertTrue(any(n.startswith("roberta.") for n in changed))
            DETAILS[method + "_optimizer_step"] = {"loss": loss.item(), "changed_names": changed,
                "initial_frozen_sha256": initial["frozen_parameters_sha256"],
                "final_frozen_sha256": final["frozen_parameters_sha256"],
                "frozen_weights_unchanged": initial["frozen_parameters_sha256"] == final["frozen_parameters_sha256"]}

    def test_07_reproducible_two_step_synthetic(self):
        import torch
        def execute():
            model = MODELS.build_model("lora", 42, CONFIG, synthetic=True).train()
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.001, weight_decay=0)
            inputs, labels = batch()
            losses = []
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss = model(inputs, labels=labels).loss
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            model.eval()
            with torch.no_grad():
                predictions = model(inputs).logits.argmax(-1).tolist()
            return {"losses": losses, "predictions": predictions,
                    "model_sha256": support.tensor_hash(model.named_parameters())}
        first, second = execute(), execute()
        self.assertEqual(first, second)
        DETAILS["two_step_synthetic_repeat"] = first

    def test_08_train_eval_no_merge_and_config_guards(self):
        import torch
        model = MODELS.build_model("lora", 42, CONFIG, synthetic=True)
        for module in model.modules():
            if hasattr(module, "lora_B"):
                with torch.no_grad():
                    module.lora_B.fill_(0.1)
        before = support.tensor_hash(model.named_parameters())
        model.eval()
        model.train()
        model.eval()
        self.assertEqual(before, support.tensor_hash(model.named_parameters()))
        with self.assertRaises(ValueError):
            MODELS.build_model("invalid", 42, CONFIG, synthetic=True)
        with self.assertRaises(ValueError):
            MODELS.build_model("lora", 42, {}, synthetic=True)
        DETAILS["train_eval_does_not_merge_weights"] = True


def main():
    global MODELS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    MODELS = load_models()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MacModelDiagnostics)
    test_ids = [test.id() for test in suite]
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "supplementary_models.txt").write_text(stream.getvalue(), encoding="utf-8")
    summary = {"suite": "new_mac_supplementary_diagnostics", "original_module_suite": False,
        "missing_integrated_project_replaced": False,
        "helper_substitution": False, "direct_import": "src.models",
        "test_ids": test_ids, "tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped), "passed": result.wasSuccessful(),
        "device": "cpu", "synthetic_only": True, "details": DETAILS,
        "execution_context": support.execution_context()}
    support.save_json(args.output / "supplementary_models.json", summary)
    print(stream.getvalue())
    print(json.dumps({k: summary[k] for k in ("suite", "tests_run", "failures", "errors", "passed")}, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
