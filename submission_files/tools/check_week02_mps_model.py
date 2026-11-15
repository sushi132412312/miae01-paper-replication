"""Real MPS checks importing the owned tiny model directly.

Not the original production integration suite. No weights/data downloads.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_lora_mac import CONFIG, batch, load_models
from tools.mac_audit_support import execution_context, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import torch
    result = {"category": "real_mps_model_direct_import", "helper_substitution": False, "device": "mps",
              "mps_available": torch.backends.mps.is_available(), "synthetic_only": True,
              "original_suite": False, "cases": [], "execution_context": execution_context()}
    result["execution_context"]["source_sha256"][str(Path(__file__).resolve().relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if not result["mps_available"]:
        result.update(status="not_run_device_unavailable", passed_count=0)
        save_json(args.output, result)
        return 2
    models = load_models()
    cpu_inputs, cpu_labels = batch()
    inputs, labels = cpu_inputs.to("mps"), cpu_labels.to("mps")
    def check(name, function):
        try:
            details = function()
            result["cases"].append({"id": name, "status": "passed", "details": details})
        except Exception as error:
            result["cases"].append({"id": name, "status": "failed", "error": repr(error)})
    def forward():
        model = models.build_model("lora", 42, CONFIG, synthetic=True).eval().to("mps")
        with torch.no_grad():
            logits = model(inputs).logits
        torch.mps.synchronize()
        assert logits.device.type == "mps" and tuple(logits.shape) == (2, 2)
        assert bool(torch.isfinite(logits).all())
        return {"logits": logits.cpu().tolist(), "shape": list(logits.shape), "finite": True}
    def paired():
        ft = models.build_model("ft", 42, CONFIG, synthetic=True).eval().to("mps")
        lora = models.build_model("lora", 42, CONFIG, synthetic=True).eval().to("mps")
        cpu = models.build_model("lora", 42, CONFIG, synthetic=True).eval()
        with torch.no_grad():
            first, second, reference = ft(inputs).logits.cpu(), lora(inputs).logits.cpu(), cpu(cpu_inputs).logits
        torch.mps.synchronize()
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(second, reference, atol=1e-5, rtol=1e-5)
        return {"ft_lora_max_abs_diff": (first-second).abs().max().item(),
                "cpu_mps_max_abs_diff": (second-reference).abs().max().item(),
                "ft_lora_atol": 1e-6, "cpu_mps_atol": 1e-5, "rtol": 1e-5}
    def backward():
        model = models.build_model("lora", 42, CONFIG, synthetic=True).train().to("mps")
        before = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
        loss = model(inputs, labels=labels).loss
        assert bool(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            if ".lora_A" in name:
                assert parameter.grad is not None and torch.count_nonzero(parameter.grad).item() == 0
            if not parameter.requires_grad:
                assert parameter.grad is None
        optimizer.step()
        torch.mps.synchronize()
        changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p.detach().cpu())]
        assert any(".lora_B" in n for n in changed)
        assert any(n.startswith("classifier.") for n in changed)
        assert all(p.requires_grad or torch.equal(before[n], p.detach().cpu()) for n, p in model.named_parameters())
        return {"loss": loss.item(), "changed_names": changed, "frozen_weights_unchanged": True, "A_first_gradient_zero": True}
    check("mps_forward_finite", forward)
    check("mps_paired_initial_logits_and_cpu_tolerance", paired)
    check("mps_backward_optimizer_and_frozen_weights", backward)
    result["passed_count"] = sum(c["status"] == "passed" for c in result["cases"])
    result["status"] = "passed" if result["passed_count"] == 3 else "failed"
    save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
