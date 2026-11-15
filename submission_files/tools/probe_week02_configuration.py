"""Real CPU characterization of the directly imported model's input boundaries.

No real-data branch is called. Accepted unsafe cases are findings, not validation passes.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_lora_mac import CONFIG, load_models
from tools.mac_audit_support import execution_context, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    models = load_models()
    cases = [
        ("rank_1", "lora", dict(CONFIG, lora_r=1)),
        ("rank_4", "lora", dict(CONFIG)),
        ("rank_8", "lora", dict(CONFIG, lora_r=8)),
        ("rank_0", "lora", dict(CONFIG, lora_r=0)),
        ("rank_negative", "lora", dict(CONFIG, lora_r=-1)),
        ("rank_fractional", "lora", dict(CONFIG, lora_r=2.5)),
        ("missing_rank", "lora", {k: v for k, v in CONFIG.items() if k != "lora_r"}),
        ("alpha_zero", "lora", dict(CONFIG, lora_alpha=0)),
        ("alpha_nan", "lora", dict(CONFIG, lora_alpha=float("nan"))),
        ("alpha_inf", "lora", dict(CONFIG, lora_alpha=float("inf"))),
        ("invalid_method", "unknown", dict(CONFIG)),
        ("wrong_synthetic_status", "lora", dict(CONFIG, status="pending")),
    ]
    results = []
    for name, method, config in cases:
        row = {"case": name, "input": {k: repr(v) for k, v in config.items()}}
        try:
            model = models.build_model(method, 42, config, synthetic=True).eval()
            with torch.no_grad():
                logits = model(torch.tensor([[0, 4, 5, 2]])).logits
            report = models.parameter_report(model)
            row.update(accepted=True, logits_finite=bool(torch.isfinite(logits).all()),
                       lora_parameters=report["lora_parameters"], trainable_parameters=report["trainable_parameters"])
        except Exception as error:
            row.update(accepted=False, exception_type=type(error).__name__, message=str(error))
        results.append(row)
    save_json(args.output, {"category": "real_model_boundary_characterization_direct_import", "helper_substitution": False,
        "cases": results, "case_count": len(results), "not_a_pass_count": True,
        "probe_mutated_source": False, "execution_context": execution_context()})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
