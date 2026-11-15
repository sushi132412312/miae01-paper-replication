"""Verify the supplied handoff, exact source hashes, and missing integration files."""
import argparse
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.mac_audit_support import PROJECT_ROOT, execution_context, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    packet = args.handoff.resolve()
    manifest = json.loads((packet / "交付文件校验.json").read_text(encoding="utf-8"))
    entries = []
    for relative, expected in manifest.items():
        path = (packet / relative).resolve()
        if not path.is_relative_to(packet):
            raise ValueError("Unsafe manifest path")
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        entries.append({"path": relative, "expected_sha256": expected, "actual_sha256": actual,
                        "passed": actual == expected})
    provenance = json.loads((PROJECT_ROOT / "third_party/loralib_source.json").read_text())
    official = []
    for relative, expected in provenance["files"].items():
        path = PROJECT_ROOT / "third_party" / ("LICENSE.loralib.md" if relative == "LICENSE.md" else relative)
        raw = path.read_bytes()
        sha256 = hashlib.sha256(raw).hexdigest()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        official.append({"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256,
                         "git_blob_sha1": blob, "passed": sha256 == expected["sha256"] and blob == expected["git_blob_sha1"]})
    missing_candidates = ["src/common.py", "src/data.py", "src/train.py", "src/evaluation.py",
        "src/__init__.py", "configs/selfcheck.json", "configs/pilot.pending.json", "tests/test_data_metrics.py",
        "tools/prepare_data.py", "tools/recompute_metrics.py", "tools/run_checks.py", "docs/week02/README_运行指南_中文.md"]
    missing = [name for name in missing_candidates if not (PROJECT_ROOT / name).is_file()]
    original_test_file = PROJECT_ROOT / "tests/test_models_pipeline.py"
    if not original_test_file.is_file():
        original_test_file = PROJECT_ROOT / "tests/integration_pending/test_models_pipeline.py"
    tree = ast.parse(original_test_file.read_text())
    tests = [f"{c.name}.{n.name}" for c in tree.body if isinstance(c, ast.ClassDef)
             for n in c.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    code_source = packet / "01_代码待本人审阅/src/models.py"
    json_files = list(packet.rglob("*.json"))
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8-sig"))
    historical = packet / "02_助手执行结果/合成LoRA自检"
    metrics = json.loads((historical / "metrics.json").read_text())
    validation = json.loads((historical / "validation_manifest.json").read_text())
    expected_samples = {row["sample_id"]: row["label"] for row in validation["samples"]}
    with (historical / "predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    correct = sum(row["gold_label"] == row["predicted_label"] for row in rows)
    csv_ok = (bool(rows) and len(validation["samples"]) == len(expected_samples)
              and all(type(label) is int and label in (0, 1) for label in expected_samples.values())
              and all(row["gold_label"] in ("0", "1") and row["predicted_label"] in ("0", "1") for row in rows)
              and len(rows) == len(expected_samples) == len({row["sample_id"] for row in rows})
              and all(expected_samples.get(row["sample_id"]) == int(row["gold_label"]) for row in rows)
              and all(all(str(metrics[key]) == row[key] for key in ("run_id", "method", "seed", "checkpoint")) for row in rows)
              and correct == metrics["correct"] and correct / len(rows) == metrics["accuracy"])
    result = {"packet_entries": len(entries), "packet_all_passed": all(e["passed"] for e in entries),
        "packet_files": entries, "official_source_files": official,
        "official_reference_commit": "c4593f060e6a368d7bb5af5273b8e42810cdef90",
        "official_blob_comparison_basis": "bundled metadata previously crosschecked with GitHub; no network check in this invocation",
        "models_py_unchanged_from_packet": (PROJECT_ROOT / "src/models.py").read_bytes() == code_source.read_bytes(),
        "packet_json_files_parsed": len(json_files),
        "historical_prediction_recount": {"passed": csv_ok, "count": len(rows), "correct": correct,
            "accuracy": correct / len(rows) if rows else None, "new_mac_training_result": False},
        "available_original_module_test_count": len(tests), "original_module_tests": tests,
        "expected_shared_test_count_from_packet": 26, "missing_integration_files": missing,
        "integration_complete": not missing, "execution_context": execution_context()}
    save_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("packet_entries", "packet_all_passed", "models_py_unchanged_from_packet", "integration_complete", "missing_integration_files")}, ensure_ascii=False, indent=2))
    return 0 if result["packet_all_passed"] and all(e["passed"] for e in official) and csv_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
