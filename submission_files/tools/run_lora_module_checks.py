"""Standalone environment/LoRA acceptance; no missing production dependencies."""
import argparse
import io
import json
import os
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.mac_audit_support import execution_context, save_json
from tools.run_week02_independent_tests import RecordedResult
from tools.check_lora_mac import DETAILS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "records/week02/personal_module_checks")
    parser.add_argument("--mps", action="store_true", help="Also require the three real MPS model checks")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import torch
    torch.set_num_threads(1)
    started = datetime.now(timezone.utc).isoformat()
    jobs = []
    def run(name, command):
        try:
            done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
            log, code = done.stdout + done.stderr, done.returncode
        except subprocess.TimeoutExpired:
            log, code = "TIMEOUT after 180 seconds\n", 124
        (output / (name + ".txt")).write_text(log)
        jobs.append({"name": name, "command": command, "exit_code": code})
        return code == 0
    environment_ok = run("environment", [sys.executable, "tools/check_environment.py", "--output", str(output / "environment.json")])
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_lora_module.py")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordedResult).run(suite)
    (output / "module_tests.txt").write_text(stream.getvalue())
    expected = json.loads((ROOT / "tests/week02_module_cases.json").read_text())["test_ids"]
    actual = [case["id"] for case in result.records if case["status"] == "passed"]
    ids_verified = bool(expected) and len(actual) == len(set(actual)) == len(expected) == len(set(expected)) and set(actual) == set(expected)
    strict_success = result.wasSuccessful() and not result.skipped and not result.expectedFailures and ids_verified
    mps_ok = None
    if args.mps:
        mps_ok = run("mps_model_tests", [sys.executable, "tools/check_week02_mps_model.py", "--output", str(output / "mps_model_tests.json")])
    context = execution_context()
    for relative in ("tools/run_lora_module_checks.py", "tests/test_lora_module.py", "tests/week02_module_cases.json",
                     "third_party/loralib/layers.py", "third_party/loralib/utils.py"):
        import hashlib
        context["source_sha256"][relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    record = {"scope": "standalone_environment_lora_module", "direct_production_import": "src.models",
        "mock_used": False, "helper_substitution": False, "synthetic_only": True,
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": jobs, "tests_run": result.testsRun, "passed_count": len(actual),
        "expected_test_count": len(expected), "test_ids_verified": ids_verified,
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "expected_failures": len(result.expectedFailures), "test_results": result.records,
        "environment_passed": environment_ok, "mps_required": args.mps, "mps_passed": mps_ok,
        "module_acceptance_passed": environment_ok and strict_success and (mps_ok if args.mps else True),
        "original_shared_26_executed": False, "original_shared_suite_status": "pending_full_integrated_source",
        "model_details": DETAILS, "execution_context": context}
    save_json(output / "module_summary.json", record)
    print(stream.getvalue())
    print(json.dumps({k: record[k] for k in ("scope", "tests_run", "passed_count", "test_ids_verified", "module_acceptance_passed", "original_shared_suite_status")}, indent=2))
    return 0 if record["module_acceptance_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
