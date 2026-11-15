"""Run Mac checks and record genuine failures; never report missing suites as passed."""
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.mac_audit_support import PROJECT_ROOT, execution_context, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "records/week02/mac_20261115")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MPLCONFIGDIR=str(PROJECT_ROOT / ".cache/matplotlib"))
    original_model_tests = "tests" if (PROJECT_ROOT / "tests/test_models_pipeline.py").is_file() else "tests/integration_pending"
    jobs = [
        ("environment", [sys.executable, "tools/check_environment.py", "--output", str(output / "environment.json")]),
        ("supplementary_models", [sys.executable, "tools/check_lora_mac.py", "--output", str(output)]),
        ("original_module_tests", [sys.executable, "-m", "unittest", "discover", "-s", original_model_tests, "-p", "test_models_pipeline.py", "-v"]),
        ("original_data_tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_data_metrics.py", "-v"]),
    ]
    runs = []
    for name, command in jobs:
        print(f"Running {name}", flush=True)
        started = datetime.now(timezone.utc).isoformat()
        try:
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=300)
            log = result.stdout + result.stderr
            code = result.returncode
        except subprocess.TimeoutExpired as error:
            def decode(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
            log = decode(error.stdout) + decode(error.stderr) + "\nTIMEOUT after 300 seconds\n"
            code = 124
        path = output / (name + ".txt")
        path.write_text(log, encoding="utf-8")
        runs.append({"name": name, "command": command, "started_at_utc": started,
                     "finished_at_utc": datetime.now(timezone.utc).isoformat(), "exit_code": code,
                     "log_file": path.name})
        print(f"{name}: exit {code}", flush=True)
    missing = [name for name in ("src/common.py", "src/data.py", "src/evaluation.py", "src/train.py",
                                "configs/selfcheck.json", "tests/test_data_metrics.py")
               if not (PROJECT_ROOT / name).is_file()]
    manifest = PROJECT_ROOT / "tests/week02_expected_original.json"
    expected = json.loads(manifest.read_text())["test_ids"] if manifest.is_file() else []
    actual = []
    for name in ("original_module_tests", "original_data_tests"):
        actual.extend(re.findall(r"^\S+ \(([^()]+)\) \.\.\. ok$", (output / (name + ".txt")).read_text(), re.M))
    verified = len(expected) == len(set(expected)) == len(actual) == len(set(actual)) == 26 and set(actual) == set(expected)
    result = {"scope": "original_shared_integration_checks", "runs": runs, "original_module_expected_tests": 8, "shared_expected_tests": 26,
        "original_suites_status": "blocked_missing_integrated_project" if missing else ("verified_26" if verified else "test_ids_not_verified"),
        "original_shared_test_ids_verified": verified, "actual_passed_test_ids": actual,
        "missing_integration_files": missing,
        "overall_requested_task_complete": not missing and verified and all(r["exit_code"] == 0 for r in runs),
        "supplementary_tests_replace_original_suites": False, "execution_context": execution_context()}
    save_json(output / "run_summary.json", result)
    print(json.dumps({k: result[k] for k in ("original_suites_status", "overall_requested_task_complete")}, ensure_ascii=False, indent=2))
    return 0 if result["overall_requested_task_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
