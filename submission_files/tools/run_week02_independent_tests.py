"""Run continuation tests with per-case evidence and explicit real/mock categories."""
import argparse
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append({"id": test.id(), "status": "passed"})

    def addFailure(self, test, error):
        super().addFailure(test, error)
        self.records.append({"id": test.id(), "status": "failed", "detail": self._exc_info_to_string(error, test)})

    def addError(self, test, error):
        super().addError(test, error)
        self.records.append({"id": test.id(), "status": "error", "detail": self._exc_info_to_string(error, test)})

    def addExpectedFailure(self, test, error):
        super().addExpectedFailure(test, error)
        self.records.append({"id": test.id(), "status": "expected_failure_known_gap", "detail": self._exc_info_to_string(error, test)})

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self.records.append({"id": test.id(), "status": "unexpected_success_requires_review"})

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.records.append({"id": test.id(), "status": "skipped", "detail": reason})

    def addSubTest(self, test, subtest, error):
        super().addSubTest(test, subtest, error)
        if error:
            self.records.append({"id": subtest.id(), "status": "subtest_failure", "detail": self._exc_info_to_string(error, test)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("real", "mock"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import torch
    torch.set_num_threads(1)
    pattern = "test_real_modules.py" if args.group == "real" else "test_mock_workflow.py"
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests/week02_independent"), pattern=pattern)
    started = datetime.now(timezone.utc).isoformat()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordedResult).run(suite)
    files = sorted({*ROOT.glob("src/*.py"), *ROOT.glob("tools/*.py"), *ROOT.glob("tests/week02_independent/*.py"),
                    *ROOT.glob("third_party/loralib/*.py"), ROOT / "records/week02/recount_verification/verify_evaluation.py"})
    record = {
        "category": "real_existing_module_execution" if args.group == "real" else "mock_simulated_workflow_only",
        "no_missing_production_module_created": True, "original_26_suite": False,
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "tests_run": result.testsRun,
        "passed_count": sum(r["status"] == "passed" for r in result.records),
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "expected_failures_known_gaps": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses),
        "unittest_runner_success": result.wasSuccessful(),
        "all_desired_assertions_satisfied": result.wasSuccessful() and not result.expectedFailures,
        "test_results": result.records,
        "execution_context": {"executor": "AI assistant with user authorization", "student_independent_work": False,
            "python": sys.executable, "platform": platform.platform(),
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(),
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},
        "notes": ["Fixtures use artificial data; no pretrained weights, SST-2 or cloud training.",
            "Review evaluation functions are existing review code; identity to missing src/evaluation.py is unverified.",
            "Expected failures are unmet desired behavior, not passed tests.",
            "Mock outcomes do not attest hardware, subprocess execution or integration completeness."]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / (args.group + "_tests.json")).write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.output / (args.group + "_tests.txt")).write_text(stream.getvalue())
    print(stream.getvalue())
    print(json.dumps({k: record[k] for k in ("category", "tests_run", "passed_count", "expected_failures_known_gaps", "failures", "errors")}, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
