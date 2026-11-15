"""MOCK ONLY: simulated subprocess/hardware branches; not integration or device proof."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import check_environment, run_week02_mac


class MockDriverTests(unittest.TestCase):
    def execute(self, effects, files_present=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "records"
            if files_present:
                for name in ("src/common.py", "src/data.py", "src/evaluation.py", "src/train.py", "configs/selfcheck.json", "tests/test_data_metrics.py"):
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("# EMPTY MOCK FIXTURE, NOT PRODUCTION SOURCE\n")
                manifest = run_week02_mac.PROJECT_ROOT / "tests/week02_expected_original.json"
                (root / "tests/week02_expected_original.json").write_bytes(manifest.read_bytes())
            with patch.object(run_week02_mac, "PROJECT_ROOT", root), \
                 patch.object(run_week02_mac, "execution_context", return_value={"executor": "MOCK", "not_actual_subprocess_execution": True}), \
                 patch.object(run_week02_mac.subprocess, "run", side_effect=effects), \
                 patch.object(sys, "argv", ["mock-driver", "--output", str(output)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = run_week02_mac.main()
            summary = json.loads((output / "run_summary.json").read_text())
            logs = {p.name: p.read_text() for p in output.glob("*.txt")}
            return code, summary, logs

    def ok(self):
        return subprocess.CompletedProcess([], 0, "SIMULATED stdout\n", "")

    def test_mock_success_codes_still_block_when_files_absent(self):
        code, summary, _ = self.execute([self.ok() for _ in range(4)])
        self.assertEqual(code, 1)
        self.assertFalse(summary["overall_requested_task_complete"])

    def test_mock_nonzero_subprocess_exit_is_kept(self):
        code, summary, _ = self.execute([subprocess.CompletedProcess([], 7, "", "SIMULATED failure")] + [self.ok() for _ in range(3)], files_present=True)
        self.assertEqual(code, 1)
        self.assertEqual(summary["runs"][0]["exit_code"], 7)

    def test_mock_timeout_bytes_logged_and_other_jobs_continue(self):
        timeout = subprocess.TimeoutExpired(["SIMULATED"], 300, output=b"partial", stderr=b"error")
        code, summary, logs = self.execute([timeout] + [self.ok() for _ in range(3)])
        self.assertEqual(code, 1)
        self.assertEqual(summary["runs"][0]["exit_code"], 124)
        self.assertEqual(len(summary["runs"]), 4)
        self.assertIn("partialerror", logs["environment.txt"])

    def test_mock_timeout_text_is_logged(self):
        timeout = subprocess.TimeoutExpired(["SIMULATED"], 300, output="partial text", stderr="error text")
        _, _, logs = self.execute([timeout] + [self.ok() for _ in range(3)])
        self.assertIn("partial texterror text", logs["environment.txt"])

    def test_mock_zero_exit_cannot_bypass_test_id_verification(self):
        code, summary, _ = self.execute([self.ok() for _ in range(4)], files_present=True)
        self.assertEqual(code, 1)
        self.assertFalse(summary["original_shared_test_ids_verified"])
        self.assertFalse(summary["overall_requested_task_complete"])

    def original_logs(self, duplicate=False):
        manifest = run_week02_mac.PROJECT_ROOT / "tests/week02_expected_original.json"
        ids = json.loads(manifest.read_text())["test_ids"]
        module = [name for name in ids if name.startswith("test_models_pipeline.")]
        data = [name for name in ids if name.startswith("test_data_metrics.")]
        if duplicate:
            data[-1] = data[0]
        logs = []
        for names in (module, data):
            text = "\n".join(f"SIMULATED ({name}) ... ok" for name in names)
            logs.append(subprocess.CompletedProcess([], 0, "", text))
        return [self.ok(), self.ok()] + logs

    def test_mock_complete_unique_id_set_validates_gate(self):
        code, summary, _ = self.execute(self.original_logs(), files_present=True)
        self.assertEqual(code, 0)
        self.assertTrue(summary["original_shared_test_ids_verified"])

    def test_mock_duplicate_id_cannot_bypass_gate(self):
        code, summary, _ = self.execute(self.original_logs(duplicate=True), files_present=True)
        self.assertEqual(code, 1)
        self.assertFalse(summary["original_shared_test_ids_verified"])


class MockEnvironmentTests(unittest.TestCase):
    def collect(self, pip_code=0, mps_available=False, operation_error=False):
        import torch
        real_ones = torch.ones
        def ones(*args, **kwargs):
            if kwargs.get("device") == "mps" and operation_error:
                raise RuntimeError("SIMULATED MPS operation failure")
            return real_ones(*args, **kwargs)
        def process(command, **kwargs):
            if command[0] == "nvidia-smi":
                raise FileNotFoundError("SIMULATED no nvidia-smi")
            return subprocess.CompletedProcess(command, pip_code, "SIMULATED pip result", "")
        with patch.object(check_environment.subprocess, "run", side_effect=process), \
             patch.object(check_environment.platform, "system", return_value="Linux"), \
             patch.object(check_environment.platform, "platform", return_value="MOCK-platform-not-hardware-evidence"), \
             patch.object(check_environment.platform, "processor", return_value="MOCK-processor"), \
             patch.object(check_environment, "execution_context", return_value={"executor": "MOCK"}), \
             patch.object(torch.backends.mps, "is_built", return_value=True), \
             patch.object(torch.backends.mps, "is_available", return_value=mps_available), \
             patch.object(torch, "ones", side_effect=ones):
            return check_environment.collect()

    def test_mock_mps_unavailable_does_not_claim_operation(self):
        result = self.collect()
        self.assertFalse(result["mps"]["operation_passed"])
        self.assertEqual(result["validated_devices"], ["cpu"])

    def test_mock_pip_failure_raises(self):
        with self.assertRaises(RuntimeError):
            self.collect(pip_code=1)

    def test_mock_mps_error_is_recorded(self):
        result = self.collect(mps_available=True, operation_error=True)
        self.assertFalse(result["mps"]["operation_passed"])
        self.assertIn("SIMULATED MPS", result["mps"]["error"])
