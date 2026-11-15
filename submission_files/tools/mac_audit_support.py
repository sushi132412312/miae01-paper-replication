"""Evidence helpers; model seed/hash are shared with the owned model module."""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from src.model_support import seed_everything, tensor_hash


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def execution_context():
    files = sorted(set(PROJECT_ROOT.glob("src/*.py")) | set(PROJECT_ROOT.glob("tools/*mac*.py")))
    try:
        query = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True, timeout=10)
        commit = query.stdout.strip() if query.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        commit = None
    return {
        "executor": "AI assistant, executed with user authorization",
        "student_independent_work": False,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python_executable": sys.executable,
        "git_head_at_execution": commit,
        "git_context_available": commit is not None,
        "source_sha256": {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "real_roberta_base_training": False, "sst2_downloaded": False,
    }
