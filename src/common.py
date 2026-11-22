"""通用记录与随机性设置；不在导入时联网或加载模型。"""
import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def json_hash(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def tensor_hash(tensors):
    """名称和张量字节一起校验，避免只比较参数数量。"""
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors, key=lambda item: item[0]):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def execution_context():
    def git(*args):
        result = subprocess.run(["git", "-C", str(PROJECT_ROOT), *args],
                                text=True, capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    code = {}
    for directory in ("src", "tools", "tests", "third_party", "configs"):
        for path in sorted((PROJECT_ROOT / directory).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json"):
                code[path.relative_to(PROJECT_ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    operator = os.environ.get("MIAE01_RUN_OPERATOR") or None
    return {
        "executor": operator,
        "student_independent_work": False if operator == "Codex" else None,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "git_parent_commit": git("rev-parse", "HEAD"),
        "git_worktree_status": git("status", "--porcelain=v1"),
        "executed_source_sha256": code,
    }
