"""执行已确认的FT/LoRA工程试跑并验收配对条件；不生成正式论文结论。"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import read_json, save_json
from src.data import validate_real_config
from src.evaluation import recompute_accuracy, summarize_seed_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hf-endpoint", default="https://huggingface.co")
    parser.add_argument("--run-tag", default="pilot")
    args = parser.parse_args()
    os.chdir(ROOT)
    config = read_json(args.config)
    validate_real_config(config)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = read_json(output / "environment_gpu.json")
    checks = read_json(output / "code_checks/test_summary.json")
    if not environment["gpu_test"]["matrix_and_backward_passed"] or not checks["passed"]:
        raise RuntimeError("环境或代码验收尚未通过")
    env = dict(os.environ, HF_ENDPOINT=args.hf_endpoint, HF_HUB_DISABLE_XET="1",
               HF_HUB_ETAG_TIMEOUT="15", HF_HUB_DOWNLOAD_TIMEOUT="60",
               HF_HOME=str(ROOT / "cache/hf_home"), CUBLAS_WORKSPACE_CONFIG=":4096:8",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("HF_DATASETS_OFFLINE", None)
    status_path = output / "runner_status.json"
    def status(stage, **values):
        save_json(status_path, {"stage": stage, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                                "hf_endpoint": args.hf_endpoint, "executor": "AI assistant", **values})
    metrics, folders = {}, {}
    try:
        for method in ("ft", "lora"):
            folder = output / f"{args.run_tag}_{method}"
            if folder.exists():
                raise FileExistsError(f"不覆盖既有运行：{folder}")
            folders[method] = folder
            status("running", method=method, output=str(folder))
            with (output / f"{args.run_tag}_{method}.stdout.txt").open("w", encoding="utf-8") as stdout, \
                 (output / f"{args.run_tag}_{method}.stderr.txt").open("w", encoding="utf-8") as stderr:
                process = subprocess.run([sys.executable, "-m", "src.train", "--config", str(args.config.resolve()),
                                          "--method", method, "--output", str(folder), "--device", "cuda"],
                                         env=env, stdout=stdout, stderr=stderr, timeout=900)
            if process.returncode:
                raise RuntimeError(f"{method}运行失败，退出码{process.returncode}；查看对应stderr记录")
            metrics[method] = read_json(folder / "metrics.json")
            recounted = recompute_accuracy(folder / "predictions.csv", folder / "validation_manifest.json")
            if recounted["accuracy"] != metrics[method]["accuracy"] or recounted["count"] != 64:
                raise AssertionError("预测复算不一致或样本数不是已确认的64")
            save_json(folder / "independent_recount.json", recounted)
        first, second = folders["ft"], folders["lora"]
        ft_init = read_json(first / "model_initialization.json")
        lora_init = read_json(second / "model_initialization.json")
        paired = {
            "same_training_samples": read_json(first / "train_manifest.json") == read_json(second / "train_manifest.json"),
            "same_validation_samples": read_json(first / "validation_manifest.json") == read_json(second / "validation_manifest.json"),
            "same_common_backbone": ft_init["common_backbone_sha256"] == lora_init["common_backbone_sha256"],
            "same_classifier_initialization": ft_init["classifier_sha256"] == lora_init["classifier_sha256"],
            "same_update_count": metrics["ft"]["steps"] == metrics["lora"]["steps"] == 10,
            "same_training_order": [r["sample_ids"] for r in read_json(first / "training_steps.json")] ==
                                   [r["sample_ids"] for r in read_json(second / "training_steps.json")],
            "lora_initial_B_zero": lora_init["lora_B_all_zero"],
            "lora_frozen_weights_unchanged": read_json(second / "parameter_check.json")["frozen_weights_unchanged"],
        }
        if not all(paired.values()):
            raise AssertionError("配对条件核验失败：" + json.dumps(paired))
        summary = {"status": "engineering_pilot_complete", "paper_reproduction_result": False,
                   "seed_count": 1, "seed": 42, "paired_checks": paired, "metrics": metrics,
                   "formal_summary": {method: summarize_seed_results([value]) for method, value in metrics.items()},
                   "note": "仅固定64/64样本、每种方法10步的真实流程试跑；不能用作三种子论文结论。"}
        save_json(output / "pilot_pair_summary.json", summary)
        hashes = {}
        cache = ROOT / "cache"
        for path in cache.rglob("*"):
            if path.is_file() and not path.is_symlink() and not path.name.endswith((".lock", ".incomplete")):
                if path.stat().st_size > 1_000_000:
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                            digest.update(block)
                    hashes[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        save_json(output / "downloaded_file_hashes.json", hashes)
        status("complete", summary="pilot_pair_summary.json")
        print(json.dumps({"status": "complete", "paired_checks": paired}), flush=True)
    except Exception as error:
        status("failed", error_type=type(error).__name__, message=str(error))
        raise


if __name__ == "__main__":
    main()
