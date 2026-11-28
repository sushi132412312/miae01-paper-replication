"""在 CPU 上用合成输入演练两个条件、两种方法和三个种子的完整流程。"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.checkpoints import restore_best_weights
from src.common import PROJECT_ROOT, execution_context, read_json, save_json, tensor_hash
from src.evaluation import export_predictions, recompute_accuracy
from src.models import build_model
from src.paired import summarize
from src.replay import replay
from src.train import run


def rejection(callback, exception, message):
    try:
        callback()
    except exception as failure:
        if message not in str(failure):
            raise
        return {"rejected": True, "message": str(failure)}
    raise AssertionError(f"预期拒绝被错误放行: {message}")


def verify_later_checkpoint_wins(config):
    """固定合成评分，单独检验中途替换权重文件的路径。"""
    scores = iter([2, 5, 3])
    states = []

    def grade(model, records, batch_size, device, context, destination):
        correct = next(scores)
        states.append(tensor_hash(model.named_parameters()))
        export_predictions(destination, [
            {**context, "sample_id": row["sample_id"], "gold_label": row["label"],
             "predicted_label": row["label"] if i < correct else 1 - row["label"]}
            for i, row in enumerate(records)
        ])
        return recompute_accuracy(destination, {"samples": records}, context)

    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory) / "scripted_scores"
        with patch("src.train.evaluate_model", side_effect=grade):
            result = run(config, "ft", folder, synthetic=True)
        selection = read_json(folder / "checkpoint_selection.json")
        restored = build_model("ft", config["train_seed"], config, synthetic=True)
        restore_best_weights(restored, folder, selection["weights"])
        if result["best_epoch"] != 2 or tensor_hash(restored.named_parameters()) != states[1]:
            raise AssertionError("后续轮次得分更高时，保存的权重未更新到该轮")
        return {"source": "scripted_validation_fixture", "correct_by_epoch": [2, 5, 3],
                "selected_epoch": 2, "restored_weights_match": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)

    seeds = [42, 43, 44]
    config = read_json(PROJECT_ROOT / "configs/epoch_rehearsal.json")
    pending = read_json(PROJECT_ROOT / "configs/reduced_scale.proposed.json")
    blocked_run = output / "pending_real_run"
    guard = rejection(lambda: run(pending, "ft", blocked_run), PermissionError, "尚待组长确认")
    if blocked_run.exists():
        raise AssertionError("未定真实配置创建了运行目录")
    later_best = verify_later_checkpoint_wins(config)
    folders, replays = [], []
    for label_flips in (0, 2):
        for seed in seeds:
            current = dict(config, train_seed=seed, train_label_flips=label_flips)
            for method in ("ft", "lora"):
                folder = output / f"flips{label_flips}_{method}_seed{seed}"
                result = run(current, method, folder, synthetic=True)
                verification = replay(folder)
                folders.append(folder)
                replays.append(verification)
                print(f"{folder.name} · {result['steps']} 步 · 回读一致", flush=True)
    combined = summarize(folders, seeds)
    for condition in combined["conditions_by_flipped_label_count"].values():
        if condition["status"] != "complete":
            raise AssertionError("完整合成演练未通过配对")
    partial = summarize(folders[:2], seeds)["conditions_by_flipped_label_count"]["0"]
    if partial["status"] != "incomplete" or "methods" in partial or len(partial["missing"]) != 4:
        raise AssertionError("缺种子时未正确保留为 incomplete")

    duplicate = rejection(lambda: summarize([folders[0], folders[0]], seeds), ValueError, "重复运行")
    with tempfile.TemporaryDirectory() as temporary:
        copied = Path(temporary) / folders[0].name
        shutil.copytree(folders[0], copied)
        manifest = copied / "checkpoint_selection.json"
        selection = read_json(manifest)
        original = selection["best_epoch"]
        selection["best_epoch"] = 2 if original == 1 else 1
        save_json(manifest, selection)
        wrong_epoch = rejection(lambda: summarize([copied], seeds), ValueError, "最佳轮次")
        metrics_path = copied / "metrics.json"
        metrics = read_json(metrics_path)
        metrics["best_epoch"] = selection["best_epoch"]
        save_json(metrics_path, metrics)
        replay_wrong_epoch = rejection(lambda: replay(copied), ValueError, "最佳轮次")
        metrics["best_epoch"] = original
        save_json(metrics_path, metrics)
        selection["best_epoch"] = original
        selection["weights"]["sha256"] = "0" * 64
        save_json(manifest, selection)
        wrong_hash = rejection(lambda: replay(copied), ValueError, "SHA256")

        lora_copy = Path(temporary) / "lo_ra_config_probe"
        shutil.copytree(folders[1], lora_copy)
        config_path = lora_copy / "config.json"
        changed = read_json(config_path)
        changed["lora_alpha"] *= 2
        save_json(config_path, changed)
        replay_wrong_alpha = rejection(lambda: replay(lora_copy), ValueError, "原训练协议")

    save_json(output / "paired_summary.json", combined)
    save_json(output / "checkpoint_replay.json", replays)
    save_json(output / "acceptance.json", {
        "synthetic_only": True,
        "runs_completed": len(folders),
        "all_predictions_replayed": all(row["all_predictions_match"] for row in replays),
        "pending_config_rejection": guard,
        "partial_grid_missing_runs": partial["missing"],
        "duplicate_run_rejection": duplicate,
        "wrong_best_epoch_rejection": wrong_epoch,
        "wrong_weights_sha256_rejection": wrong_hash,
        "replay_inconsistent_best_epoch_rejection": replay_wrong_epoch,
        "replay_changed_lora_alpha_rejection": replay_wrong_alpha,
        "later_best_checkpoint_fixture": later_best,
        "execution_context": execution_context(),
    })


if __name__ == "__main__":
    main()
