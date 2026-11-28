"""在相同运行环境中，从保存的最佳权重重新生成验证预测。"""
import argparse
import tempfile
from pathlib import Path

from .checkpoints import restore_best_weights
from .common import PROJECT_ROOT, json_hash, read_json, save_json
from .data import validate_real_config
from .evaluation import evaluate_model
from .models import build_model, parameter_report
from .paired import read_manifest, read_verified_selection
from .protocol import training_steps, uses_epoch_schedule, validate_epoch_schedule
from .train import load_run_data


def replay(folder, device=None, cache_dir=None):
    folder = Path(folder)
    config = read_json(folder / "config.json")
    metrics = read_json(folder / "metrics.json")
    if not uses_epoch_schedule(config):
        raise ValueError("回读需要按轮训练记录")
    synthetic = config.get("status") == "synthetic_selfcheck"
    if synthetic:
        validate_epoch_schedule(config)
    else:
        validate_real_config(config)
    protocol = {key: value for key, value in config.items() if key != "train_seed"}
    if (metrics["protocol_sha256"] != json_hash(protocol) or metrics["seed"] != config["train_seed"]
            or metrics["steps"] != training_steps(config)):
        raise ValueError("保存的指标与原训练协议、种子或步数不一致")
    validation = read_manifest(folder, "validation", config["validation_size"])
    if validation["selection_sha256"] != metrics["validation_selection_sha256"]:
        raise ValueError("保存的验证清单与原运行不同")
    selection = read_verified_selection(folder, config, metrics, validation)
    initial = read_json(folder / "model_initialization.json")
    device = device or metrics["device"]
    cache_dir = Path(cache_dir or PROJECT_ROOT / "cache")
    data, manifests = load_run_data(config, synthetic, cache_dir)
    if manifests["validation"]["selection_sha256"] != metrics["validation_selection_sha256"]:
        raise ValueError("回读时的验证集与原运行不同")

    context = {key: metrics[key] for key in ("run_id", "method", "seed", "checkpoint")}
    model = build_model(context["method"], context["seed"], config, synthetic, str(cache_dir))
    original = parameter_report(model)
    if original["common_backbone_sha256"] != initial["common_backbone_sha256"]:
        raise ValueError("回读使用的基础模型与原运行不同")
    restore_best_weights(model, folder, selection["weights"])
    model.to(device)
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / "predictions.csv"
        result = evaluate_model(model, data["validation"], config["batch_size"], device, context, destination)
        if destination.read_text(encoding="utf-8") != (folder / "predictions.csv").read_text(encoding="utf-8"):
            raise ValueError("回读最佳权重后，逐例预测与保存文件不一致")
    if any(result[key] != metrics[key] for key in ("count", "correct", "accuracy_percent")):
        raise ValueError("回读后的准确率与运行指标不符")
    return {**result, "best_epoch": selection["best_epoch"], "device": device,
            "verified_weights_sha256": selection["weights"]["sha256"],
            "all_predictions_match": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("请指定新的回读记录文件")
    result = replay(args.run, args.device, args.cache_dir)
    save_json(args.output, result)
    print(f"回读通过，最佳轮次 {result['best_epoch']}，{result['count']} 条预测完全一致")


if __name__ == "__main__":
    main()
