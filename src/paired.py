"""用标准库复算按轮训练结果并检查 FT/LoRA 的配对条件。"""
import argparse
import math
import random
import statistics
from pathlib import Path

from .common import json_hash, read_json, save_json
from .data import validate_real_config, validate_records
from .evaluation import recompute_accuracy, summarize_seed_results
from .protocol import training_steps, uses_epoch_schedule, validate_epoch_schedule


def read_manifest(folder, split, count):
    manifest = read_json(folder / f"{split}_manifest.json")
    validate_records(manifest["samples"])
    if (manifest["split"] != split or manifest["count"] != count
            or len(manifest["samples"]) != count
            or json_hash(manifest["samples"]) != manifest["selection_sha256"]):
        raise ValueError(f"{folder.name} 的 {split} 清单与记录不符")
    return manifest


def read_verified_selection(folder, config, metrics, validation):
    """按原始逐轮预测复算并验证保存的最佳轮次。"""
    context = {key: metrics[key] for key in ("run_id", "method", "seed")}
    selection = read_json(folder / "checkpoint_selection.json")
    history = selection["epochs"]
    batches = math.ceil(config["train_size"] / config["batch_size"])
    if selection["rule"] != config["checkpoint_rule"] or len(history) != config["epochs"]:
        raise ValueError(f"{folder.name} 的检查点规则或轮数不符")
    for epoch, record in enumerate(history, start=1):
        relative = f"epochs/{epoch:03}/predictions.csv"
        checkpoint = f"epoch-{epoch}-step-{epoch * batches}"
        expected = {**context, "checkpoint": checkpoint}
        score = recompute_accuracy(folder / relative, validation, expected)
        if (record["epoch"] != epoch or record["predictions"] != relative
                or record["updates"] != epoch * batches
                or any(record[key] != value for key, value in score.items())):
            raise ValueError(f"{folder.name} 第 {epoch} 轮预测与指标不一致")
    best = max(history, key=lambda result: result["correct"])
    score = recompute_accuracy(folder / "predictions.csv", validation, {**context, "checkpoint": best["checkpoint"]})
    if (selection["best_epoch"] != best["epoch"] or metrics["best_epoch"] != best["epoch"]
            or metrics["best_epoch_steps"] != best["updates"]
            or selection["best_checkpoint"] != best["checkpoint"]
            or any(metrics[key] != value or best[key] != value for key, value in score.items())
            or (folder / "predictions.csv").read_text(encoding="utf-8")
            != (folder / best["predictions"]).read_text(encoding="utf-8")):
        raise ValueError(f"{folder.name} 的最佳轮次或逐例预测不符合选择规则")
    return selection


def read_run(folder):
    folder = Path(folder)
    config = read_json(folder / "config.json")
    metrics = read_json(folder / "metrics.json")
    if not uses_epoch_schedule(config):
        raise ValueError(f"{folder.name} 不是按轮训练记录")
    validate_epoch_schedule(config)
    synthetic = config["status"] == "synthetic_selfcheck"
    if not synthetic:
        validate_real_config(config)
    kind = "synthetic_epoch_rehearsal" if synthetic else "reduced_scale_experiment"
    protocol = {key: value for key, value in config.items() if key != "train_seed"}
    train = read_manifest(folder, "train", config["train_size"])
    validation = read_manifest(folder, "validation", config["validation_size"])
    if (metrics["seed"] != config["train_seed"] or metrics["experiment_kind"] != kind
            or metrics["protocol_sha256"] != json_hash(protocol)
            or metrics["train_selection_sha256"] != train["selection_sha256"]
            or metrics["validation_selection_sha256"] != validation["selection_sha256"]
            or metrics["steps"] != training_steps(config)):
        raise ValueError(f"{folder.name} 的配置、指标、样本清单或更新步数不一致")
    read_verified_selection(folder, config, metrics, validation)

    steps = read_json(folder / "training_steps.json")
    sample_order = [step["sample_ids"] for step in steps]
    batches = math.ceil(config["train_size"] / config["batch_size"])
    if len(steps) != metrics["steps"] or any(step["step"] != i for i, step in enumerate(steps, start=1)):
        raise ValueError(f"{folder.name} 的训练步次记录缺失或乱序")
    train_ids = sorted(sample["sample_id"] for sample in train["samples"])
    for first in range(0, len(steps), batches):
        ids = sorted(sample_id for group in sample_order[first:first + batches] for sample_id in group)
        if ids != train_ids:
            raise ValueError(f"{folder.name} 的训练轮次没有完整覆盖固定训练子集")

    flip_count = config["train_label_flips"]
    noise_seed = config["label_noise_seed"]
    position = random.Random(noise_seed).sample(range(len(train["samples"])), flip_count)
    flip_ids = {train["samples"][i]["sample_id"] for i in position}
    clean_samples = [{**row, "label": 1 - row["label"]} if row["sample_id"] in flip_ids else row
                     for row in train["samples"]]
    if (train["train_label_flips"] != flip_count or train["label_noise_seed"] != noise_seed
            or train["flipped_sample_ids"] != sorted(flip_ids)
            or train["clean_selection_sha256"] != json_hash(clean_samples)):
        raise ValueError(f"{folder.name} 的训练标签翻转证据不一致")

    initial = read_json(folder / "model_initialization.json")
    check = read_json(folder / "parameter_check.json")
    if (check["initial"] != initial or not check["frozen_weights_unchanged"]
            or initial["frozen_parameters_sha256"] != check["final"]["frozen_parameters_sha256"]
            or not check["changed_trainable_names"]):
        raise ValueError(f"{folder.name} 的冻结或更新证据不一致")
    if metrics["method"] == "lora" and initial["lora_B_all_zero"] is not True:
        raise ValueError(f"{folder.name} 的 LoRA B 初始化不为零")
    return {"metrics": metrics, "protocol": protocol, "train": train, "validation": validation,
            "initial": initial, "sample_order_sha256": json_hash(sample_order)}


def summarize_differences(values):
    return {"by_seed_pp": values, "mean_pp": statistics.mean(values.values()),
            "sample_sd_pp": statistics.stdev(values.values())}


def summarize(folders, seeds):
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("需要至少三个不同的计划种子")
    if not folders:
        raise ValueError("没有运行目录")
    runs = [read_run(folder) for folder in folders]
    first = runs[0]
    base_protocol = {k: v for k, v in first["protocol"].items() if k != "train_label_flips"}
    groups = {}
    seed_pairing = {}
    for run in runs:
        metrics, protocol = run["metrics"], run["protocol"]
        seed, method = metrics["seed"], metrics["method"]
        if seed not in seeds or method not in ("ft", "lora"):
            raise ValueError("结果包含计划外的种子或方法")
        if ({k: v for k, v in protocol.items() if k != "train_label_flips"} != base_protocol
                or run["validation"] != first["validation"]
                or run["train"]["clean_selection_sha256"] != first["train"]["clean_selection_sha256"]
                or metrics["device"] != first["metrics"]["device"]):
            raise ValueError("结果混用了不同的基础协议、原始数据、验证集或设备类型")
        pairing = (run["initial"]["common_backbone_sha256"], run["initial"]["classifier_sha256"],
                   run["sample_order_sha256"])
        if seed_pairing.setdefault(seed, pairing) != pairing:
            raise ValueError(f"种子 {seed} 的共同初值或样本顺序没有配对")
        flips = protocol["train_label_flips"]
        grid = groups.setdefault(flips, {})
        if (method, seed) in grid:
            raise ValueError(f"重复运行，标签翻转数 {flips}、方法 {method}、种子 {seed}")
        if grid and run["train"] != next(iter(grid.values()))["train"]:
            raise ValueError("同一条件下的实际训练清单不同")
        grid[(method, seed)] = run

    wanted = {(method, seed) for method in ("ft", "lora") for seed in seeds}
    summaries = {}
    for flips, grid in sorted(groups.items()):
        missing = [{"method": method, "seed": seed} for method, seed in sorted(wanted - grid.keys())]
        summary = {"status": "incomplete" if missing else "complete", "missing": missing, "runs": []}
        for (method, seed), run in sorted(grid.items()):
            metrics = run["metrics"]
            keys = ("run_id", "method", "seed", "best_epoch", "steps", "checkpoint",
                    "count", "correct", "accuracy_percent")
            summary["runs"].append({key: metrics[key] for key in keys})
        if not missing:
            summary["methods"] = {method: summarize_seed_results([grid[(method, s)]["metrics"] for s in seeds])
                                  for method in ("ft", "lora")}
            summary["lora_minus_ft"] = summarize_differences({
                s: grid[("lora", s)]["metrics"]["accuracy_percent"] - grid[("ft", s)]["metrics"]["accuracy_percent"]
                for s in seeds})
        summaries[str(flips)] = summary

    extension = {}
    if 0 in groups and summaries["0"]["status"] == "complete":
        for flips, grid in sorted(groups.items()):
            if flips and summaries[str(flips)]["status"] == "complete":
                extension[str(flips)] = {
                    method: summarize_differences({
                        s: groups[0][(method, s)]["metrics"]["accuracy_percent"]
                        - grid[(method, s)]["metrics"]["accuracy_percent"] for s in seeds})
                    for method in ("ft", "lora")
                }
    return {"experiment_kind": first["metrics"]["experiment_kind"], "planned_seeds": sorted(seeds),
            "train_size": base_protocol["train_size"], "validation_size": base_protocol["validation_size"],
            "conditions_by_flipped_label_count": summaries, "clean_minus_noise": extension,
            "difference_unit": "percentage points"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("请指定新的汇总文件")
    result = summarize(args.runs, args.seeds)
    save_json(args.output, result)
    print(f"已复算 {len(args.runs)} 次运行，类型 {result['experiment_kind']}")
    for flips, summary in result["conditions_by_flipped_label_count"].items():
        print(f"训练翻转 {flips} 条标签，配对状态 {summary['status']}")


if __name__ == "__main__":
    main()
