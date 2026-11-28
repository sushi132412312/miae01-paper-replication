"""Recount and audit the archived three-seed V100 experiments."""
import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.common import read_json, save_json
from src.data import batch_order
from src.paired import summarize

SEEDS = (42, 43, 44)
METHODS = ("ft", "lora")
PAPER_ACCURACY = {"ft": 94.8, "lora": 95.1}


def check_draws(folder, config, manifest):
    samples = manifest["samples"]
    batch_size = config["batch_size"]
    expected = []
    for epoch, order in enumerate(batch_order(len(samples), config["train_seed"], config["epochs"]), 1):
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            expected.append((epoch, [samples[index]["sample_id"] for index in batch]))
    observed = [(row["epoch"], row["sample_ids"]) for row in read_json(folder / "training_steps.json")]
    if observed != expected:
        raise ValueError(f"{folder.name}: batch order does not match its recorded seed")


def audit(records_root, verify_sample_order=False):
    folders = [
        records_root / f"seed{seed}_20261122" / f"{condition}_{method}_seed{seed}"
        for seed in SEEDS for condition in ("clean", "noise") for method in METHODS
    ]
    paired = summarize(folders, SEEDS)
    conditions = paired["conditions_by_flipped_label_count"]
    if (paired["experiment_kind"] != "reduced_scale_experiment" or paired["train_size"] != 512
            or paired["validation_size"] != 872 or set(conditions) != {"0", "51"}
            or any(condition["status"] != "complete" for condition in conditions.values())):
        raise ValueError("The archived V100 experiment matrix is incomplete or has a different scope")

    runs = []
    for folder in folders:
        config = read_json(folder / "config.json")
        metrics = read_json(folder / "metrics.json")
        replay = read_json(folder / "replay.json")
        accepted = read_json(folder / "verified_run.json")
        manifest = read_json(folder / "validation_manifest.json")
        selected = read_json(folder / "checkpoint_selection.json")
        parameters = read_json(folder / "model_initialization.json")
        if (config["epochs"] != 3 or config["batch_size"] != 8 or config["max_length"] != 128
                or metrics["device"] != "cuda" or manifest["source_split_count"] != 872):
            raise ValueError(f"{folder.name}: expected the approved V100 protocol and full validation split")
        keys = ("run_id", "method", "seed", "checkpoint", "best_epoch", "count", "correct", "accuracy_percent")
        if (any(replay[key] != metrics[key] for key in keys) or not replay["all_predictions_match"]
                or replay["verified_weights_sha256"] != selected["weights"]["sha256"]
                or not accepted["all_predictions_replayed"]):
            raise ValueError(f"{folder.name}: the stored checkpoint replay evidence is inconsistent")
        if verify_sample_order:
            check_draws(folder, config, read_json(folder / "train_manifest.json"))
        runs.append({
            **{key: metrics[key] for key in keys},
            "folder_relative_to_records_root": folder.relative_to(records_root).as_posix(),
            "train_label_flips": config["train_label_flips"],
            "trainable_parameters": parameters["trainable_parameters"],
            "lora_parameters": parameters["lora_parameters"],
            "classifier_parameters": parameters["classifier_parameters"],
            "peak_allocated_mib": metrics["gpu_peak_total_allocated_mib"],
            "training_process_seconds": accepted["training_process_seconds"],
            "replay_process_seconds": accepted["replay_process_seconds"],
            "steps": metrics["steps"],
            "epoch_accuracies_percent": [epoch["accuracy_percent"] for epoch in selected["epochs"]],
        })

    curves = {}
    for flips in (0, 51):
        curves[str(flips)] = {}
        for method in METHODS:
            group = [r for r in runs if r["method"] == method and r["train_label_flips"] == flips]
            curves[str(flips)][method] = []
            for epoch in (1, 2, 3):
                scores = [r["epoch_accuracies_percent"][epoch - 1] for r in group]
                curves[str(flips)][method].append({
                    "epoch": epoch, "steps": 64 * epoch,
                    "mean_percent": statistics.mean(scores), "sample_sd_pp": statistics.stdev(scores),
                })

    paper = {}
    for method, reported in PAPER_ACCURACY.items():
        ours = conditions["0"]["methods"][method]["mean_percent"]
        paper[method] = {
            "paper_percent_as_displayed": reported,
            "ours_mean_percent": ours,
            "ours_minus_paper_pp": ours - reported,
            "relative_difference_percent": (ours - reported) / reported * 100,
        }

    resources = {}
    for method in METHODS:
        group = [r for r in runs if r["method"] == method]
        params = {r["trainable_parameters"] for r in group}
        if len(params) != 1:
            raise ValueError(f"Trainable parameter counts vary for {method}")
        resources[method] = {
            "trainable_parameters": params.pop(),
            "max_peak_allocated_mib": max(r["peak_allocated_mib"] for r in group),
            "training_process_seconds_min": min(r["training_process_seconds"] for r in group),
            "training_process_seconds_max": max(r["training_process_seconds"] for r in group),
        }

    result = {
        "run_count": len(runs),
        "epoch_file_count": sum(len(r["epoch_accuracies_percent"]) for r in runs),
        "epoch_prediction_rows": sum(len(r["epoch_accuracies_percent"]) * r["count"] for r in runs),
        "recorded_best_weight_replays_verified": len(runs),
        "sample_order_regenerated": verify_sample_order,
        "evidence": {"training_records_date": "2026-11-22", "subdirs": [f"seed{s}_20261122" for s in SEEDS]},
        "paired": paired,
        "by_epoch": curves,
        "paper_table_2": paper,
        "resources": resources,
        "runs": runs,
    }
    if verify_sample_order:
        import torch
        result["sample_order_torch_version"] = str(torch.__version__)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, default=PROJECT / "records/week03")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-sample-order", action="store_true", help="Also regenerate the CPU torch permutations")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    result = audit(args.records_root, args.verify_sample_order)
    save_json(args.output, result)
    print(json.dumps({key: result[key] for key in (
        "run_count", "epoch_file_count", "epoch_prediction_rows", "sample_order_regenerated"
    )}))


if __name__ == "__main__":
    main()
