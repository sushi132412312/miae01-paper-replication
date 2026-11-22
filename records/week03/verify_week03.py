"""林树辉 · week03 交付独立复核（逐轮复算 / 样本配对 / 标签翻转）。

只使用 Python 标准库；对象是 PR #6 里的 records/week03/20261122/。
用法：python verify_week03.py <records 目录> [--out 输出 json]
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

TOL = 1e-9


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_dirs(root: Path):
    return sorted(p for p in (root / "runs").iterdir() if p.is_dir())


def read_preds(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    root = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "verify_shuhuilin.json"
    report = {"reviewer": "林树辉 2260031586", "date": "2026-11-22",
              "checked": {}, "problems": []}

    # ---------- 1. 逐轮复算与最佳轮次 ----------
    per_run = {}
    for d in run_dirs(root):
        manifest = load(d / "validation_manifest.json")
        expected = {s["sample_id"]: s["label"] for s in manifest["samples"]}
        sel = load(d / "checkpoint_selection.json")
        metrics = load(d / "metrics.json")
        epochs = []
        for e in sorted((d / "epochs").iterdir()):
            rows = read_preds(e / "predictions.csv")
            if {r["sample_id"] for r in rows} != set(expected):
                report["problems"].append(f"{d.name} epoch {e.name}: 样本集合与清单不一致")
            if len(rows) != len(expected):
                report["problems"].append(f"{d.name} epoch {e.name}: 行数 {len(rows)} != {len(expected)}")
            correct = sum(int(r["predicted_label"]) == expected[r["sample_id"]] for r in rows)
            epochs.append({"epoch": int(e.name), "count": len(rows), "correct": correct,
                           "accuracy": correct / len(rows),
                           "accuracy_percent": 100 * correct / len(rows),
                           "checkpoint": rows[0]["checkpoint"]})
        # 与交付记录逐轮比对
        recorded = {x["epoch"]: x for x in sel["epochs"]}
        for ep in epochs:
            rec = recorded.get(ep["epoch"])
            if rec is None:
                report["problems"].append(f"{d.name}: 记录缺少 epoch {ep['epoch']}")
                continue
            if rec["correct"] != ep["correct"] or abs(rec["accuracy_percent"] - ep["accuracy_percent"]) > TOL:
                report["problems"].append(
                    f"{d.name} epoch {ep['epoch']}: 记录 {rec['correct']}/{rec['count']} "
                    f"vs 复算 {ep['correct']}/{ep['count']}")
        # 最佳轮次：准确率最高，同分取最早
        best = max(epochs, key=lambda x: (x["accuracy"], -x["epoch"]))
        if best["epoch"] != sel["best_epoch"]:
            report["problems"].append(
                f"{d.name}: 最佳轮次 {sel['best_epoch']} != 复算 {best['epoch']}")
        if metrics["best_epoch"] != sel["best_epoch"]:
            report["problems"].append(f"{d.name}: metrics.best_epoch 与选择记录不一致")
        if metrics["correct"] != best["correct"] or abs(metrics["accuracy_percent"] - best["accuracy_percent"]) > TOL:
            report["problems"].append(f"{d.name}: metrics 与最佳轮次复算不一致")
        per_run[d.name] = {
            "method": metrics["method"], "seed": metrics["seed"],
            "best_epoch": best["epoch"], "best_accuracy_percent": best["accuracy_percent"],
            "epochs": [{"epoch": e["epoch"], "correct": e["correct"], "percent": e["accuracy_percent"]} for e in epochs],
            "all_epochs_recorded_match": True,
            "steps": metrics["steps"],
            "train_selection_sha256": metrics["train_selection_sha256"],
            "validation_selection_sha256": metrics["validation_selection_sha256"],
        }
    report["checked"]["1_epoch_recount"] = {"runs": len(per_run), "detail": per_run}

    # ---------- 2. 种子配齐 / 重复 ----------
    grid = {}
    for name, rec in per_run.items():
        key = (name.rsplit("_", 2)[0], rec["method"])
        grid.setdefault(key, []).append(rec["seed"])
    seed_ok = all(sorted(v) == [42, 43, 44] for v in grid.values())
    dup = [k for k, v in grid.items() if len(set(v)) != len(v)]
    report["checked"]["2_seed_grid"] = {
        "grid": {f"{k[0]}|{k[1]}": sorted(v) for k, v in grid.items()},
        "all_have_42_43_44": seed_ok, "duplicate_cells": dup}

    # ---------- 3. 标签翻转 ----------
    flip_check = {}
    for method in ("ft", "lora"):
        a = load(root / "runs" / f"flips0_{method}_seed42" / "train_manifest.json")
        b = load(root / "runs" / f"flips2_{method}_seed42" / "train_manifest.json")
        ids_a = [s["sample_id"] for s in a["samples"]]
        ids_b = [s["sample_id"] for s in b["samples"]]
        tok_a = [s["tokens_sha256"] for s in a["samples"]]
        tok_b = [s["tokens_sha256"] for s in b["samples"]]
        diff = [i for i, (x, y) in enumerate(zip(a["samples"], b["samples"])) if x["label"] != y["label"]]
        flip_check[method] = {
            "same_sample_ids_and_order": ids_a == ids_b,
            "same_tokens": tok_a == tok_b,
            "label_diff_count": len(diff),
            "diff_positions": diff,
            "labels_flips0": [s["label"] for s in a["samples"]],
            "labels_flips2": [s["label"] for s in b["samples"]],
        }
    report["checked"]["3_label_flip"] = flip_check

    # ---------- 4. FT / LoRA 配对 ----------
    pair = {}
    for flips in ("flips0", "flips2"):
        for seed in (42, 43, 44):
            f = load(root / "runs" / f"{flips}_ft_seed{seed}" / "model_initialization.json")
            l = load(root / "runs" / f"{flips}_lora_seed{seed}" / "model_initialization.json")
            fm = load(root / "runs" / f"{flips}_ft_seed{seed}" / "metrics.json")
            lm = load(root / "runs" / f"{flips}_lora_seed{seed}" / "metrics.json")
            pair[f"{flips}_seed{seed}"] = {
                "common_backbone_same": f["common_backbone_sha256"] == l["common_backbone_sha256"],
                "classifier_init_same": f["classifier_sha256"] == l["classifier_sha256"],
                "train_selection_same": fm["train_selection_sha256"] == lm["train_selection_sha256"],
                "validation_selection_same": fm["validation_selection_sha256"] == lm["validation_selection_sha256"],
                "steps_same": fm["steps"] == lm["steps"],
                "trainable_ft": f["trainable_parameters"], "trainable_lora": l["trainable_parameters"],
                "lora_params": l["lora_parameters"],
            }
    report["checked"]["4_pairing"] = pair

    # ---------- 5. 汇总复算 ----------
    summary = load(root / "paired_summary.json")
    recomputed = {}
    for flips_key, block in summary["conditions_by_flipped_label_count"].items():
        rec = {}
        for method in ("ft", "lora"):
            vals = [block["methods"][method]] and None
            accs = sorted((r["seed"], r["accuracy_percent"]) for r in block["runs"] if r["method"] == method)
            values = [v for _, v in accs]
            rec[method] = {"seeds": [s for s, _ in accs], "mean": statistics.mean(values),
                           "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                           "recorded_mean": block["methods"][method]["mean_percent"],
                           "recorded_sd": block["methods"][method]["sample_sd_pp"]}
        by_seed = {}
        for seed in (42, 43, 44):
            fv = next(r["accuracy_percent"] for r in block["runs"] if r["method"] == "ft" and r["seed"] == seed)
            lv = next(r["accuracy_percent"] for r in block["runs"] if r["method"] == "lora" and r["seed"] == seed)
            by_seed[seed] = lv - fv
        rec["lora_minus_ft"] = {"by_seed": by_seed, "mean": statistics.mean(by_seed.values()),
                                "sample_sd": statistics.stdev(list(by_seed.values())),
                                "recorded": block["lora_minus_ft"]}
        recomputed[flips_key] = rec
    report["checked"]["5_summary_recount"] = recomputed

    # ---------- 一致性判定 ----------
    consistent = {
        "epoch_counts_match": not any("复算" in p or "不一致" in p for p in report["problems"]),
        "seeds_complete": seed_ok,
        "flip_is_exactly_two": all(v["label_diff_count"] == 2 for v in flip_check.values()),
        "pairing_same": all(all(x is True for k, x in v.items() if k.endswith("same"))
                            for v in pair.values()),
        "summary_means_match": all(
            abs(rec[m]["mean"] - rec[m]["recorded_mean"]) < 1e-6
            for rec in recomputed.values() for m in ("ft", "lora")),
    }
    report["consistent"] = consistent
    report["problems"] = report["problems"] or []
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"consistent": consistent, "problems": report["problems"][:10],
                      "runs": len(per_run)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
