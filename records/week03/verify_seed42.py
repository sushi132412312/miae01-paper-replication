"""林树辉 · seed42 正式缩小规模实验的独立复核。

对象：`林树辉_正式实验_seed42_20261122` 交付包中的 `02_运行记录/`
  clean_ft_seed42 / clean_lora_seed42 / noise_ft_seed42 / noise_lora_seed42
每轮 872 条验证；三个 epoch；最佳轮次按验证准确率、同分取最早。

只使用 Python 标准库。
用法：python verify_seed42.py <交付包目录> <输出 json>
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

RUNS = ("clean_ft_seed42", "clean_lora_seed42", "noise_ft_seed42", "noise_lora_seed42")
TOL = 1e-9


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    root = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    report = {"reviewer": "林树辉 2260031586", "date": "2026-11-22",
              "scope": "seed42 四种运行（clean/noise x ft/lora），每轮 872 条验证",
              "problems": [], "checks": {}}

    runs = {}
    for name in RUNS:
        d = root / name
        manifest = load(d / "validation_manifest.json")
        expected = {s["sample_id"]: s["label"] for s in manifest["samples"]}
        sel = load(d / "checkpoint_selection.json")
        metrics = load(d / "metrics.json")
        epochs = []
        for e in sorted((d / "epochs").iterdir()):
            rows = list(csv.DictReader((e / "predictions.csv").open(encoding="utf-8-sig", newline="")))
            if len(rows) != len(expected):
                report["problems"].append(f"{name} epoch {e.name}: 行数 {len(rows)} != {len(expected)}")
            if {r["sample_id"] for r in rows} != set(expected):
                report["problems"].append(f"{name} epoch {e.name}: 样本集合与清单不一致")
            correct = sum(int(r["predicted_label"]) == expected[r["sample_id"]] for r in rows)
            epochs.append({"epoch": int(e.name), "count": len(rows), "correct": correct,
                           "percent": 100 * correct / len(rows)})
        rec = {x["epoch"]: x for x in sel["epochs"]}
        for ep in epochs:
            r = rec.get(ep["epoch"])
            if not r or r["correct"] != ep["correct"] or abs(r["accuracy_percent"] - ep["percent"]) > TOL:
                report["problems"].append(f"{name}: epoch {ep['epoch']} 与记录不一致")
        best = max(epochs, key=lambda x: (x["percent"], -x["epoch"]))
        if best["epoch"] != sel["best_epoch"]:
            report["problems"].append(f"{name}: 最佳轮次 {sel['best_epoch']} != 复算 {best['epoch']}")
        if metrics["correct"] != best["correct"]:
            report["problems"].append(f"{name}: metrics.correct 与最佳轮次不一致")
        runs[name] = {
            "method": metrics["method"], "seed": metrics["seed"],
            "epochs": epochs, "best_epoch": best["epoch"],
            "best_correct": best["correct"], "best_percent": best["percent"],
            "checkpoint": sel["best_checkpoint"],
            "selection_rule": sel["rule"],
            "validation_size": len(expected),
            "train_selection_sha256": metrics["train_selection_sha256"],
            "validation_selection_sha256": metrics["validation_selection_sha256"],
            "steps": metrics["steps"],
        }
    report["checks"]["1_epoch_recount"] = runs

    # 2. 标签翻转：clean vs noise 训练清单
    flips = {}
    for method in ("ft", "lora"):
        a = load(root / f"clean_{method}_seed42" / "train_manifest.json")
        b = load(root / f"noise_{method}_seed42" / "train_manifest.json")
        same_ids = [s["sample_id"] for s in a["samples"]] == [s["sample_id"] for s in b["samples"]]
        diff = [i for i, (x, y) in enumerate(zip(a["samples"], b["samples"])) if x["label"] != y["label"]]
        flips[method] = {"same_sample_ids_and_order": same_ids,
                         "label_diff_count": len(diff),
                         "expected_flips_from_verified_run": 51}
    report["checks"]["2_label_flips"] = flips

    # 3. 配对：同种子的 FT / LoRA 共用验证集与训练集
    pair = {}
    for cond in ("clean", "noise"):
        f = load(root / f"{cond}_ft_seed42" / "model_initialization.json")
        l = load(root / f"{cond}_lora_seed42" / "model_initialization.json")
        fm = load(root / f"{cond}_ft_seed42" / "metrics.json")
        lm = load(root / f"{cond}_lora_seed42" / "metrics.json")
        pair[cond] = {
            "common_backbone_same": f["common_backbone_sha256"] == l["common_backbone_sha256"],
            "classifier_init_same": f["classifier_sha256"] == l["classifier_sha256"],
            "train_selection_same": fm["train_selection_sha256"] == lm["train_selection_sha256"],
            "validation_selection_same": fm["validation_selection_sha256"] == lm["validation_selection_sha256"],
            "steps_same": fm["steps"] == lm["steps"],
            "trainable_ft": f["trainable_parameters"], "trainable_lora": l["trainable_parameters"],
            "lora_params": l["lora_parameters"],
        }
    report["checks"]["3_pairing"] = pair

    # 4. 与三种子汇总表核对（只验 seed42 行的算术自洽）
    summary = None
    for p in root.rglob("paired_summary_all_seeds.json"):
        summary = load(p)
        break
    if summary:
        rows = {}
        for cond_key, block in summary["conditions_by_flipped_label_count"].items():
            for r in block["runs"]:
                if r["seed"] == 44:
                    rows[f"{cond_key}:{r['method']}"] = {"correct": r["correct"],
                                                         "percent": r["accuracy_percent"]}
        listed = {"clean_ft": runs["clean_ft_seed42"]["best_percent"],
                  "clean_lora": runs["clean_lora_seed42"]["best_percent"],
                  "noise_ft": runs["noise_ft_seed42"]["best_percent"],
                  "noise_lora": runs["noise_lora_seed42"]["best_percent"]}
        mapping = {"0:ft": "clean_ft", "0:lora": "clean_lora", "51:ft": "noise_ft", "51:lora": "noise_lora"}
        cmp = {}
        for k, v in rows.items():
            mine = listed[mapping.get(k, "")] if k in mapping else None
            cmp[k] = {"summary_percent": v["percent"], "my_percent": mine,
                      "match": mine is not None and abs(v["percent"] - mine) < 1e-9}
        report["checks"]["4_summary_rows_seed42"] = cmp

    problems = report["problems"]
    report["consistent"] = {
        "epoch_recount_matches": not problems,
        "flips_are_51": all(v["label_diff_count"] == 51 for v in flips.values()),
        "pairing_ok": all(all(x is True for k, x in v.items() if k.endswith("same")) for v in pair.values()),
        "summary_rows_match": all(v["match"] for v in report["checks"].get("4_summary_rows_seed42", {}).values()),
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"consistent": report["consistent"], "problems": problems[:5],
                      "runs": {k: {"best_epoch": v["best_epoch"], "correct": v["best_correct"],
                                   "percent": round(v["best_percent"], 4)} for k, v in runs.items()}},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
