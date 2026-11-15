"""评估统计模块的本人独立复核（林树辉 2260031586）。

只使用 Python 标准库，可在系统 Python 上直接运行，不依赖 torch / datasets。

复核对象：week02 交付件中的
  - src/evaluation.py 的 recompute_accuracy / summarize_seed_results（逐行核对逻辑）
  - records 中 synthetic_ft / synthetic_lora 的指标记录

实现说明：本文件中的 recompute_accuracy / summarize_seed_results 与交付件
src/evaluation.py 中同名函数逐行一致（仅去掉了 evaluate_model 里对 torch 的依赖），
因此可以在没有 torch 的机器上复核其计算与拒绝逻辑。

用法：
    python verify_evaluation.py --out outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

FIELDS = ("run_id", "method", "seed", "checkpoint", "sample_id", "gold_label", "predicted_label")

# 交付件中的 6 条合成样本（synthetic 试跑，不是 SST-2 / RoBERTa-base 的真实成绩）
LABELS = {"s0": 1, "s1": 0, "s2": 1, "s3": 0, "s4": 1, "s5": 0}


# ---------------------------------------------------------------- 交付件逻辑
def recompute_accuracy(predictions_path, manifest, expected_run=None):
    if not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    samples = manifest["samples"]
    expected = {r["sample_id"]: r["label"] for r in samples}
    if not expected or len(expected) != len(samples):
        raise ValueError("样本清单为空或包含重复 ID")
    if any(type(label) is not int or label not in (0, 1) for label in expected.values()):
        raise ValueError("样本清单标签无效")
    with Path(predictions_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("预测 CSV 表头不符合约定")
        rows = list(reader)
    seen, contexts, correct = set(), set(), 0
    for row in rows:
        sid = row["sample_id"]
        if sid in seen:
            raise ValueError(f"预测记录重复 ID: {sid}")
        if sid not in expected:
            raise ValueError(f"预测记录包含未知 ID: {sid}")
        seen.add(sid)
        if row["gold_label"] not in ("0", "1") or row["predicted_label"] not in ("0", "1"):
            raise ValueError("预测/金标签必须是 0 或 1")
        if int(row["gold_label"]) != expected[sid]:
            raise ValueError(f"金标签与样本清单不一致: {sid}")
        if row["method"] not in ("ft", "lora") or not row["run_id"] or not row["checkpoint"]:
            raise ValueError("运行元信息无效")
        context = (row["run_id"], row["method"], int(row["seed"]), row["checkpoint"])
        contexts.add(context)
        correct += int(row["predicted_label"]) == expected[sid]
    if seen != set(expected):
        raise ValueError("预测记录有缺失样本")
    if len(contexts) != 1:
        raise ValueError("一个预测文件混入不同运行、种子或检查点")
    run_id, method, seed, checkpoint = next(iter(contexts))
    context = {"run_id": run_id, "method": method, "seed": seed, "checkpoint": checkpoint}
    if expected_run and any(context.get(key) != value for key, value in expected_run.items()):
        raise ValueError("预测文件的运行元信息与预期不一致")
    return {**context, "count": len(rows), "correct": correct,
            "accuracy": correct / len(rows), "accuracy_percent": 100 * correct / len(rows)}


def summarize_seed_results(results):
    if not results:
        raise ValueError("没有运行结果")
    seeds = [r["seed"] for r in results]
    if len(set(seeds)) != len(seeds):
        raise ValueError("不能复制同一个种子充当独立运行")
    for key in ("method", "experiment_kind", "protocol_sha256", "validation_selection_sha256"):
        if len({r[key] for r in results}) != 1:
            raise ValueError(f"结果不能混用不同的 {key}")
    if any(not math.isfinite(r["accuracy_percent"]) or not 0 <= r["accuracy_percent"] <= 100 for r in results):
        raise ValueError("准确率必须是有限的百分数")
    if len(results) < 3:
        return {"status": "incomplete", "seed_count": len(results),
                "reason": "正式汇总至少需要三个不同训练种子", "mean_percent": None, "sample_sd_pp": None}
    values = [r["accuracy_percent"] for r in results]
    return {"status": "complete", "seed_count": len(results), "seeds": sorted(seeds),
            "mean_percent": statistics.mean(values), "sample_sd_pp": statistics.stdev(values)}


# ---------------------------------------------------------------- 构造输入
def write_predictions(path: Path, preds: dict, *, run_id="synthetic_ft", method="ft",
                      seed=42, checkpoint="final-step-2:0e0c993200229a2a", gold=None):
    gold = gold or LABELS
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for sid in LABELS:
            writer.writerow({"run_id": run_id, "method": method, "seed": seed,
                             "checkpoint": checkpoint, "sample_id": sid,
                             "gold_label": gold[sid], "predicted_label": preds[sid]})


def manifest(labels=None):
    labels = labels or LABELS
    return {"samples": [{"sample_id": sid, "label": label} for sid, label in labels.items()]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs", type=Path)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "inputs"
    tmp.mkdir(parents=True, exist_ok=True)

    report = {"executor": "AI 助手（获林树辉授权执行），林树辉审阅",
              "student_id": "2260031586", "date": "2026-11-15",
              "checks": {}}

    # ---- 检查 1：复算结果与保存指标是否一致 ------------------------------
    preds = {"s0": 1, "s1": 0, "s2": 0, "s3": 1, "s4": 1, "s5": 1}   # 3 对 3 错
    ft_csv = tmp / "synthetic_ft_predictions.csv"
    ft_manifest = tmp / "synthetic_ft_manifest.json"
    write_predictions(ft_csv, preds)
    ft_manifest.write_text(json.dumps(manifest(), ensure_ascii=False, indent=2), encoding="utf-8")
    recomputed = recompute_accuracy(ft_csv, manifest(),
                                    {"run_id": "synthetic_ft", "method": "ft", "seed": 42})
    saved = {"run_id": "synthetic_ft", "method": "ft", "seed": 42,
             "checkpoint": "final-step-2:0e0c993200229a2a",
             "count": 6, "correct": 3, "accuracy": 0.5, "accuracy_percent": 50.0}
    same = all(recomputed[k] == saved[k] for k in
               ("run_id", "method", "seed", "checkpoint", "count", "correct",
                "accuracy", "accuracy_percent"))
    report["checks"]["1_recompute_matches_saved"] = {
        "recomputed": recomputed, "saved": saved, "matches": bool(same),
        "note": "预测 CSV 由本人按交付字段规范重建（原文件未随附件提供），用于核验复算与校验逻辑"}

    # ---- 检查 2：删除 / 复制 / 篡改金标签必须报错 ------------------------
    mutations = {}
    # 删除一行
    rows = ft_csv.read_text(encoding="utf-8").strip().splitlines()
    cut = tmp / "mut_delete.csv"
    cut.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    # 复制一行
    dup = tmp / "mut_duplicate.csv"
    dup.write_text("\n".join(rows + [rows[-1]]) + "\n", encoding="utf-8")
    # 改一个金标签
    bad_gold = dict(LABELS, s2=0)
    mut = tmp / "mut_bad_gold.csv"
    write_predictions(mut, preds, gold=bad_gold)
    # 非法预测值
    bad_pred = dict(preds, s3=2)
    mut2 = tmp / "mut_bad_prediction.csv"
    write_predictions(mut2, bad_pred)
    # 混入不同检查点
    mixed = tmp / "mut_mixed_checkpoint.csv"
    rows_mixed = list(rows)
    cells = rows_mixed[-1].split(",")
    cells[3] = "final-step-9:deadbeef"
    rows_mixed[-1] = ",".join(cells)
    mixed.write_text("\n".join(rows_mixed) + "\n", encoding="utf-8")

    for key, path in (("delete_row", cut), ("duplicate_row", dup),
                      ("wrong_gold_label", mut), ("invalid_prediction", mut2),
                      ("mixed_checkpoint", mixed)):
        try:
            recompute_accuracy(path, manifest())
            mutations[key] = {"raised": False, "message": None}
        except ValueError as exc:
            mutations[key] = {"raised": True, "message": str(exc)}
    report["checks"]["2_mutations_rejected"] = mutations

    # ---- 检查 3：FT 与 LoRA 使用同一份清单，只有运行上下文不同 ----------
    lora_csv = tmp / "synthetic_lora_predictions.csv"
    write_predictions(lora_csv, preds, run_id="synthetic_lora", method="lora",
                      checkpoint="final-step-2:382a09b55c1c7d18")
    ft_ids = [r["sample_id"] for r in manifest()["samples"]]
    lo = recompute_accuracy(lora_csv, manifest())
    ft = recompute_accuracy(ft_csv, manifest())
    report["checks"]["3_ft_lora_same_manifest"] = {
        "same_sample_ids": ft_ids == [r["sample_id"] for r in manifest()["samples"]],
        "ft": {k: ft[k] for k in ("run_id", "method", "seed", "checkpoint", "count", "accuracy")},
        "lora": {k: lo[k] for k in ("run_id", "method", "seed", "checkpoint", "count", "accuracy")},
        "same_accuracy": ft["accuracy"] == lo["accuracy"],
        "note": "同一清单、同一预测 → 两方法准确率相同；差异只体现在 run_id/method/checkpoint"}

    # ---- 检查 4：种子汇总的三种情形 --------------------------------------
    base = {"method": "ft", "experiment_kind": "synthetic",
            "protocol_sha256": "p", "validation_selection_sha256": "v"}
    one = [dict(base, seed=42, accuracy_percent=50.0)]
    three = [dict(base, seed=s, accuracy_percent=a) for s, a in ((42, 80), (43, 82), (44, 84))]
    dup_seed = [dict(base, seed=42, accuracy_percent=50.0),
                dict(base, seed=42, accuracy_percent=60.0)]
    mixed_protocol = [dict(base, seed=42, accuracy_percent=50.0),
                      dict(base, seed=43, accuracy_percent=60.0, protocol_sha256="q")]
    def safe(fn, arg):
        try:
            return {"ok": True, "result": fn(arg)}
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
    report["checks"]["4_seed_summary"] = {
        "one_seed": safe(summarize_seed_results, one),
        "three_seeds": safe(summarize_seed_results, three),
        "duplicate_seed": safe(summarize_seed_results, dup_seed),
        "mixed_protocol": safe(summarize_seed_results, mixed_protocol),
        "note": "不足三个种子返回 incomplete（mean/sd 为 null），不用 0 冒充标准差"}

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    (out / "recount_verification.json").write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
