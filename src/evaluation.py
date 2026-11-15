"""逐例预测和可独立复算的统计；正确标签取自可信的样本清单。"""
import csv
import math
import statistics
from pathlib import Path

from .common import read_json, save_json

FIELDS = ("run_id", "method", "seed", "checkpoint", "sample_id", "gold_label", "predicted_label")


def export_predictions(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def recompute_accuracy(predictions_path, manifest, expected_run=None):
    if not isinstance(manifest, dict):
        manifest = read_json(manifest)
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


def evaluate_model(model, records, batch_size, device, context, destination):
    import torch
    was_training = model.training
    model.eval()
    rows = []
    try:
        with torch.no_grad():
            for start in range(0, len(records), batch_size):
                batch = records[start:start + batch_size]
                inputs = {key: torch.tensor([r[key] for r in batch], dtype=torch.long, device=device)
                          for key in ("input_ids", "attention_mask")}
                logits = model(**inputs).logits
                if not torch.isfinite(logits).all():
                    raise ValueError("验证 logits 包含 NaN/Inf")
                predicted = logits.argmax(dim=-1).cpu().tolist()
                rows.extend({**context, "sample_id": r["sample_id"], "gold_label": r["label"],
                             "predicted_label": pred} for r, pred in zip(batch, predicted))
    finally:
        model.train(was_training)
    export_predictions(destination, rows)
    manifest = {"samples": [{"sample_id": r["sample_id"], "label": r["label"]} for r in records]}
    return recompute_accuracy(destination, manifest, context)
