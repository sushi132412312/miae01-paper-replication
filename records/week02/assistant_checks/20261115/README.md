# 2026-11-15 · 助手执行记录 + 林树辉本人复核

本目录是 week02「预测复算与种子纠错」的执行记录包。执行者为 AI 助手（组长授权），
**本人复核由林树辉于 2026-11-15 完成**，结论写在 `docs/week02/03_预测复算与种子纠错_中文.md` 第 4 节。

## 文件

| 文件 | 来源 / 含义 |
|---|---|
| `validation_manifest.json` | 6 条 synthetic 样本清单（`subset_seed=2026`，标签 3:3 平衡） |
| `recount_inputs/ft/predictions.csv` | FT 的逐例预测（7 列：run_id, method, seed, checkpoint, sample_id, gold_label, predicted_label） |
| `recount_inputs/ft/metrics.json` | FT 的保存指标（count=6, correct=3, 50.0%） |
| `recount_inputs/lora/predictions.csv` | LoRA 的逐例预测 |
| `recount_inputs/lora/metrics.json` | LoRA 的保存指标（count=6, correct=3, 50.0%） |
| `independent_recount.json` | **本次复核结果**：从 CSV + 清单重算，逐项与保存指标比对 |
| `test_summary.json` | `run_checks` 的 26 项测试汇总（全部通过，`synthetic_only=true`） |
| `test_results.txt` | unittest 原始输出 |
| `handoff_status.json` | 交接状态：`student_independently_verified=false`、`real_training_status=pending_user_configuration` |
| `pilot_metrics_ft.json` | 云端流程试跑的 FT 结果（64 条验证、34 正确、53.125%） |
| `downloaded_file_hashes.json` | 试跑时下载的数据与权重（GLUE/SST-2、roberta-base）的字节数与 SHA256 |

## 复算方法（可独立重跑）

```python
import csv, json
man = json.load(open("validation_manifest.json", encoding="utf-8"))
expected = {s["sample_id"]: s["label"] for s in man["samples"]}
rows = list(csv.DictReader(open("recount_inputs/ft/predictions.csv", encoding="utf-8-sig")))
correct = sum(int(r["predicted_label"]) == expected[r["sample_id"]] for r in rows)
print(correct, "/", len(rows), "=", correct / len(rows))
```

## 复核得到的两点结论

1. **复算一致**：FT 与 LoRA 都是 3/6 = 50.0%，与各自 `metrics.json` 记录完全一致；6 条样本无缺失、无重复，金标签与清单一致。
2. **成绩不可用**：FT 的 6 条预测全部为 0，LoRA 全部为 1——两个模型都坍缩到单一类别，
   在 3:3 平衡样本上 50% 恰好等于随机水平。云端 10 步流程试跑同样是两个方法都只输出一类（34/64）。
   因此这些结果**只证明管道跑通，不构成任何复现成绩**，不进入 Paper vs Ours。

## 待办

- 正式训练设置（学习率、轮数、LoRA 目标模块、子集大小）需组长确认；
- 真实三个种子 42/43/44 与完整验证集的正式实验尚未开始；
- 正式结果出来前，报告里不得引用本目录的任何准确率作为复现结论。
