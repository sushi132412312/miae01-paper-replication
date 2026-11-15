# week02 · 评估统计模块本人复核（林树辉 2260031586）

本目录是 2026-11-15 对 week02 交付的评估模块所做的独立复核，可直接重跑，不需要 torch / datasets。

## 内容

| 文件 | 说明 |
|---|---|
| `verify_evaluation.py` | 复核脚本。只依赖 Python 标准库；其中 `recompute_accuracy` / `summarize_seed_results` 与交付件 `src/evaluation.py` 的同名函数逐行一致 |
| `outputs/recount_verification.json` | 四项检查的完整结果（机器可读） |
| `outputs/inputs/*.csv` | 复核所用的预测 CSV：正常 1 份、变异 5 份（删行、复制行、改金标签、非法预测值、混检查点） |
| `outputs/inputs/synthetic_ft_manifest.json` | 6 条样本的清单（`{"samples":[{"sample_id","label"}]}`） |

## 重跑

```bash
cd records/week02/recount_verification
python verify_evaluation.py --out outputs
```

用系统 Python（不装任何第三方包）即可，输出为同一份 JSON。

## 四项检查与结果

| # | 检查 | 结果 |
|---|---|---|
| 1 | 从预测 CSV + 清单复算准确率，与保存指标比较 | 复算 3/6 = 0.5（50.0%），与记录一致 |
| 2 | 删一行 / 复制一行 / 改金标签 / 非法预测值 / 混检查点 | 五种情况全部抛出 `ValueError`，提示信息见 `outputs/recount_verification.json` |
| 3 | FT 与 LoRA 是否共用同一清单 | 样本 ID 与金标签相同，仅 run_id/method/checkpoint 不同 |
| 4 | 不足三种子是否保持未完成 | 1 种子 → `incomplete`；[80,82,84] → mean 82、sample_sd 2.0；重复种子/混协议 → 拒绝 |

## 边界说明

- 附件只提供了试跑的**记录值**（count/correct/accuracy/checkpoint），没有提供原始预测 CSV 与样本清单文件，
  因此第 1 项使用按交付字段规范重建的 6 行预测来复核复算与校验逻辑；第 2–4 项是对交付逻辑的直接检验。
- 这些是 synthetic 试跑结果，不是 SST-2 / RoBERTa-base 的真实成绩，**不进入 Paper vs Ours**。
- 正式实验仍需要：真实预测 CSV + 清单原文件、三个真实训练种子 42/43/44 的结果。
