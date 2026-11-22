# 郑子俊 · 种子 42 正式实验记录

学号 2260032070。记录来自 V100 上完成的 RoBERTa-base／SST-2 缩小规模实验。训练使用 [PR #6 的固定版本 7fcffbb](https://github.com/sushi132412312/miae01-paper-replication/tree/7fcffbb6aa7170a93692abf8a40417ee56f56722)。实际代码、模型和数据版本保存在 [部署清单](deployment_manifest.json) 与 [资源校验记录](verified_assets.json)。

512 条训练样本由子集种子 2026 固定抽取，评估使用完整 872 条验证样本。每次训练 3 轮，批次大小 8，每轮 64 步，共 192 步。最大长度 128，FP32。FT 学习率 2e-5，LoRA 学习率 5e-4，Query/Value 接入 r=8、alpha=8。扩展用种子 2027 固定翻转 51 条训练标签，保留相同训练样本 ID、验证集及预算。每次取验证准确率最高的一轮，同分取最早一轮。

| 条件 | 方法 | 最佳轮次 | 答对数 | 验证准确率 |
|---|---|---:|---:|---:|
| 原始标签 | [FT](clean_ft_seed42/metrics.json) | 3 | 779 / 872 | 89.3349% |
| 原始标签 | [LoRA](clean_lora_seed42/metrics.json) | 3 | 777 / 872 | 89.1055% |
| 翻转 51 条训练标签 | [FT](noise_ft_seed42/metrics.json) | 3 | 759 / 872 | 87.0413% |
| 翻转 51 条训练标签 | [LoRA](noise_lora_seed42/metrics.json) | 3 | 756 / 872 | 86.6972% |

固定标签翻转下，FT 和 LoRA 比各自干净条件分别低 2.2936 和 2.4083 个百分点。这里报告的是种子 42 的单次结果；三种子汇总需要合并另外两个种子。

## 复核入口

每个运行目录保存 `config.json`、训练步次、选中轮次与三轮逐例预测、样本清单、参数检查、运行日志及最佳权重回读记录。四次回读均在 V100 上重新加载了最佳权重，并与保存的逐例预测对照一致。最佳权重保存在云端数据盘，文件 SHA-256 见各运行的 `checkpoint_selection.json`。

在仓库根目录用 Python 标准库复算任意一次准确率。例如原始标签 FT 的命令如下。

```bash
python tools/recompute_metrics.py --predictions records/week03/seed42_20261122/clean_ft_seed42/predictions.csv --manifest records/week03/seed42_20261122/clean_ft_seed42/validation_manifest.json
```

将输入换成该运行的 `epochs/001/predictions.csv` 至 `epochs/003/predictions.csv`，即可逐轮复算并核对最佳轮次。提交前用 PR #6 的配对汇总程序从原始记录复核了这 12 份逐轮预测，汇总输出见 [种子 42 复算与配对核对](seed42_recount.json)。训练环境见 [GPU 检查](environment_gpu.json) 与 [本次 Python 依赖](requirements.v100.local.lock.txt)。
