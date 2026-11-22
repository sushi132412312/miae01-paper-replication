# week03 · seed44 正式缩小规模实验：复核记录

对象：`林树辉_正式实验_seed44_20261122` 交付包（V100 正式缩小规模复现，seed 44），
代码基线为 PR #6 的 `7fcffbb6aa7170a93692abf8a40417ee56f56722`。

## 设置（来自 `结果汇总_中文.md` 与各 run 的 `metrics.json`）

| 项 | 值 |
|---|---|
| 模型 / 数据 | FacebookAI/roberta-base `e2da8e2f` / nyu-mll GLUE SST-2 `bcdcba79` |
| 训练 / 验证 | 固定 512 条 / 完整 872 条（subset_seed=2026） |
| 轮数 / 步数 | 3 轮，每轮 64 步，共 192 步 |
| 批次 / 长度 | batch 8，max_length 128，FP32 |
| 学习方法率 | FT 2e-5，LoRA 5e-4（q/v，r=8，α=8，weight_decay 0.1） |
| 选轮规则 | 验证准确率最高，同分取最早轮次 |
| 条件 | `clean`（原始标签）与 `noise`（固定翻转 51 条训练标签，label_noise_seed=2027） |

## seed44 四个运行（本轮独立复算）

| 运行 | 最佳轮次 | 验证答对 | 准确率 |
|---|---:|---:|---:|
| clean_ft_seed44 | 3 | 769 / 872 | 88.1881% |
| clean_lora_seed44 | 3 | 783 / 872 | 89.7936% |
| noise_ft_seed44 | 3 | 733 / 872 | 84.0596% |
| noise_lora_seed44 | 3 | 757 / 872 | 86.8119% |

与交付的 `checkpoint_selection.json`、`metrics.json` 以及
`paired_summary_all_seeds.json` 的 seed44 行**完全一致**。

## 复核怎么重跑

```bash
python records/week03/verify_seed44.py records/week03/seed44_20261122 records/week03/verify_seed44.json
```

脚本只用 Python 标准库：读取每个运行的 `validation_manifest.json` 与
`epochs/00X/predictions.csv`，重算每轮准确率、按"同分取最早"选最佳轮次，
再与记录比对；同时检查 clean 与 noise 的训练清单差 51 条标签、FT 与 LoRA 的配对。

## 目录内容

```
seed44_20261122/
├── 结果汇总_中文.md               # 组内 12 次运行的汇总（三种子 × 两条件 × 两方法）
├── paired_summary_all_seeds.json  # 机器可读的三种子配对汇总
├── clean_ft_seed44/
│   ├── metrics.json               # 选轮后的指标与协议哈希
│   ├── checkpoint_selection.json  # 三轮验证记录与最佳权重 SHA-256
│   ├── train_manifest.json        # 512 条训练样本清单
│   ├── validation_manifest.json   # 872 条验证样本清单
│   ├── model_initialization.json  # 主干/分类头初值哈希
│   └── epochs/00{1,2,3}/predictions.csv
└── clean_lora_seed44/ … noise_ft_seed44/ … noise_lora_seed44/（同上）
```

## 边界

- 本次只复核 **seed44**；seed 42/43 的逐例记录未包含在本交付包中，其数值以
  `paired_summary_all_seeds.json` 为准，未逐轮复算。
- 云端最佳权重（`.safetensors`）不在仓库内，只有 `checkpoint_selection.json` 里的
  大小与 SHA-256；权重本身未回读验证。
- 这些是本组缩小规模（512/872/3 轮）的验证准确率，与论文 Table 2 的设置有差异，
  不能直接当作论文结果。
