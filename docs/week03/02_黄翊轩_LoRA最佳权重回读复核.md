# 第三周 LoRA 技术复核报告

> **核验来源及范围：**本报告由黄翊轩（2260032761，GitHub `hyxuan26`）负责阅读、确认并提交技术审查。

**核验结论：**　Q/V 接入、主干冻结、分类头训练、最佳权重保存和同种子配对均通过本次 CPU 合成演练。12 份最佳权重在训练结束后分别用新进程回读，72 条预测全部一致，其中 LoRA 为 36/36。**建议在正式实验前补强独立回读入口**。隔离副本上的两个对照表明，配置或最佳轮次记录发生漂移时，单独执行 `src.replay` 可能仍报告成功。

| 复核对象 | 固定值 |
|---|---|
| 仓库与分支 | `sushi132412312/miae01-paper-replication` · PR #6 · `week03/epoch-workflow-20261122` |
| 实测提交 | `7fcffbb6aa7170a93692abf8a40417ee56f56722`，基线 `main` 为 `69743a591879d26228380ea540dbe4b886fcd0e6` |
| 对照源码 | Microsoft LoRA 的仓库指定提交 `c4593f060e6a368d7bb5af5273b8e42810cdef90` |
| 环境与范围 | 2026-11-22，macOS arm64，CPU，Python 3.11.18，PyTorch 2.9.1，Transformers 4.57.3，safetensors 0.8.0。离线合成 token，训练 8 条、验证 6 条，batch 2，3 轮，每次更新 12 步 |

独立目录中的源码以 detached HEAD 签出。四个指定模块均按上述提交审阅，其中 `models.py` 沿用基线代码。审阅只在本地完成，正式 SST-2 训练次数为 0。[PR #6](https://github.com/sushi132412312/miae01-paper-replication/pull/6) · [本次源码快照](https://github.com/sushi132412312/miae01-paper-replication/tree/7fcffbb6aa7170a93692abf8a40417ee56f56722)

## 一、可能需要处理的两项发现

| 优先级与位置 | 复现结果 | 建议修复与验收 |
|---|---|---|
| P2 · `src/replay.py` 第 21–23 行，最佳轮次 | 在同一运行的临时副本中，将 `metrics.best_epoch` 和 `checkpoint_selection.best_epoch` 同时从 1 改为 2。权重、逐轮记录和预测均保持源值。回读退出码为 0，输出同时带有 `best_epoch=2` 和 `checkpoint=epoch-1-step-4`。 | 从逐轮 CSV 复算正确数，确定首个最高分，再核对轮次、步数和检查点名称。加入上述副本作为回归用例，期望回读失败。 |
| P2 · `src/replay.py` 第 24–42 行，LoRA 缩放 | 在另一副本中只把 `config.lora_alpha` 从 8 改为 16。相同权重得到的最大绝对 logit 差约为 `1.13e-5`，6 个 argmax 标签仍相同，回读也以 0 退出。 | 建模前用训练端相同口径重算协议 SHA256，与 `metrics.protocol_sha256` 核对，继续单独核对训练种子。验收用例应在推理前报告配置不一致。 |

两个对照均被现有 `src.paired.read_run` 拒绝。可把其中的协议绑定和首个最佳轮次校验抽成共用检查，供回读和汇总调用。当前原始运行的记录已通过两条路径复算。[对照结果](../../records/week03/hyxuan26_pr6_review_20261122/checkpoint_results.json) · [固定版本回读代码](https://github.com/sushi132412312/miae01-paper-replication/blob/7fcffbb6aa7170a93692abf8a40417ee56f56722/src/replay.py#L20-L51)

## 二、模型、冻结范围和参数量

将本仓库的 `loralib/layers.py`、`utils.py`、`__init__.py` 及许可证与 Microsoft 指定提交逐字节比对，4 个文件的 SHA256 和 Git blob ID 均相符。上游 RoBERTa 将 LoRA 接入 query/value，key 仍为普通线性层。上游 GLUE 入口在 LoRA 模式下训练适配器和任务头。本仓库在现代 Transformers 上替换同名投影，复制原权重与偏置，并显式设置 `merge_weights=False`。[上游 RoBERTa](https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/examples/NLU/src/transformers/models/roberta/modeling_roberta.py#L165-L175) · [上游 GLUE](https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/examples/NLU/examples/text-classification/run_glue.py#L407-L418) · [本仓库 models.py](https://github.com/sushi132412312/miae01-paper-replication/blob/7fcffbb6aa7170a93692abf8a40417ee56f56722/src/models.py#L28-L46)

| 合成模型实测 | 计算依据 | 标量参数 |
|---|---|---:|
| Q/V 适配器 | 2 层 × 2 个投影 × (4 × 32 + 32 × 4) | 1,024 |
| 分类头 | dense 权重与偏置、out_proj 权重与偏置，即 32 × 32 + 32 + 2 × 32 + 2 | 1,122 |
| LoRA 全部可训练项 | 8 个 A/B 张量 + 4 个分类头张量 | **2,146** |
| LoRA 冻结项 / 模型总量 | 原始主干冻结，适配器新增 | 22,368 / 24,514 |
| 同架构 FT 全部可训练项 | 全参数更新 | 23,490 |

在 42、43、44 三个种子上，同种子 FT 与 LoRA 的共同张量初值和构造后的 CPU RNG 状态一致。B 初始全零，适配器初始贡献为零，测试批次初始 logits 逐元素相同。单独的两步梯度探针使用恒定非零学习率，第一步检测到 B 与分类头更新，第二步 A 也更新。冻结项始终无梯度。完整矩阵的 6 个 LoRA 运行中，冻结张量训练前后哈希均一致，每份最佳权重仅含指定的 12 个张量。[三种子梯度与数值记录](../../records/week03/hyxuan26_pr6_review_20261122/model_results.json) · [六份权重的张量清单](../../records/week03/hyxuan26_pr6_review_20261122/checkpoint_tensor_inventory.json)

候选 RoBERTa-base 的口径可另行计算。12 层、隐藏维度 768、秩 8 时，Q/V 适配器共 294,912 个参数，加上分类头 592,130 个，合计 887,042 个可训练参数。本周的实际更新和回读使用上表中的两层模型。

## 三、最佳轮次、权重与逐条预测

`train.py` 第 103–115 行每轮按验证集正确数比较，仅在严格提高时替换最佳权重，因而同分保留最早一轮。`checkpoints.py` 先写临时文件再替换。恢复时先核对文件名、字节数和 SHA256，再核对全部张量名称、数量、形状和 dtype，全部通过后才写入模型。修改摘要、缺失一个张量、形状错误或 dtype 错误的四个独立对照均被拒绝，目标模型参数未发生部分更新。[训练选轮](https://github.com/sushi132412312/miae01-paper-replication/blob/7fcffbb6aa7170a93692abf8a40417ee56f56722/src/train.py#L103-L131) · [保存与恢复](https://github.com/sushi132412312/miae01-paper-replication/blob/7fcffbb6aa7170a93692abf8a40417ee56f56722/src/checkpoints.py#L16-L47)

| 合成矩阵（每格均含种子 42、43、44） | 运行数 | 各种子第 1 / 2 / 3 轮答对数 | 最佳轮次 | 独立回读 |
|---|---:|---|---:|---:|
| 干净训练标签 · LoRA | 3 | 3 / 3 / 3，各轮分母 6 | 1 | 18/18 |
| 翻转 2 条训练标签 · LoRA | 3 | 3 / 3 / 3，各轮分母 6 | 1 | 18/18 |
| 干净训练标签 · FT | 3 | 3 / 3 / 3，各轮分母 6 | 1 | 18/18 |
| 翻转 2 条训练标签 · FT | 3 | 3 / 3 / 3，各轮分母 6 | 1 | 18/18 |

训练进程退出后，每次运行各启动一个新的 `python -m src.replay` 进程。另用新进程从 safetensors 恢复参数，直接对 logits 取 argmax 写出 CSV。逐行核对运行 ID、方法、种子、检查点、样本 ID、金标签和预测标签，同时比较源 CSV、所选轮 CSV 与新 CSV 的 SHA256。三个文件在每次本机运行内均相同。另设 LoRA 受控评分 2、5、5，保留真实合成训练更新，结果选中第 2 轮。恢复后对模型全部 49 个参数张量逐一比较，与第 1、2、3 轮分别有 12、0、12 个张量不同。[逐条对照](../../records/week03/hyxuan26_pr6_review_20261122/fresh_replay_row_comparisons.csv) · [每份权重和 CSV 的 SHA256](../../records/week03/hyxuan26_pr6_review_20261122/fresh_replay_summary.json) · 各进程 PID、命令、时间与退出码记录保存在本地原始交付包中

与 PR 中 Windows CPU 演练对照，12 组最佳轮次、逐轮 CSV 和最佳预测 CSV 的字段均一致。本次 macOS 重训得到的 12 份权重文件 SHA256 均与 PR 登记值不同。两台机器的操作系统、架构与 Python 版本有差异，PR 仅保存原权重摘要，现有材料无法定位文件字节差异的来源。这里确认的是本机文件完整性和跨环境的逐条分类结果；6 条合成验证样本不足以判断 SST-2 表现。[两端逐组对照](../../records/week03/hyxuan26_pr6_review_20261122/cross_platform.json)

## 四、种子配对与标签翻转

由 `subset_seed=2026` 独立重建合成 token 和原始标签，由训练种子重建逐轮样本顺序，再用 `label_noise_seed=2027` 抽取翻转位置。12 次运行共核对 144 个训练批次和 216 条逐轮预测。同一训练种子的两种方法、两个标签条件具有相同的主干初值、分类头初值、训练 ID 顺序和更新步数。三个种子的顺序摘要各不相同。

翻转条件固定将 `synthetic-train:1` 和 `synthetic-train:3` 的标签从 1 改为 0，其余 6 条训练标签及全部 6 条验证标签保持原值。各条件各方法三种子的最佳验证准确率均为 50.0%，样本标准差为 0 个百分点，所有配对差值为 0。演练中的翻转比例为 2/8。候选正式协议中的 51/512 未在本次执行。[配对、翻转和汇总的独立复算](../../records/week03/hyxuan26_pr6_review_20261122/pairing_rerun.json)

**配对检查的边界。** 缺少一个种子与方法组合时，汇总给出缺项；重复运行、错误翻转 ID、单条运行使用错误的种子顺序均能检出。若把种子 42 的四份记录全部换成种子 43 的顺序，各方法之间仍对齐，汇总会返回 `complete`。若验收还要求证明“确由声明的种子生成”，需从种子重新生成顺序并核对。本次已对原始 PR 记录和本机重跑分别执行此项检查，两套记录都通过。[原始 PR 记录复算](https://github.com/sushi132412312/miae01-paper-replication/tree/7fcffbb6aa7170a93692abf8a40417ee56f56722/records/week03/20261122) · [本仓库配对检查](https://github.com/sushi132412312/miae01-paper-replication/blob/7fcffbb6aa7170a93692abf8a40417ee56f56722/src/paired.py#L70-L78)

## 五、仓库交付与复核

本仓库只新增上述复核文档及筛选出的 JSON/CSV 证据。完整原始审查 ZIP 保留在工作目录作为归档。

这次复核暴露的是独立 `src.replay` 对自洽但失真的元数据检查不足；当前读回路径的正确性和两端预测类别一致已得到合成数据测试支持。正式 SST-2 模型训练及准确率结论仍未完成。
