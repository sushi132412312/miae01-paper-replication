# PR #6 LoRA 技术审查证据（2026-11-22）

审查目标：`sushi132412312/miae01-paper-replication` PR #6，固定提交 `7fcffbb6aa7170a93692abf8a40417ee56f56722`。

## 内容

- `model_results.json`、`checkpoint_tensor_inventory.json`：Q/V 接入、参数与冻结检查。
- `checkpoint_results.json`：最佳权重选择、恢复及两项元数据漂移对照。
- `fresh_replay_summary.json`、`fresh_replay_row_comparisons.csv`：12 次独立回读的逐例记录。
- `pairing_rerun.json`：三种子配对、训练标签翻转与汇总核对。
- `cross_platform.json`：跨平台预测字段一致、权重哈希不相同的事实。
- `regression_test_summary.json`：历史 26 项回归检查通过记录。
- `source_verification.json`、`environment.json`、`pr_snapshot.json`：来源、源码校验及运行环境。

## 重要限制

以上为 Codex 在授权环境中生成并由本包整理的历史记录；本次整理没有重新运行训练、回归或回读实验。合成小模型只证明检查流程，不能替代真实 RoBERTa-base / SST-2 多种子复现。报告中的两项回读校验缺口应在 PR #6 正式合并前由代码作者回应。

本文件夹没有训练源代码、模型权重、机器凭据、`.git` 或外部人员的个人文件。`SHA256SUMS.txt` 仅校验本文件夹中的选定证据。
