# LoRA 论文复现

课程：MIAE01。小组成员：郑子俊2260032070、黄翊轩2260032761、林树辉2260031586。

## 当前阶段

正式训练已完成。2026-11-22 在单张 V100 上完成原始标签与固定翻转 51 条训练标签两种条件，每种条件均有 FT 和 LoRA 的三个种子，共 12 次真实 RoBERTa-base／SST-2 运行。每次使用 512 条训练、872 条完整验证、3 轮。原始标签的验证准确率为 FT 89.14% ± 0.88 个百分点，LoRA 89.72% ± 0.58 个百分点（均值 ± 样本标准差）。

2026-11-28 已补齐并复算三位成员的逐轮证据，表格、图、论文对照及下周分工见 [第四周实验验收](docs/week04/实验收尾_中文.md)。较早的合成检查见 [第三周结果与三人交接](docs/week03/README_本周结果_中文.md)。

早期 RTX 4090 工程试跑使用固定 64 条训练候选、64 条验证、种子 42、每种方法 10 步。FT 与 LoRA 均答对 34/64，全部预测为类别 0。原始指标和局限见 [云端试跑记录](docs/week02/04_云端真实试跑记录_中文.md)，安装与离线检查见 [第二周运行指南](docs/week02/README_运行指南_中文.md)。

林树辉已提交 [评估模块复核](https://github.com/sushi132412312/miae01-paper-replication/blob/69743a591879d26228380ea540dbe4b886fcd0e6/records/week02/recount_verification/README.md)。黄翊轩的 [LoRA 独立模块 PR #5](https://github.com/sushi132412312/miae01-paper-replication/pull/5) 和本组 [试跑集成 PR #4](https://github.com/sushi132412312/miae01-paper-replication/pull/4) 已合并。

## 阅读入口

1. [从启动脚本到训练主程序：中文串讲](docs/week01/00_从脚本到训练主程序_中文.md)
2. [郑子俊建议主责：数据与训练入口](docs/week01/01_数据与训练入口_中文.md)
3. [黄翊轩建议主责：LoRA接入与初始化](docs/week01/02_LoRA接入与初始化_中文.md)
4. [林树辉建议主责：评估统计与日志](docs/week01/03_评估统计与日志_中文.md)
5. [今天三人的任务与提交方式](docs/week01/04_今日三人任务与提交_中文.md)

## 已核对的参考代码

官方仓库：https://github.com/microsoft/LoRA

参考版本：c4593f060e6a368d7bb5af5273b8e42810cdef90。第一周检查SST-2脚本、任务主程序、RoBERTa模型接入和LoRA线性层；第二周引入未改动的官方loralib，保留MIT许可证与[来源校验](third_party/loralib_source.json)，另写现代Transformers训练入口，未沿用上游旧版NLU训练栈。

选择复现论文 Table 2 中的 RoBERTa-base／SST-2，并按附录 Table 9 对照训练设置。课程允许声明后的缩小规模，要求至少三个种子、一项扩展、英文最终报告、展示和持续 Git 周更。正式实跑使用 [项目训练提交 7fcffbb6](https://github.com/sushi132412312/miae01-paper-replication/tree/7fcffbb6aa7170a93692abf8a40417ee56f56722)。包括训练、数据、模型与评估的源文件校验值见 [部署清单](records/week03/seed42_20261122/deployment_manifest.json)，固定模型和数据 revision 见各次 `config.json`。

## 复算和复跑

先在仓库根目录用 Python 3.10 或更新版本复算 12 次真实运行。汇总报告和制图数据位于 [records/week04/20261128](records/week04/20261128)。

~~~bash
python tools/audit_v100_results.py --output local_runs/verified_v100.json
~~~

要从模型重新训练，使用匹配 CUDA 12.6 的 NVIDIA GPU 和 Linux conda 环境。实跑依赖详见 [环境记录](records/week03/seed42_20261122/environment_gpu.json)。执行以下命令安装、预检并运行完整矩阵：

~~~bash
conda create -n miae01-v100 python=3.10.12 pip -y
conda activate miae01-v100
python -m pip install "torch==2.7.1+cu126" --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r records/week03/seed42_20261122/requirements.v100.local.lock.txt
python -m pip check
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_DISABLE_XET=1
python tools/check_cloud_environment.py --output local_runs/reproduction_environment.json
for condition in clean noise; do
  config="records/week03/seed42_20261122/${condition}_ft_seed42/config.json"
  for seed in 42 43 44; do
    for method in ft lora; do
      run="local_runs/reproduction/${condition}_${method}_seed${seed}"
      python -m src.train --config "$config" --method "$method" --train-seed "$seed" --device cuda --output "$run" || exit 1
      python -m src.replay --run "$run" --output "$run/replay.json" || exit 1
    done
  done
done
python -m src.paired --runs local_runs/reproduction/* --seeds 42 43 44 --output local_runs/reproduction_summary.json
~~~

首次运行按配置下载固定版本模型与数据。默认 Hugging Face；本次 V100 环境的实际镜像与哈希见 [资源校验](records/week03/seed42_20261122/verified_assets.json)。以 V100 实测为参考，一次训练进程约 62–72 秒，首次设置与下载、权重回读另计。每个目录会生成 `metrics.json`、逐轮和最佳轮次的预测、样本清单、模型权重及回读记录。缓存和权重存于本地运行环境。

## 第一周源码核对记录

只读核对工具：[tools/week01_readonly_audit.py](tools/week01_readonly_audit.py)。仅使用Python标准库，读取官方源码文本，记录校验值和源码位置，进行明确标记的人工教学计算。它不导入PyTorch、不调用GPU、不下载数据或模型、不训练模型。

- [数据与入口：助手执行的静态核对记录](records/week01/assistant_data_audit.json)
- [LoRA：助手执行的静态核对记录](records/week01/assistant_lora_audit.json)
- [统计：助手执行的教学核对记录](records/week01/assistant_statistics_audit.json)

在自己的电脑上复查时，先下载同一版官方源码，然后运行：

~~~text
python tools/week01_readonly_audit.py --repo "官方源码所在目录" --part data --actor "本人姓名与学号"
python tools/week01_readonly_audit.py --repo "官方源码所在目录" --part lora --actor "本人姓名与学号"
python tools/week01_readonly_audit.py --repo "官方源码所在目录" --part statistics --actor "本人姓名与学号"
~~~

可用--output指定保存位置。相对源码路径、SHA256和生成者保存在记录中；机器记录只是证据的一部分，个人还应说明看过什么、如何核对、发现和修改了什么。

## 后续执行

使用已验收的实验表、图和原始证据撰写英文报告、整理展示材料并互审。按 [第四周分工](docs/week04/实验收尾_中文.md) 完成最终检查后发布 v1.0。

每条Git提交信息包含实际提交者学号和全名；三位成员分别使用本人账号。现有LICENSE保留；如后续引入上游源码，保留相应原版权和许可证。
