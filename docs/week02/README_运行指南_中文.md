# 第二周代码与检查运行指南

更新：下文主要保留CPU准备阶段的说明。随后用户已确认并完成云端真实工程试跑，最新状态与实际结果见[云端试跑记录](04_云端真实试跑记录_中文.md)。真实试跑仍不等于正式三种子复现。

本文保留最初的CPU准备阶段说明；后续已批准并完成真实工程试跑，最新设置与结果见文末链接。三名成员分别理解、复查与修改，个人贡献以实际提交和审核记录为准。

## 当前范围

目前实现的是可执行的工程准备代码和离线小模型自检。真实RoBERTa-base预训练权重、SST-2数据与云GPU均未用于本阶段检查；真实训练入口受待确认配置保护。

自检使用随机初始化的2层、隐藏维度32的小RoBERTa，8条合成训练输入、6条合成验证输入，每种方法2个更新步；不是论文模型、不是SST-2任务，也不是复现准确率。具体通过情况查看实际生成的 `records/week02/assistant_checks/20261115/test_summary.json` 和 `test_results.txt`，不能仅凭本说明认定通过。

## 目录与三人拟分工

| 对应负责人 | 主要代码 | 本人后续要核对 |
|---|---|---|
| 郑子俊 | src/data.py、src/train.py、tools/prepare_data.py、配置和共同记录工具 | 固定数据、两个种子的含义、公共训练入口、FT执行 |
| 黄翊轩 | src/models.py、环境依赖、tools/check_environment.py、third_party来源 | q/v接入、随机状态、参数冻结、CPU/云GPU差异 |
| 林树辉 | src/evaluation.py、tools/recompute_metrics.py、种子文档纠错 | 预测完整性、复算、样本标准差、数据配对 |

这些是成果的交接方向，不是三名成员已独立实现的声明。大文件夹中三份个人任务包包含各自待合并代码、助手执行证据和本人复核清单；完整集成代码在本仓库。

## 环境安装

本地检查使用Python 3.12和CPU版PyTorch。先在仓库目录创建隔离环境：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.cpu.lock.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements.txt`固定直接依赖；`requirements.cpu.lock.txt`记录本次实际安装的全部包，限定本次Windows/Python3.12 CPU环境。云GPU需要选择与驱动、CUDA适配的构建并重新验证、记录锁定版本，不能直接声称本地环境已验证云GPU。

本次官方索引指向的CPU包下载超时后，使用了同属PyTorch官方的直连wheel并核验SHA256。其他依赖通过清华TUNA的PyPI镜像下载，每个wheel均与PyPI发布元数据中的SHA256对照后离线安装；没有修改系统全局软件源。下载校验记录保存于 `records/week02/dependency_downloads.json`。软件包和wheel缓存不提交GitHub。

在本机已有完整wheel缓存时，可以用 `python -m pip install --no-index --find-links wheel_cache -r requirements.txt` 离线重建；换电脑仍需按上面的联网安装步骤获取包。本次下载器保存在大文件夹的 `outputs/download_week02_wheels.py`，属于安装辅助工具，不是实验代码或训练成绩。

## 运行环境检查与全部自检

在仓库根目录执行，输出目录使用新的名字，工具会拒绝覆盖既有运行：

```powershell
.\.venv\Scripts\python.exe tools/check_environment.py --output local_runs/environment_cpu.json
.\.venv\Scripts\python.exe tools/run_checks.py --output local_runs/my_selfcheck
```

`run_checks.py`会主动打开Hugging Face离线模式，运行数据/指标边界检查和小模型结构检查，再保存FT与LoRA各一次合成运行。输出包括：

- test_summary.json、test_results.txt：实际测试数量、错误和具体用例。
- synthetic_ft/、synthetic_lora/：每种方法的配置、初始参数、参数检查、每步输入ID/loss、末步预测和指标。
- independent_recount.json：从CSV重新计数，并将只有一个种子的正式汇总标为未完成。
- execution_context.json：实际执行者、时间、父提交、工作区状态与执行代码校验值。代码尚未提交时工作区状态会如实保留，不能把父提交SHA当成本次代码已经存在于GitHub的证明。

只复算某次运行的准确率时，不需要PyTorch，系统Python标准库即可：

```powershell
python tools/recompute_metrics.py --predictions local_runs/my_selfcheck/synthetic_ft/predictions.csv --manifest local_runs/my_selfcheck/synthetic_ft/validation_manifest.json --output local_runs/my_selfcheck/ft_recount.json
```

原始预测与样本清单须匹配；缺行、重复、标签错误或混入不同检查点会报错。人工测试数据和合成运行不能加入正式结果表。

## 真实模型尚待确认

`configs/pilot.pending.json`保留候选设置，`status=pending_confirmation`，模型/数据revision为空。当前执行以下命令应被明确拒绝，不下载模型或数据，不创建运行目录：

```powershell
.\.venv\Scripts\python.exe -m src.train --config configs/pilot.pending.json --method ft --output local_runs/real_pilot
```

确认模型任务、训练资源、数据规模、学习率和预算后，需核验并固定模型/数据完整revision，形成获组长确认的新配置，再执行真实工程试跑。不能只为绕过检查而修改状态字段。

此入口当前使用FP32、恒定学习率、末步评估，不保存可重载模型权重；预测记录用训练末态张量的校验值标识检查点。原论文的最佳轮次选择、完整训练日程、正式三个种子、扩展与结果图仍需后续实现和验证。本阶段不具备直接声明“完成论文复现”的条件。

## GitHub交付

三人用本人账号实际复查各自模块、补充自己的理解与修改，再如实提交。提交信息包含实际提交者学号和全名，具体互审意见在PR中记录。最终报告另外按课程要求整理贡献声明和AI使用声明。

缓存、数据、模型权重、虚拟环境不上传。当前 `.gitignore` 的 `*.log` 会忽略普通log文件，检查记录采用JSON/CSV/TXT。最终完整报告需要英文；本阶段中文说明用于学习和协作。
