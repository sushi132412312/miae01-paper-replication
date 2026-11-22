# LoRA 论文复现

课程：MIAE01。小组成员：郑子俊2260032070、黄翊轩2260032761、林树辉2260031586。

## 当前阶段

2026-11-22 补齐按轮评估、最佳权重回读、固定标签翻转和三种子配对复算。26 项回归检查通过，12 次本机合成演练完成。正式 SST-2 训练规模和 GPU 资源由组长定稿。

本周看 [第三周结果与三人交接](docs/week03/README_本周结果_中文.md) 和 [实验协议与运行指南](docs/week03/01_实验协议与运行指南_中文.md)。

上周已在 RTX 4090 完成固定 64 条训练候选、64 条验证、种子 42、每种方法 10 步的真实试跑。FT 与 LoRA 均答对 34/64，全部预测为类别 0。原始指标和局限见 [云端试跑记录](docs/week02/04_云端真实试跑记录_中文.md)，安装与离线检查见 [第二周运行指南](docs/week02/README_运行指南_中文.md)。

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

本次真实工程试跑采用RoBERTa-base / SST-2，已获组长确认；老师没有指定必须采用这一格实验。正式训练规模、轮次及扩展设计仍需确定。老师允许声明后的缩小规模，要求至少三个种子、一项扩展、英文最终报告、展示和持续Git周更。

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

锁定正式协议后，执行三种子主实验和一项扩展，收集逐例预测、统计汇总及差异分析所需证据。随后整理完整图表、英文报告和展示材料，并在最终检查完成后发布 v1.0。

每条Git提交信息包含实际提交者学号和全名；三位成员分别使用本人账号。现有LICENSE保留；如后续引入上游源码，保留相应原版权和许可证。
