# LoRA论文复现：中文源码阅读与方案准备

课程：MIAE01。小组成员：郑子俊2260032070、黄翊轩2260032761、林树辉2260031586。

## 当前阶段

2026-11-15已在租用RTX4090上完成RoBERTa-base／SST-2真实工程试跑：固定64条训练候选、64条验证，seed42，FT与LoRA各10个更新步。两种方法均答对34/64，准确率53.125%，本次均输出类别0，未展示超过常数基线的分类效果。这是流程验证，尚未完成正式三种子论文复现。

最新成果：[云端真实试跑记录、证据和局限](docs/week02/04_云端真实试跑记录_中文.md)。实际训练、环境及最终26项代码检查均有记录；本次提交集成代码和运行证据，具体复核意见通过PR保留。

准备阶段入口：[第二周CPU环境、代码与检查运行指南](docs/week02/README_运行指南_中文.md)。其中的小模型合成自检与后续真实任务试跑分别保存，不能混入论文结果表。

组长已读完官方SST-2启动脚本。林树辉已于11月15日提交[评估模块复核](records/week02/recount_verification/README.md)，黄翊轩的[LoRA独立模块PR #3](https://github.com/sushi132412312/miae01-paper-replication/pull/3)待审。成员继续用本人账号提交实际核验与修改，集成提交与个人贡献分别记录。

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

## 可以复查的今天成果

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

云端环境与FT/LoRA工程试跑已通过。下一步锁定正式协议，完成三个种子的主实验和一项扩展；最佳轮次/训练后权重保存、完整图表、差异分析和英文报告仍需后续实现与核验。最终v1.0在完成检查后发布。

每条Git提交信息包含实际提交者学号和全名；三位成员分别使用本人账号。现有LICENSE保留；如后续引入上游源码，保留相应原版权和许可证。
