# LoRA接入与初始化：源码阅读成果草稿

建议核验负责人：黄翊轩，2260032761，hyxuan26。本文由AI辅助检查和起草，本人独立核验与提交待开展。

## 1. LoRA接在模型哪里

在modeling_roberta.py 165-175行，开启apply_lora后，query和value使用lora.Linear；key保持普通nn.Linear。

可理解为：模型注意力模块中选定两类投影，加入低秩更新分支。它不是把整套模型换成另外一个小模型。

源码相对路径：examples/NLU/src/transformers/models/roberta/modeling_roberta.py。

## 2. 低秩更新和参数量

loralib/layers.py的Linear类定义A、B及alpha/r缩放，冻结原权重，并把低秩分支加到原线性层输出。

对12层、隐藏维度768、query/value两处、rank8的分析假设：

    12 × 2 × (768 × 8 + 8 × 768) = 294,912

这是低秩矩阵的理论参数量，不包括分类头，也不是本次执行模型后测出的运行时参数统计。正式实验仍要按实际允许训练的张量统计。

## 3. 新发现：初始化与注释要分别看

论文Section 4描述A随机高斯初始化、B为零。

下载版本的Linear类124-125行实际调用：

    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    nn.init.zeros_(self.lora_B)

因此该实现是A采用Kaiming均匀初始化、B为零。其122行邻近注释写成“B按默认Linear初始化、A为零”，与实际语句相反。阅读时以执行语句判断，并记录这个差异。

B为零时，BA为零，所以初始低秩更新为零。这可以通过代数确认，但不能据此推断不同A初始化分布的最终训练效果一定相同。

## 4. 分类头也要纳入

run_glue.py 385行把lora加入可训练参数关键词。409-418行冻结RoBERTa主干中不符合关键词的参数，而主干以外的参数仍可训练。因此需要单独核对并统计任务分类头，不能把294,912写成所有可训练参数。

## 5. 另一个已有差异

论文表9的RoBERTa-base配置写alpha8，原SST-2脚本写alpha16。最终采用哪个值由组长确定，并在偏差表中说明。判断差异对准确率的影响需要控制其他条件进行验证。

## 6. 可复查证据

- 模型接入：modeling_roberta.py 165-175行。
- 线性层：loralib/layers.py 109-125、144-152行。
- 参数冻结：run_glue.py 379-418行。
- 助手核对记录：records/week01/assistant_lora_audit.json，包含源码位置、理论计数和人工零矩阵演示。

源码链接：https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/loralib/layers.py

## 7. 本人实际核验后填写

### 7.1 本人源码核验记录

**核验日期：** 2026-11-07

**参考代码：** Microsoft LoRA

**Commit ID：** c4593f060e6a368d7bb5af5273b8e42810cdef90

**核验工具：** tools/week01_readonly_audit.py

**核验结果：**

1. 已确认本地 Microsoft LoRA 源码版本与团队指定版本一致。
2. 已成功执行 LoRA 静态核验脚本，并生成 records/week01/my_lora_audit.json。
3. 在 12 层、隐藏维度 768、Query/Value 两处、LoRA rank=8 的假设下，理论低秩矩阵参数量为：12 × 2 × (768 × 8 + 8 × 768) = 294,912
4. 已通过人工小矩阵演示验证：当 B 矩阵为零矩阵时，BA=0，因此初始化时 LoRA 权重更新为零。
5. 核验限制：本次完成的是源码静态检查与理论计算，尚未执行实际模型训练或运行时参数统计。
6. 团队互审：等待其他成员提交模块，之后需要进行独立审查并记录反馈。


### 7.2 LoRA 初始化源码核验

**核验文件：** `loralib/layers.py`

**核验发现：**

1. `self.weight.requires_grad = False` 用于冻结预训练权重。
2. LoRA 矩阵 A 使用 Kaiming Uniform 初始化。
3. LoRA 矩阵 B 使用全零初始化。
4. 当 B=0 时，BA=0，因此初始权重增量为零。
5. 初始化部分的代码注释与实际实现存在不一致。

**核验结论：**

已通过官方源码确认 LoRA Linear 层的权重冻结和初始化策略。本次属于源码静态核验，尚未完成实际模型训练或下游任务性能评估。


### 7.3 Query/Value 接入位置核验

**核验文件：**
`examples/NLU/src/transformers/models/roberta/modeling_roberta.py`

**源码行号：** 165–175

**核验结果：**

1. 当 `config.apply_lora=True` 时，Query 使用 `lora.Linear`，位于第 165–168 行。
2. Key 使用普通 `nn.Linear`，位于第 170 行。
3. 当 `config.apply_lora=True` 时，Value 使用 `lora.Linear`，位于第 172–175 行。
4. LoRA 的秩通过 `config.lora_r` 传递，缩放超参数通过 `config.lora_alpha` 传递。
5. 当 `config.apply_lora=False` 时，Query 和 Value 使用普通 `nn.Linear`。

**核验结论：**

确认该实现支持在 Query 和 Value 投影层接入 LoRA，而 Key 保留普通线性层。
LoRA 是否在实际实验中启用，还需要进一步检查训练配置。

**核验范围：**

本次为源码静态核验，未进行模型训练或运行时验证。



### 7.4 LoRA 训练配置与参数冻结核验

**核验文件：** `examples/NLU/examples/text-classification/run_glue.py`

**参考版本：** `c4593f060e6a368d7bb5af5273b8e42810cdef90`

**源码位置：**

- 第 176–191 行：定义 LoRA 开关及相关超参数。
- 第 353–355 行：将 LoRA 参数传入模型配置。
- 第 369–376 行：根据配置加载预训练模型。
- 第 378–379 行：初始化可训练参数列表并判断是否启用 LoRA。
- 第 385 行：将 `lora` 加入可训练参数筛选列表。
- 第 409–418 行：实现参数冻结及可训练参数筛选。

**核验结果：**

1. `apply_lora` 的默认值为 `False`，实际执行 LoRA 微调时需要启用该配置。
2. `lora_alpha` 和 `lora_r` 的默认值均为 `None`，实际使用时需提供有效参数。
3. 训练程序通过 `AutoConfig.from_pretrained()` 将 LoRA 参数传入模型配置。
4. 当 `apply_lora=True` 时，程序将 `lora` 加入 `trainable_params`。
5. 第 409–418 行通过 `requires_grad` 控制参数的可训练状态。
6. 在仅启用 LoRA 的配置下，RoBERTa 主干中的普通参数被冻结，而名称匹配 `lora` 的参数保持可训练。
7. 分类头等主干之外的参数仍可训练，因此模型实际可训练参数量可能大于理论 LoRA 参数量 294,912。

**核验结论：**

确认了训练程序具有 LoRA 配置传递与选择性参数冻结机制。
在仅使用 LoRA 的配置下，模型可以冻结预训练 Transformer 主干权重，同时训练 LoRA 低秩矩阵及分类头等必要参数。

**核验限制：**

本次属于源码静态核验，尚未执行实际模型训练，也未加载 RoBERTa 模型统计完整可训练参数量。因此不能将本次结果作为模型复现性能的实验证据。
