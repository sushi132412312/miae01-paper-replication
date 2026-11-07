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

- 实际日期与核对的源码位置：待本人填写。
- 独立参数量推导与矩阵形状：待本人填写。
- 我对初始化差异的解释及限制：待本人填写。
- 我修正或补充了什么：待本人填写。
- 对林树辉模块的一项具体互审及回应：待本人填写。

建议提交信息：[2260032761 黄翊轩] 核对LoRA接入位置与初始化差异
