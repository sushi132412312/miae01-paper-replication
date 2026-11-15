# 第二周：环境与LoRA独立模块交付

对应成员：黄翊轩2260032761，GitHub hyxuan26。本文最初由 AI 辅助检查和起草，现已由本人结合 Microsoft LoRA 官方指定版本源码完成独立静态核验。本交付范围是环境与LoRA模块。

## 已完成的模块

- `src/models.py` 可直接导入，调用本模块拥有的 `src/model_support.py`，不再依赖缺失的src.common或诊断命名空间替换。
- `model_support.py` 提供明确的种子/参数验证与张量哈希，不伪装成原共同common模块。正常float32的哈希格式保留与前轮Mac诊断的兼容性，并支持scalar/bfloat16与重复名称拒绝。
- q/v接入、Key普通Linear、A的Kaiming uniform/B全零、alpha/r、主干冻结、分类头训练和merge_weights=False的核心数学路径保持不变。官方loralib源文件/许可证字节未改。
- 修复真实探测发现的输入边界：LoRA rank必须为正整数，alpha必须为有限正数，拒绝bool及缺失值；seed限定在NumPy支持的32位范围；config必须为映射，synthetic必须为bool。非法请求在构造模型、消耗随机状态之前拒绝。
- 实际预训练分支仍要求完整项目的src.data验证；缺件时给出明确RuntimeError，不自动下载或猜测真实训练设置。

## 环境与真实验收

实际执行环境：Python3.11.18，torch2.9.1，transformers4.57.3，datasets4.4.1，numpy2.2.6；项目专属.venv，未复用其他项目环境。宿主机实测Apple M5、16GiB、macOS26.5、arm64，与用户最初给出的macOS15/24GB不同，记录按实测保留。CPU/MPS可用，CUDA不可用；pip check通过。

| 验证范围 | 最终结果 | 证据 |
|---|---|---|
| 直接导入的模块验收 | 20/20，0失败/错误/跳过，无依赖替换 | module_release_20261115/acceptance/module_summary.json |
| 真实MPS小模型子集 | 3/3：前向、配对/CPU容差、反向与冻结 | acceptance/mps_model_tests.json |
| 现有模块与工具真实回归 | 52/52，0 expected failure | regression/real_tests.json |
| Mock控制流程回归 | 10/10，单独统计 | regression/mock_tests.json |
| 原共同26项（含原模块8项） | 仍缺完整源码，未通过集成验收 | integration_pending/run_summary.json |

以上相对证据路径均位于records/week02/module_release_20261115。Mock不作为设备或集成通过证据；原26与本模块20不能相互替代。所有模型输入是合成token，模型为随机初始化小RoBERTa；没有加载预训练checkpoint/SST-2，没有论文成绩或正式三种子结果。

小模型参数：FT总量/可训练23,490；LoRA总量24,514，低秩1,024，分类头1,122，可训练2,146，冻结22,368。低秩公式为2×2×(32×4+4×32)。CPU初始FT/LoRA logits差0；一步优化后冻结权重不变，B与分类头有更新，第一步A梯度为零符合B初始为零。12层/768/rank8的294,912仍只是不含分类头的理论量。

## 工具修复

1. 交接校验在历史准确率复算失败时返回非零；空预测文件不除零，非二分类预测即使准确率数值碰巧不变也拒绝。
2. 原共同检查入口要求26个唯一、具体且匹配历史清单的成功用例ID；只有退出0、文件存在、重复ID或0用例均不能宣告完成。10项Mock包含成功/失败/重复/超时分支。
3. Git上下文在ZIP预览中缺失时记录null，不伪造commit。独立模块验收无需Git仓库才能运行。

## 与原交接包的关系

原43项文件校验和早期诊断保留为历史证据。原41文件补丁已原样归档；本版源码在该增量上修复缺陷并补足独立导入。不要再把早期delivery_sha256.json用于校验最新源码，也不要声称models.py仍逐字节等同原包。

原test_models_pipeline.py字节未改，移至tests/integration_pending/，避免把未具备依赖的共同套件混入独立模块默认验收。原26验收继续显式记录阻塞。第一周文档/记录、现有统计复核和许可证不改写。

本模块已具备可提交的源码、环境说明、直接验收入口与真实记录。成员本人清单未代填。未执行Commit/Push/创建PR/Merge。
