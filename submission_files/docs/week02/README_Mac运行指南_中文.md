# 环境与LoRA模块：Mac运行指南

本模块可以独立运行，不需要缺失的src.common/data/train/evaluation。它只验收环境和LoRA小模型；真实训练及原共同26项仍待完整工程。

## 创建环境

已有本项目.venv时直接使用。新位置可选择以下一种方式，不能移动旧Conda prefix或复用其他项目环境：

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.mac.lock.txt
```

或：

```sh
conda create --prefix "$PWD/.venv" python=3.11 pip -c conda-forge -y
.venv/bin/python -m pip install -r requirements.mac.lock.txt
```

Mac使用PyPI的arm64 wheel，不使用原Windows/Linux的torch==2.9.1+cpu锁文件。本次实测版本和下载来源保存在记录中。

## 一条命令验收本模块

在项目根目录：

```sh
.venv/bin/python tools/run_lora_module_checks.py --output records/week02/my_module_check
# 普通Mac终端可访问MPS时，把真实MPS子集也列为必需
.venv/bin/python tools/run_lora_module_checks.py --mps --output records/week02/my_module_check_mps
```

入口会检查真实环境、直接导入src.models、运行并核对20个具体用例ID，保存JSON/日志。--mps额外要求3项真实MPS模型检查；设备不可用时不会假报通过。module_acceptance_passed只表示本模块验收，不表示原26或论文复现完成。

也可以使用标准库测试命令：

```sh
.venv/bin/python -m unittest discover -s tests -p test_lora_module.py -v
```

全部数据为合成token，无模型/数据下载。Git下载ZIP中没有.git时上下文commit记为null，不影响独立模块运行。

## 可选工具与集成检查

```sh
# 8项核心行为单独生成详细记录
.venv/bin/python tools/check_lora_mac.py --output records/week02/my_core_check
# 真实参数边界探测，观察结果不等于通过数
.venv/bin/python tools/probe_week02_configuration.py --output records/week02/my_parameters.json
# 原交接共同套件：当前应返回非零并记录缺件
sh tools/run_week02_mac.sh --output records/week02/my_shared_check
```

原模块8项位于tests/integration_pending/test_models_pipeline.py，代码未改；完整共同26项需要原生产模块和配置。本入口显式核对真实用例ID，禁止把0用例或Mock当作原套件通过。

52项额外真实回归需要原个人包与指定官方源码作为本地测试夹具；它们不属于默认20项验收的前置条件。10项Mock仅验证控制流程，始终单独计数。需要复跑可在实际Git工作副本中设置WEEK02_HANDOFF_DIR/WEEK02_OFFICIAL_DIR后使用tools/run_week02_independent_tests.py的real/mock分组。

本次执行者是AI，个人复核清单由本人实际复核后填写。
