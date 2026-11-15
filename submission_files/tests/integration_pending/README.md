# 原共同工程的待集成测试

`test_models_pipeline.py` 原样来自个人交接包，SHA256：

`a05521cd29d4ae5bc1e616b12fa2bc20b601a51c4da7697ed04aeab3ce444509`

原8项包含在共同26项内。它依赖尚未提供的 common/data/evaluation/train 和配置，目前不能运行到测试体。独立模块20项验收不是原8/26通过。

默认独立测试入口是 `tools/run_lora_module_checks.py`。显式运行原交接检查：

```sh
sh tools/run_week02_mac.sh --output records/week02/personal_integration_check
```

该入口当前预期返回非零，并在完整工程取得后检查26个唯一用例ID，而不只看退出码。此目录不设Python包初始化，避免将未就绪套件混入独立模块的默认discover；原测试源码没有删除或改写。
