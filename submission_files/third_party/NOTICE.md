# 第三方来源与实现边界

`loralib/__init__.py`、`layers.py`、`utils.py` 原样提取自用户下载的 Microsoft LoRA ZIP，版本为 `c4593f060e6a368d7bb5af5273b8e42810cdef90`。三个文件的 Git blob SHA 与 GitHub 该固定版本的目录记录一致，校验值保存在 `loralib_source.json`；原许可证保存在 `LICENSE.loralib.md`，文件内版权声明保留。

本组的 `src/models.py` 在现代 Hugging Face Transformers 上替换 RoBERTa Query/Value 投影，调用上述官方 Linear；不直接使用原仓库自带的旧 Transformers NLU 全栈。为便于冻结权重核验，设置 `merge_weights=False`。实际 A 为 Kaiming uniform、B 为零。优化器、环境、训练预算和数据规模均须作为相对论文的差异记录。

相关来源：

- https://github.com/microsoft/LoRA/tree/c4593f060e6a368d7bb5af5273b8e42810cdef90/loralib
- https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/LICENSE.md
- https://huggingface.co/docs/transformers/v4.57.3/model_doc/roberta

新模块与接入代码由 AI 在组长授权下编写和检查，尚不表示三名学生分别独立实现或审核。
