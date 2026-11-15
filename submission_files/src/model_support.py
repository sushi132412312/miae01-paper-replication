"""Owned helpers for the standalone environment/LoRA module.

This is not a reconstruction of the missing shared src.common module.
Float32 hashes retain the format used by the earlier Mac diagnostics.
"""
import hashlib
import json
import math
import random
from collections.abc import Mapping
from numbers import Real


def validate_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed 必须是 [0, 2**32) 内的整数，不接受 bool")


def validate_model_request(method, seed, config, synthetic):
    if method not in ("ft", "lora"):
        raise ValueError("method 只能为 ft 或 lora")
    validate_seed(seed)
    if not isinstance(config, Mapping):
        raise ValueError("config 必须是映射对象")
    if type(synthetic) is not bool:
        raise ValueError("synthetic 必须是 bool")
    if synthetic and config.get("status") != "synthetic_selfcheck":
        raise ValueError("合成检查必须使用专门配置")
    if method == "lora":
        rank = config.get("lora_r")
        alpha = config.get("lora_alpha")
        if type(rank) is not int or rank <= 0:
            raise ValueError("LoRA lora_r 必须是正整数，不接受 bool")
        try:
            valid_alpha = not isinstance(alpha, bool) and isinstance(alpha, Real) and math.isfinite(alpha) and alpha > 0
        except (OverflowError, TypeError):
            valid_alpha = False
        if not valid_alpha:
            raise ValueError("LoRA lora_alpha 必须是有限正数")


def seed_everything(seed):
    validate_seed(seed)
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tensor_hash(named_tensors):
    import torch
    items = list(named_tensors)
    names = [name for name, _ in items]
    if len(names) != len(set(names)):
        raise ValueError("张量名称不得重复")
    digest = hashlib.sha256()
    for name, tensor in sorted(items, key=lambda item: item[0]):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
