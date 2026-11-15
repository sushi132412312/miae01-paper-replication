"""复用CPU环境检查，再验证已租服务器的真实CUDA运算。"""
import argparse
import json
from pathlib import Path

from check_environment import collect
from src.common import save_json


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = collect()
    if not torch.cuda.is_available():
        raise RuntimeError("云端没有可用CUDA设备；未通过GPU检查")
    value = torch.randn(32, 32, device="cuda", requires_grad=True)
    (value @ value).square().mean().backward()
    torch.cuda.synchronize()
    if not torch.isfinite(value.grad).all():
        raise RuntimeError("GPU梯度不是有限数值")
    record.update({"validated_device": "cuda", "cloud_gpu_connected": True,
                   "execution_location": "user-rented FunHPC instance",
                   "gpu_test": {"matrix_and_backward_passed": True,
                                "device_name": torch.cuda.get_device_name(0),
                                "memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
                                "compute_capability": list(torch.cuda.get_device_capability(0))}})
    save_json(args.output, record)
    print(json.dumps({"gpu_test": record["gpu_test"], "packages": record["packages"],
                      "pip_check_passed": record["pip_check_passed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
