"""采集实际环境；GPU 硬件存在与当前 PyTorch 是否支持 CUDA 分别记录。"""
import argparse
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import execution_context, save_json


def collect():
    import torch
    from datasets import Dataset
    from transformers import RobertaConfig
    local_dataset = Dataset.from_list([{"sample_id": "artificial:0", "label": 0}])
    data_import_ok = local_dataset.select([0])[0]["label"] == 0
    config_import_ok = RobertaConfig(hidden_size=32).hidden_size == 32
    packages = {}
    for name in ("torch", "transformers", "datasets", "numpy", "huggingface-hub", "tokenizers", "safetensors"):
        packages[name] = metadata.version(name)
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        gpu_info = gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        gpu_info = str(error)
    pip_check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=45)
    result = {"python_executable": sys.executable, "python_version": platform.python_version(),
              "platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
              "packages": packages, "nvidia_smi": gpu_info,
              "torch_cuda_build": torch.version.cuda, "torch_cuda_available": torch.cuda.is_available(),
              "validated_device": "cpu", "cloud_gpu_connected": False,
              "pip_check_passed": pip_check.returncode == 0,
              "pip_check_output": (pip_check.stdout + pip_check.stderr).strip(),
              "torch_cpu_operation_passed": torch.equal(torch.ones(2) + 1, torch.tensor([2., 2.])),
              "datasets_local_operation_passed": data_import_ok,
              "transformers_config_operation_passed": config_import_ok,
              "execution_context": execution_context()}
    if not all(result[key] for key in ("pip_check_passed", "torch_cpu_operation_passed",
                                      "datasets_local_operation_passed", "transformers_config_operation_passed")):
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = collect()
    save_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("python_version", "packages", "validated_device", "pip_check_passed")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
