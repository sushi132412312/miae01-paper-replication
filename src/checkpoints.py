"""保存和恢复最佳轮次的可训练张量，文件保留在本地运行目录。"""
import hashlib
from pathlib import Path

BEST_WEIGHTS = "best.safetensors"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_best_weights(model, directory):
    from safetensors.torch import save_file

    path = Path(directory) / BEST_WEIGHTS
    temporary = path.with_name("best.pending.safetensors")
    state = {name: parameter.detach().cpu().contiguous()
             for name, parameter in model.named_parameters() if parameter.requires_grad}
    save_file(state, str(temporary))
    temporary.replace(path)
    return {"filename": path.name, "bytes": path.stat().st_size,
            "sha256": file_sha256(path), "tensor_count": len(state)}


def restore_best_weights(model, directory, metadata):
    import torch
    from safetensors.torch import load_file

    path = Path(directory) / BEST_WEIGHTS
    if (metadata["filename"] != path.name or metadata["bytes"] != path.stat().st_size
            or metadata["sha256"] != file_sha256(path)):
        raise ValueError("最佳权重的文件大小或 SHA256 不符")
    state = load_file(str(path), device="cpu")
    expected = {name: parameter for name, parameter in model.named_parameters()
                if parameter.requires_grad}
    if set(state) != set(expected) or len(state) != metadata["tensor_count"]:
        raise ValueError("最佳权重和模型可训练张量集合不一致")
    for name, parameter in expected.items():
        if state[name].shape != parameter.shape or state[name].dtype != parameter.dtype:
            raise ValueError(f"最佳权重的张量形状或类型不符: {name}")
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(state[name])
