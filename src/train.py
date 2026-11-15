"""统一 FT/LoRA 工程试跑入口；未确认的真实训练配置会在联网前被拒绝。"""
import argparse
import math
import time
from pathlib import Path

from .common import PROJECT_ROOT, execution_context, json_hash, read_json, save_json, tensor_hash
from .data import batch_order, prepare_sst2, synthetic_data, tokenize_records, validate_real_config
from .evaluation import evaluate_model
from .models import build_model, parameter_report


def run(config, method, output, device="cpu", synthetic=False, cache_dir=None):
    if synthetic:
        if config.get("status") != "synthetic_selfcheck":
            raise PermissionError("需要显式的合成检查配置")
    else:
        validate_real_config(config)
    import torch
    if device not in ("cpu", "cuda"):
        raise ValueError("设备只能为 cpu 或 cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("此环境没有可用的 CUDA PyTorch；请明确选择 CPU 或部署 GPU 环境")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    run_id = output.name
    cache_dir = Path(cache_dir or PROJECT_ROOT / "cache")
    if synthetic:
        data, manifests = synthetic_data(config)
    else:
        from transformers import AutoTokenizer
        selected, manifests = prepare_sst2(config, cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"],
                                                 cache_dir=str(cache_dir))
        data = {name: tokenize_records(rows, tokenizer, config["max_length"])
                for name, rows in selected.items()}
    save_json(output / "config.json", config)
    save_json(output / "execution_context.json", execution_context())
    for name, manifest in manifests.items():
        save_json(output / f"{name}_manifest.json", manifest)
    model = build_model(method, config["train_seed"], config, synthetic, str(cache_dir)).to(device)
    initial = parameter_report(model)
    save_json(output / "model_initialization.json", initial)
    trainable = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
    frozen_before = initial["frozen_parameters_sha256"]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=config["learning_rates"][method], weight_decay=config["weight_decay"])
    batches_per_epoch = math.ceil(len(data["train"]) / config["batch_size"])
    orders = batch_order(len(data["train"]), config["train_seed"],
                         math.ceil(config["max_steps"] / batches_per_epoch))
    losses = []
    model.train()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    for order in orders:
        for start in range(0, len(order), config["batch_size"]):
            batch = [data["train"][i] for i in order[start:start + config["batch_size"]]]
            inputs = {key: torch.tensor([r[key] for r in batch], dtype=torch.long, device=device)
                      for key in ("input_ids", "attention_mask")}
            inputs["labels"] = torch.tensor([r["label"] for r in batch], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs).loss
            if not torch.isfinite(loss):
                raise ValueError("训练 loss 为 NaN/Inf")
            loss.backward()
            optimizer.step()
            losses.append({"step": len(losses) + 1, "loss": loss.item(),
                           "sample_ids": [r["sample_id"] for r in batch]})
            if len(losses) == config["max_steps"]:
                break
        if len(losses) == config["max_steps"]:
            break
    if device == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_started
    training_peak_mib = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
    final = parameter_report(model)
    changed = [name for name, p in model.named_parameters()
               if p.requires_grad and not torch.equal(trainable[name], p.detach().cpu())]
    if final["frozen_parameters_sha256"] != frozen_before:
        raise AssertionError("被冻结的权重发生变化")
    if not changed:
        raise AssertionError("没有任何允许训练的参数发生变化")
    checkpoint = f"final-step-{len(losses)}:{tensor_hash(model.named_parameters())[:16]}"
    context = {"run_id": run_id, "method": method, "seed": config["train_seed"], "checkpoint": checkpoint}
    evaluation_started = time.perf_counter()
    metrics = evaluate_model(model, data["validation"], config["batch_size"], device,
                             context, output / "predictions.csv")
    if device == "cuda":
        torch.cuda.synchronize()
    evaluation_seconds = time.perf_counter() - evaluation_started
    protocol = {key: value for key, value in config.items() if key != "train_seed"}
    result = {**metrics, "experiment_kind": "synthetic_selfcheck" if synthetic else "engineering_pilot",
              "paper_reproduction_result": False, "student_independent_work": False,
              "executor": "AI assistant", "device": device, "steps": len(losses),
              "protocol_sha256": json_hash(protocol),
              "validation_selection_sha256": manifests["validation"]["selection_sha256"],
              "elapsed_seconds": time.perf_counter() - started,
              "training_seconds": training_seconds,
              "evaluation_seconds": evaluation_seconds,
              "gpu_peak_training_allocated_mib": training_peak_mib,
              "gpu_peak_total_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
              "timing_scope": "elapsed_seconds includes setup/download/checks; training_seconds measures only the update loop, not a formal speed benchmark",
              "checkpoint_rule": config["checkpoint_rule"],
              "note": "合成检查不是SST-2成绩；真实试跑也不能替代正式三种子复现。"}
    save_json(output / "training_steps.json", losses)
    save_json(output / "parameter_check.json", {
        "initial": initial, "final": final, "frozen_weights_unchanged": True,
        "changed_trainable_names": changed,
        "synthetic_only": synthetic, "zero_step_A_change_not_required": True})
    save_json(output / "metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=("ft", "lora"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--synthetic-selfcheck", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    result = run(read_json(args.config), args.method, args.output, args.device,
                 args.synthetic_selfcheck, args.cache_dir)
    print(f"completed: {result['experiment_kind']}, {result['method']}, {result['steps']} steps")


if __name__ == "__main__":
    main()
