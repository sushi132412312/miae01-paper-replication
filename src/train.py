"""统一 FT/LoRA 训练入口，支持末步试跑和按轮评估。"""
import argparse
import math
import shutil
import time
from pathlib import Path

from .checkpoints import save_best_weights
from .common import PROJECT_ROOT, execution_context, json_hash, read_json, save_json, tensor_hash
from .data import batch_order, prepare_sst2, synthetic_data, tokenize_records, validate_real_config
from .evaluation import evaluate_model
from .models import build_model, parameter_report
from .protocol import training_steps, uses_epoch_schedule, validate_epoch_schedule


def load_run_data(config, synthetic, cache_dir):
    if synthetic:
        return synthetic_data(config)
    from transformers import AutoTokenizer

    selected, manifests = prepare_sst2(config, cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"],
                                             cache_dir=str(cache_dir))
    return ({name: tokenize_records(rows, tokenizer, config["max_length"])
             for name, rows in selected.items()}, manifests)


def run(config, method, output, device="cpu", synthetic=False, cache_dir=None):
    by_epoch = uses_epoch_schedule(config)
    if synthetic:
        if config.get("status") != "synthetic_selfcheck":
            raise PermissionError("需要显式的合成检查配置")
        if by_epoch:
            validate_epoch_schedule(config)
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
    data, manifests = load_run_data(config, synthetic, cache_dir)
    run_context = execution_context()
    save_json(output / "config.json", config)
    save_json(output / "execution_context.json", run_context)
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
    max_steps = training_steps(config) if by_epoch else config["max_steps"]
    scheduler = None
    if by_epoch:
        from transformers import get_linear_schedule_with_warmup

        warmup_steps = math.ceil(max_steps * config["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, max_steps)
    orders = batch_order(len(data["train"]), config["train_seed"],
                         math.ceil(max_steps / batches_per_epoch))
    losses = []
    history = []
    best = None
    weight_metadata = None
    training_seconds = evaluation_seconds = 0.0
    model.train()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    for epoch, order in enumerate(orders, start=1):
        training_started = time.perf_counter()
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
            record = {"step": len(losses) + 1, "loss": loss.item(),
                      "sample_ids": [r["sample_id"] for r in batch]}
            if scheduler is not None:
                record.update(epoch=epoch, learning_rate=optimizer.param_groups[0]["lr"])
                scheduler.step()
            losses.append(record)
            if len(losses) == max_steps:
                break
        if device == "cuda":
            torch.cuda.synchronize()
        training_seconds += time.perf_counter() - training_started
        if by_epoch:
            current = {"run_id": run_id, "method": method, "seed": config["train_seed"],
                       "checkpoint": f"epoch-{epoch}-step-{len(losses)}"}
            prediction_path = f"epochs/{epoch:03}/predictions.csv"
            evaluation_started = time.perf_counter()
            epoch_result = evaluate_model(model, data["validation"], config["batch_size"], device,
                                          current, output / prediction_path)
            evaluation_seconds += time.perf_counter() - evaluation_started
            epoch_result.update(epoch=epoch, updates=len(losses), predictions=prediction_path)
            history.append(epoch_result)
            if best is None or epoch_result["correct"] > best["correct"]:
                best = epoch_result
                weight_metadata = save_best_weights(model, output)
        if len(losses) == max_steps:
            break
    training_peak_mib = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" and not by_epoch else None
    final = parameter_report(model)
    changed = [name for name, p in model.named_parameters()
               if p.requires_grad and not torch.equal(trainable[name], p.detach().cpu())]
    if final["frozen_parameters_sha256"] != frozen_before:
        raise AssertionError("被冻结的权重发生变化")
    if not changed:
        raise AssertionError("没有任何允许训练的参数发生变化")
    if by_epoch:
        shutil.copyfile(output / best["predictions"], output / "predictions.csv")
        save_json(output / "checkpoint_selection.json", {
            "rule": config["checkpoint_rule"], "best_epoch": best["epoch"],
            "best_checkpoint": best["checkpoint"], "weights": weight_metadata, "epochs": history,
        })
        metrics = {key: value for key, value in best.items() if key not in ("epoch", "updates", "predictions")}
        metrics.update(best_epoch=best["epoch"], best_epoch_steps=best["updates"],
                       warmup_steps=warmup_steps, train_selection_sha256=manifests["train"]["selection_sha256"])
        kind = "synthetic_epoch_rehearsal" if synthetic else "reduced_scale_experiment"
    else:
        checkpoint = f"final-step-{len(losses)}:{tensor_hash(model.named_parameters())[:16]}"
        context = {"run_id": run_id, "method": method, "seed": config["train_seed"], "checkpoint": checkpoint}
        evaluation_started = time.perf_counter()
        metrics = evaluate_model(model, data["validation"], config["batch_size"], device,
                                 context, output / "predictions.csv")
        evaluation_seconds = time.perf_counter() - evaluation_started
        kind = "synthetic_selfcheck" if synthetic else "engineering_pilot"
    protocol = {key: value for key, value in config.items() if key != "train_seed"}
    result = {**metrics, "experiment_kind": kind,
              "paper_reproduction_result": False,
              "eligible_for_reduced_scale_summary": not synthetic and by_epoch,
              "student_independent_work": run_context["student_independent_work"],
              "executor": run_context["executor"], "device": device, "steps": len(losses),
              "protocol_sha256": json_hash(protocol),
              "validation_selection_sha256": manifests["validation"]["selection_sha256"],
              "elapsed_seconds": time.perf_counter() - started,
              "training_seconds": training_seconds,
              "evaluation_seconds": evaluation_seconds,
              "gpu_peak_training_allocated_mib": training_peak_mib,
              "gpu_peak_total_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
              "timing_scope": "elapsed_seconds includes setup/download/checks; training_seconds measures only the update loop, not a formal speed benchmark",
              "checkpoint_rule": config["checkpoint_rule"],
              "note": "结果须结合实际训练数据、检查点和协议解读。"}
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
    parser.add_argument("--train-seed", type=int, help="按轮训练使用的种子")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.train_seed is not None:
        if not uses_epoch_schedule(config):
            parser.error("--train-seed 仅用于按轮训练配置")
        config["train_seed"] = args.train_seed
    result = run(config, args.method, args.output, args.device, args.synthetic_selfcheck, args.cache_dir)
    print(f"completed: {result['experiment_kind']}, {result['method']}, {result['steps']} steps")


if __name__ == "__main__":
    main()
