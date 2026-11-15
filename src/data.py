"""固定数据选择，训练种子不参与子集选择。"""
import hashlib
import random
import re

from .common import json_hash


def validate_records(records):
    if not records:
        raise ValueError("数据不能为空")
    ids = []
    for row in records:
        if not isinstance(row.get("sample_id"), str) or not row["sample_id"]:
            raise ValueError("每条记录必须有字符串 sample_id")
        if type(row.get("label")) is not int or row["label"] not in (0, 1):
            raise ValueError("标签必须是整数 0 或 1")
        ids.append(row["sample_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("数据中存在重复 sample_id")


def fixed_subset(records, count, subset_seed):
    validate_records(records)
    if type(count) is not int or not 1 <= count <= len(records):
        raise ValueError("样本数必须处于 1 与数据总数之间")
    positions = sorted(random.Random(subset_seed).sample(range(len(records)), count))
    return [dict(records[index]) for index in positions]


def manifest_for(records, split, subset_seed):
    validate_records(records)
    items = []
    for row in records:
        item = {"sample_id": row["sample_id"], "label": row["label"]}
        if "sentence" in row:
            item["text_sha256"] = hashlib.sha256(row["sentence"].encode("utf-8")).hexdigest()
        if "input_ids" in row:
            item["tokens_sha256"] = json_hash(row["input_ids"])
        items.append(item)
    return {"split": split, "subset_seed": subset_seed, "count": len(items),
            "label_counts": {str(label): sum(r["label"] == label for r in records) for label in (0, 1)},
            "samples": items, "selection_sha256": json_hash(items)}


def batch_order(sample_count, train_seed, epochs=1):
    """独立生成器；模型构造消耗随机数不会改变训练样本顺序。"""
    import torch
    generator = torch.Generator().manual_seed(train_seed)
    return [torch.randperm(sample_count, generator=generator).tolist() for _ in range(epochs)]


def validate_real_config(config):
    if config.get("status") != "approved_engineering_pilot":
        raise PermissionError("真实模型训练设置尚待组长确认；当前仅允许离线合成自检")
    for key in ("model_revision", "dataset_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", config.get(key) or ""):
            raise ValueError(f"{key} 必须填写已核验的完整版本 SHA，不能使用浮动 main")
    if config.get("precision") != "float32":
        raise ValueError("此入口当前只验证了 float32；其他精度需另行验证")
    for key in ("train_size", "validation_size", "batch_size", "max_steps", "max_length", "lora_r"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"{key} 必须为正整数")
    if config["max_length"] > 512:
        raise ValueError("RoBERTa-base 的本入口最大文本长度限定为 512")
    if config.get("model_id") != "FacebookAI/roberta-base" or config.get("dataset_id") != "nyu-mll/glue":
        raise ValueError("此入口当前仅针对 RoBERTa-base / GLUE SST-2")
    if config.get("checkpoint_rule") != "final_step_for_engineering_pilot_only":
        raise ValueError("当前只支持末步工程试跑；正式最佳轮次选择尚未实现")
    if config.get("lora_alpha", 0) <= 0 or config.get("weight_decay", -1) < 0:
        raise ValueError("LoRA alpha/weight_decay 无效")
    for method in ("ft", "lora"):
        if config.get("learning_rates", {}).get(method, 0) <= 0:
            raise ValueError(f"缺少有效 {method} 学习率")


def prepare_sst2(config, cache_dir):
    """仅在显式确认配置后联网；本次交付未运行此函数。"""
    validate_real_config(config)
    from datasets import load_dataset
    dataset = load_dataset(config["dataset_id"], "sst2", revision=config["dataset_revision"],
                           cache_dir=str(cache_dir))
    selected, manifests = {}, {}
    for split, count in (("train", config["train_size"]), ("validation", config["validation_size"])):
        rows = [{"sample_id": f"{split}:{r['idx']}", "sentence": r["sentence"], "label": int(r["label"])}
                for r in dataset[split]]
        selected[split] = fixed_subset(rows, count, config["subset_seed"])
        manifests[split] = manifest_for(selected[split], split, config["subset_seed"])
        manifests[split]["dataset_fingerprint"] = dataset[split]._fingerprint
        manifests[split]["dataset_revision"] = config["dataset_revision"]
        manifests[split]["source_split_count"] = len(rows)
    return selected, manifests


def synthetic_data(config):
    """合成 token，无真实文本和预训练知识，只测试代码执行。"""
    rng = random.Random(config["subset_seed"])
    result, manifests = {}, {}
    for split, count in (("train", config["train_size"]), ("validation", config["validation_size"])):
        rows = []
        for index in range(count):
            tokens = [0] + [rng.randrange(4, 96) for _ in range(config["max_length"] - 2)] + [2]
            rows.append({"sample_id": f"synthetic-{split}:{index}", "label": index % 2,
                         "input_ids": tokens, "attention_mask": [1] * len(tokens)})
        result[split] = rows
        manifests[split] = manifest_for(rows, split, config["subset_seed"])
    return result, manifests


def tokenize_records(rows, tokenizer, max_length):
    encoded = tokenizer([r["sentence"] for r in rows], padding="max_length", truncation=True,
                        max_length=max_length)
    return [{"sample_id": row["sample_id"], "label": row["label"],
             "input_ids": encoded["input_ids"][i], "attention_mask": encoded["attention_mask"][i]}
            for i, row in enumerate(rows)]
