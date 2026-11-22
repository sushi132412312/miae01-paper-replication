"""按轮训练协议的字段与校验，供训练和离线汇总共用。"""
import math

BEST_EPOCH_RULE = "best_validation_accuracy_earliest_epoch"


def uses_epoch_schedule(config):
    return config.get("checkpoint_rule") == BEST_EPOCH_RULE


def validate_epoch_schedule(config):
    for key in ("train_size", "validation_size", "batch_size", "max_length", "epochs"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"{key} 必须为正整数")
    if "max_steps" in config:
        raise ValueError("按轮训练的更新步数由样本数、批次大小和轮数决定，请移除 max_steps")
    for key in ("subset_seed", "train_seed", "label_noise_seed"):
        if type(config.get(key)) is not int or not 0 <= config[key] < 2**32:
            raise ValueError(f"{key} 必须为 0 至 2^32-1 之间的整数")
    flips = config.get("train_label_flips")
    if type(flips) is not int or not 0 <= flips <= config["train_size"]:
        raise ValueError("train_label_flips 必须为 0 至训练样本数之间的整数")
    warmup = config.get("warmup_ratio")
    if type(warmup) not in (float, int) or not 0 <= warmup < 1:
        raise ValueError("warmup_ratio 必须在 [0, 1) 范围内")
    for method in ("ft", "lora"):
        rate = config.get("learning_rates", {}).get(method)
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"缺少有限且为正的 {method} 学习率")
    if config["max_length"] < 3:
        raise ValueError("max_length 至少为 3")


def training_steps(config):
    return config["epochs"] * math.ceil(config["train_size"] / config["batch_size"])
