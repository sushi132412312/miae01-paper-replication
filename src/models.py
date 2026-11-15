"""现代 Transformers 上使用固定版本官方 loralib；与原 NLU 全栈存在实现差异。"""
from .common import seed_everything, tensor_hash


def build_model(method, seed, config, synthetic=False, cache_dir=None):
    import torch
    from transformers import RobertaConfig, RobertaForSequenceClassification
    from third_party.loralib import Linear
    if method not in ("ft", "lora"):
        raise ValueError("method 只能为 ft 或 lora")
    seed_everything(seed)
    if synthetic:
        if config.get("status") != "synthetic_selfcheck":
            raise ValueError("合成检查必须使用专门配置")
        architecture = RobertaConfig(vocab_size=97, hidden_size=32, intermediate_size=64,
                                    num_hidden_layers=2, num_attention_heads=4,
                                    max_position_embeddings=64, num_labels=2,
                                    pad_token_id=1, bos_token_id=0, eos_token_id=2)
        architecture._attn_implementation = "eager"
        model = RobertaForSequenceClassification(architecture)
    else:
        from .data import validate_real_config
        validate_real_config(config)
        model = RobertaForSequenceClassification.from_pretrained(
            config["model_id"], revision=config["model_revision"], cache_dir=cache_dir,
            num_labels=2, use_safetensors=True, attn_implementation="eager")

    # 保留构造完共同主干/分类头之后的随机状态；适配器构造不改变外部 RNG。
    if method == "lora":
        with torch.random.fork_rng(devices=[]):
            for layer in model.roberta.encoder.layer:
                for name in ("query", "value"):
                    old = getattr(layer.attention.self, name)
                    new = Linear(old.in_features, old.out_features, r=config["lora_r"],
                                 lora_alpha=config["lora_alpha"], lora_dropout=0.0,
                                 bias=old.bias is not None, merge_weights=False)
                    with torch.no_grad():
                        new.weight.copy_(old.weight)
                        if old.bias is not None:
                            new.bias.copy_(old.bias)
                    setattr(layer.attention.self, name, new)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = ".lora_" in name or name.startswith("classifier.")
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True
    return model


def parameter_report(model):
    named = list(model.named_parameters())
    backbone = [(n, p) for n, p in named if n.startswith("roberta.") and ".lora_" not in n]
    head = [(n, p) for n, p in named if n.startswith("classifier.")]
    frozen = [(n, p) for n, p in named if not p.requires_grad]
    lora_b = [p for n, p in named if ".lora_B" in n]
    return {
        "total_parameters": sum(p.numel() for _, p in named),
        "trainable_parameters": sum(p.numel() for _, p in named if p.requires_grad),
        "lora_parameters": sum(p.numel() for n, p in named if ".lora_" in n),
        "classifier_parameters": sum(p.numel() for _, p in head),
        "frozen_parameters": sum(p.numel() for _, p in frozen),
        "common_backbone_sha256": tensor_hash(backbone),
        "classifier_sha256": tensor_hash(head),
        "frozen_parameters_sha256": tensor_hash(frozen),
        "trainable_parameter_names": [n for n, p in named if p.requires_grad],
        "lora_module_names": [n for n, module in model.named_modules() if hasattr(module, "lora_A")],
        "lora_merge_weights": False,
        "lora_B_all_zero": all(bool((p.detach() == 0).all()) for p in lora_b) if lora_b else None,
    }
