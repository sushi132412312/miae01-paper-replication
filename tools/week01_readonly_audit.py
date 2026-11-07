"""只读检查下载的官方LoRA源码，并输出明确标记的教学核对记录。"""
import argparse
import ast
import hashlib
import json
import re
import shlex
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPT = "examples/NLU/roberta_base_sst2.sh"
ENTRY = "examples/NLU/examples/text-classification/run_glue.py"
MODEL = "examples/NLU/src/transformers/models/roberta/modeling_roberta.py"
LAYERS = "loralib/layers.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path, help="官方源码解压目录")
    parser.add_argument("--part", required=True, choices=["data", "lora", "statistics"])
    parser.add_argument("--actor", default="未填写执行者")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.repo.resolve()
    if not (root / SCRIPT).is_file() and (root / "LoRA-main" / SCRIPT).is_file():
        root = root / "LoRA-main"
    sources = {}
    for rel in [SCRIPT, ENTRY, MODEL, LAYERS]:
        path = root / rel
        raw = path.read_bytes()
        body = raw.decode("utf-8")
        if rel.endswith(".py"):
            ast.parse(body, filename=rel)
        sources[rel] = {"text": body, "sha256": hashlib.sha256(raw).hexdigest()}

    def evidence(rel, terms):
        return [{"行号": i + 1, "源码": line.strip()}
                for i, line in enumerate(sources[rel]["text"].splitlines())
                if any(term in line for term in terms)]

    result = {
        "记录类型": "源码静态核对与人工教学计算，不是模型训练结果",
        "执行者": args.actor,
        "生成时间_UTC加8": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "核对部分": args.part,
        "是否下载模型或数据": False,
        "是否训练模型": False,
        "源码SHA256": {rel: data["sha256"] for rel, data in sources.items()},
        "说明": "源码证据不代替真实数据、训练、冻结权重及环境的运行时验证。",
    }
    if args.part == "data":
        parameters = {}
        for line in sources[SCRIPT]["text"].splitlines():
            line = line.strip().removesuffix("\\").strip()
            if line.startswith("--"):
                tokens = shlex.split(line)
                parameters[tokens[0]] = tokens[1] if len(tokens) > 1 else True
        gpu = re.search(r"^export num_gpus=(\d+)", sources[SCRIPT]["text"], re.M)
        result["原脚本参数"] = parameters
        result["原脚本GPU数"] = int(gpu.group(1)) if gpu else None
        result["原脚本是否设置max_train_samples"] = "--max_train_samples" in parameters
        result["主程序证据"] = evidence(ENTRY, [
            "HfArgumentParser((", 'load_dataset("glue"', "AutoTokenizer.from_pretrained(",
            "AutoModelForSequenceClassification.from_pretrained(",
            'train_dataset = datasets["train"]',
            "train_dataset.select(range(data_args.max_train_samples))",
            'else "validation"]', "result = tokenizer(",
        ])
    elif args.part == "lora":
        result["接入位置证据"] = evidence(MODEL, [
            "self.query = lora.Linear", "self.key = nn.Linear", "self.value = lora.Linear",
        ])
        tree = ast.parse(sources[LAYERS]["text"])
        linear = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Linear")
        layer_lines = sources[LAYERS]["text"].splitlines()
        result["Linear内部证据"] = [
            {"行号": i + 1, "源码": layer_lines[i].strip()}
            for i in range(linear.lineno - 1, linear.end_lineno)
            if any(term in layer_lines[i] for term in [
                "self.weight.requires_grad", "self.scaling =",
                "nn.init.kaiming_uniform_", "nn.init.zeros_",
            ])
        ]
        result["主干和分类头训练策略证据"] = evidence(ENTRY, [
            "trainable_params.append('lora')", "name.startswith('deberta')",
            "param.requires_grad = False", "param.requires_grad = True",
        ])
        result["低秩矩阵参数量_理论计算"] = {
            "假设": "12层、隐藏维度768、query/value两处、rank8；不含分类头",
            "计算式": "12 * 2 * (768 * 8 + 8 * 768)",
            "结果": 12 * 2 * (768 * 8 + 8 * 768),
        }
        a = [[1, 2], [3, 4]]
        b = [[0, 0], [0, 0]]
        ba = [[sum(b[i][k] * a[k][j] for k in range(2))
               for j in range(2)] for i in range(2)]
        result["零更新代数演示"] = {
            "类型": "人工小矩阵，仅演示B为零时BA为零",
            "A": a, "B": b, "BA": ba,
        }
    else:
        clean = [80.0, 82.0, 84.0]
        noisy = [79.0, 80.0, 83.0]
        drops = [a - b for a, b in zip(clean, noisy)]
        result["人工统计示例_不属于实验结果"] = {
            "准确率单位": "百分数",
            "干净条件": clean,
            "噪声条件": noisy,
            "干净均值": statistics.mean(clean),
            "干净样本标准差": statistics.stdev(clean),
            "干净总体标准差_仅供辨别": statistics.pstdev(clean),
            "逐种子下降_百分点": drops,
            "下降均值_百分点": statistics.mean(drops),
            "下降样本标准差_百分点": statistics.stdev(drops),
        }
        logits = [[0.1, 1.7], [1.6, 0.2], [0.5, 0.8]]
        labels = [1, 0, 0]
        predictions = [max(range(len(row)), key=lambda i: row[i]) for row in logits]
        result["人工分类示例_不属于实验结果"] = {
            "两个类别的示意分数": logits,
            "正确标签": labels,
            "选择分数较高的类别": predictions,
            "答对数": sum(a == b for a, b in zip(labels, predictions)),
            "总数": len(labels),
            "准确率_百分数": 100 * sum(a == b for a, b in zip(labels, predictions)) / len(labels),
        }
        result["评估与保存证据"] = evidence(ENTRY, [
            "def compute_metrics(", "np.argmax(preds, axis=1)",
            "metric.compute(predictions=preds", "trainer.train(",
            "trainer.evaluate(", 'trainer.save_metrics("train"',
            'trainer.save_metrics("eval"', "if training_args.do_predict:",
        ])
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
