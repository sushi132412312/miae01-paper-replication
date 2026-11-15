"""确认设置与版本后准备固定SST-2子集；当前待确认配置会被拒绝。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, execution_context, read_json, save_json
from src.data import prepare_sst2, validate_real_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "cache")
    args = parser.parse_args()
    config = read_json(args.config)
    validate_real_config(config)
    _, manifests = prepare_sst2(config, args.cache_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, manifest in manifests.items():
        save_json(args.output / f"{name}_manifest.json", manifest)
    save_json(args.output / "execution_context.json", execution_context())
    print("固定数据清单已保存；未运行训练。")


if __name__ == "__main__":
    main()
