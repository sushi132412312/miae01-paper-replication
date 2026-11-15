"""离线检查并保存可审阅的测试与合成运行记录，不下载或训练真实 RoBERTa-base。"""
import argparse
import io
import json
import os
import sys
import unittest
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, execution_context, read_json, save_json
from src.evaluation import recompute_accuracy, summarize_seed_results
from src.train import run


class RecordedResult(unittest.TextTestResult):
    def startTest(self, test):
        super().startTest(test)
        self.all_ids = getattr(self, "all_ids", []) + [test.id()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordedResult)
    result = runner.run(suite)
    text = stream.getvalue()
    (args.output / "test_results.txt").write_text(text, encoding="utf-8")
    summary = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "skipped": len(result.skipped), "passed": result.wasSuccessful(),
               "test_ids": getattr(result, "all_ids", []), "synthetic_only": True,
               "real_roberta_base_training_executed": False, "sst2_downloaded": False,
               "execution_context": execution_context()}
    save_json(args.output / "test_summary.json", summary)
    print(text)
    if not result.wasSuccessful():
        raise SystemExit(1)
    config = read_json(PROJECT_ROOT / "configs/selfcheck.json")
    recounted = {}
    for method in ("ft", "lora"):
        folder = args.output / f"synthetic_{method}"
        metrics = run(config, method, folder, synthetic=True)
        recounted[method] = recompute_accuracy(folder / "predictions.csv", folder / "validation_manifest.json")
        recounted[method]["matches_saved_accuracy"] = metrics["accuracy"] == recounted[method]["accuracy"]
        recounted[method]["formal_seed_summary"] = summarize_seed_results([metrics])
    save_json(args.output / "independent_recount.json", recounted)
    print(json.dumps({"all_tests_passed": True, "tests_run": result.testsRun,
                      "synthetic_runs": list(recounted), "real_training_started": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
