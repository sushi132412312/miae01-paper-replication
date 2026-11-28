"""Generate report-ready figures from the audited V100 results."""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED_COLORS = {42: "#0072B2", 43: "#D55E00", 44: "#008B69"}
METHOD_COLORS = {"ft": "#2764AA", "lora": "#B84629"}
LABELS = {"ft": "Full fine-tuning", "lora": "LoRA"}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dbe4ec", linewidth=0.7)
    ax.set_axisbelow(True)


def export(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=240, facecolor="white")
    fig.savefig(directory / f"{name}.svg", metadata={"Date": None}, facecolor="white")
    plt.close(fig)


def condition_plot(report, directory):
    conditions = report["paired"]["conditions_by_flipped_label_count"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.7), sharey=True, layout="constrained")
    fig.suptitle("SST-2 validation accuracy under fixed label noise", x=0.02, ha="left", fontweight="bold")
    all_scores = [row["accuracy_percent"] for row in report["runs"]]
    bottom = 2 * math.floor((min(all_scores) - 0.8) / 2)
    top = 2 * math.ceil((max(all_scores) + 1.0) / 2)
    for ax, method in zip(axes, ("ft", "lora")):
        draws = {
            flips: {r["seed"]: r["accuracy_percent"] for r in conditions[flips]["runs"] if r["method"] == method}
            for flips in ("0", "51")
        }
        for seed, color in SEED_COLORS.items():
            ax.plot([0, 1], [draws["0"][seed], draws["51"][seed]], "o-", color=color,
                    markersize=4, linewidth=1.35, label=f"Seed {seed}", alpha=0.85)
        stats = [conditions[flips]["methods"][method] for flips in ("0", "51")]
        ax.errorbar([0.06, 1.06], [x["mean_percent"] for x in stats],
                    yerr=[x["sample_sd_pp"] for x in stats], fmt="D", color="#182433",
                    markerfacecolor="white", markersize=5, capsize=4, label="Mean ± sample SD")
        ax.set(title=LABELS[method], xlim=(-0.25, 1.28), ylim=(bottom, top))
        ax.set_xticks([0, 1], ["Original labels", "51 labels flipped"])
        style(ax)
    axes[0].set_ylabel("Validation accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
    export(fig, directory, "condition_accuracy")


def epoch_plot(report, directory):
    fig, ax = plt.subplots(figsize=(7.9, 4.4), layout="constrained")
    ax.set_title("Validation accuracy across training epochs", loc="left", fontweight="bold")
    for method, color in METHOD_COLORS.items():
        rows = report["by_epoch"]["0"][method]
        ax.errorbar([r["steps"] for r in rows], [r["mean_percent"] for r in rows],
                    yerr=[r["sample_sd_pp"] for r in rows], color=color, marker="o",
                    linewidth=1.9, capsize=4, label=LABELS[method])
    ax.set_xlabel("Cumulative optimizer steps (epoch)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_xticks([64, 128, 192], ["64 (1)", "128 (2)", "192 (3)"])
    ax.set_xlim(47, 209)
    ax.legend(loc="lower right", frameon=False)
    style(ax)
    export(fig, directory, "epoch_accuracy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "miae01-v100"})
    condition_plot(data, args.output)
    epoch_plot(data, args.output)


if __name__ == "__main__":
    main()
