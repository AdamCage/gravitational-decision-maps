# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["matplotlib==3.10.8"]
# ///
"""Plot paired participant effects from the PAMAP2 outcome archive."""

from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SERIES = (
    ("ridge_project", "Projected ridge", "#2e788b", "s"),
    ("extra_trees_project", "Projected ExtraTrees", "#104a3a", "o"),
    ("prototype_mean_project", "Projected prototype mean", "#a05232", "v"),
)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "crossing_results/pamap2_restoration.json")
    parser.add_argument("--output", type=Path, default=ROOT / "paper/figures/pamap2_transfer.pdf")
    args = parser.parse_args()
    rows = json.loads(args.results.read_text(encoding="utf-8"))["holdouts"]
    ids = [row["subject"] for row in rows]
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.63), layout="constrained")
    outcomes = (
        ("all", "feature_mse", "Feature MSE reduction (%)"),
        ("near_range", "peak_mae_m_s2", "Near-range peak MAE reduction (%)"),
    )
    for axis, (sample, measure, title) in zip(axes, outcomes, strict=True):
        axis.axhline(0, color="#6b7377", linewidth=0.8)
        axis.grid(axis="y", color="#e2e8e9", linewidth=0.7)
        for offset, (key, label, color, marker) in zip((-0.13, 0, 0.13), SERIES, strict=True):
            effects = [100 * (1 - row["metrics"][key][sample][measure] / row["metrics"]["observed"][sample][measure])
                       for row in rows]
            axis.plot([subject + offset for subject in ids], effects, linestyle="none", marker=marker,
                      markersize=3.8, color=color, label=label)
        axis.set_title(title, loc="left", fontsize=8, fontweight="semibold")
        axis.set_xlabel("Held-out participant")
        axis.set_xticks(ids)
        axis.tick_params(length=2.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    fig.savefig(args.output, metadata={"CreationDate": None, "ModDate": None})


if __name__ == "__main__":
    main()
