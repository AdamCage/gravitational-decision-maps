# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["matplotlib==3.10.8"]
# ///
"""Render the fixed study roster from recorded summaries with Matplotlib."""

from argparse import ArgumentParser
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
STAGES = {"synthetic": "Synthetic tasks", "real": "Public datasets"}
TASK_LABELS = {
    "rings_wedge": "Concentric rings",
    "curved_wedge": "Curved regions",
    "anisotropic_wide": "Anisotropic · wider core",
    "anisotropic_narrow": "Anisotropic · narrower core",
    "disconnected": "Withheld components",
    "digits_narrow": "Digits · narrower core",
    "digits_wide": "Digits · wider core",
    "wine_narrow": "Wine · narrower core",
    "wine_wide": "Wine · wider core",
}
METHODS = {
    "gradient": ("Gradient destination", "#13788A", "o"),
    "static_champion": ("Core-selected static", "#A85216", "s"),
    "matched_static": ("Same-geometry static", "#7453A1", "^"),
    "mlp_compute": ("Compute-proxy MLP", "#526974", "D"),
}
PRIMARY = tuple(METHODS)[:3]
TEXT = "#172531"
MUTED = "#4E5F69"
GRID = "#DDE5E9"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_summary(path, stage, protocol):
    summary = read_json(path)
    if summary["stage"] != stage or summary["split"] not in {"development", "final"}:
        raise ValueError(f"Unexpected stage or split in {path}")
    seeds = protocol[f"{summary['split']}_seeds"]
    tasks = protocol["stages"][stage]["tasks"]
    expected = [(task, seed) for task in tasks for seed in seeds]
    observed = [(row["task"], row["seed"]) for row in summary["replicates"]]
    if observed != expected:
        raise ValueError(f"{path} does not contain the full frozen task and seed roster")
    return summary


def stage_tasks(protocol, stage):
    return [task for group in protocol["stages"][stage]["primary_groups"] for task in group]


def legend_handles(methods):
    return [
        Line2D([], [], linestyle="none", marker=METHODS[name][2], color=METHODS[name][1],
               markersize=7, label=METHODS[name][0])
        for name in methods
    ]


def balanced(row, method):
    return 100 * row["methods"][method]["external"]["balanced_accuracy"]


def label_figure(fig, title, subtitle, split, *, left=0.04):
    fig.text(left, 0.96, title, ha="left", va="top", fontsize=20, fontweight="bold", color=TEXT)
    fig.text(left, 0.909, subtitle, ha="left", va="top", fontsize=10.5, color=MUTED)
    if split == "development":
        fig.text(.96, .96, "DEVELOPMENT", ha="right", va="top", fontsize=10,
                 color=MUTED, fontweight="bold")


def prepare_axis(ax, title):
    ax.set_title(title, loc="left", color=TEXT, fontweight="bold", fontsize=12, pad=10)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(color=GRID, labelcolor=TEXT)


def plot_tasks(ax, summary, tasks):
    prepare_axis(ax, STAGES[summary["stage"]])
    n_tasks = len(tasks)
    for row_no, task in enumerate(tasks):
        rows = [row for row in summary["replicates"] if row["task"] == task]
        if row_no % 2 == 0:
            ax.axhspan(row_no - .5, row_no + .5, color="#F4F7F8", zorder=0)
        for method, offset in zip(PRIMARY, [-0.24, 0, 0.24]):
            _, color, marker = METHODS[method]
            scores = [balanced(row, method) for row in rows]
            positions = [row_no + offset + .025 * (i - (len(scores) - 1) / 2) for i in range(len(scores))]
            ax.scatter(scores, positions, s=18, color=color, marker=marker, alpha=.42, linewidths=0)
            ax.scatter(mean(scores), row_no + offset, s=48, color=color, marker=marker,
                       linewidths=.7, edgecolors="white", zorder=4, clip_on=False)
    ax.set_yticks(range(n_tasks), [TASK_LABELS[task] for task in tasks])
    ax.set(ylim=(n_tasks - .5, -.5), xlim=(0, 100), xlabel="External balanced accuracy")
    ax.set_xticks(range(0, 101, 20))
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis="x", color=GRID, linewidth=.8)
    ax.tick_params(axis="y", length=0, pad=7)


def plot_stage_effect(ax, summary, comparisons, minimum_gain, max_magnitude):
    prepare_axis(ax, "Stage effect")
    ax.axvline(0, linewidth=.9, color=MUTED, zorder=0)
    ax.axvline(100 * minimum_gain, linewidth=.9, color=MUTED, linestyle="--", zorder=0)
    for row_no, (candidate, control) in enumerate(comparisons):
        effect = summary["contrasts"][f"{candidate} vs {control}"]
        _, color, marker = METHODS[control]
        low, high = [100 * endpoint for endpoint in effect["interval_95"]]
        ax.hlines(row_no, low, high, color=color, linewidth=2.1)
        ax.vlines([low, high], row_no - .075, row_no + .075, color=color, linewidth=1)
        ax.scatter(100 * effect["mean_delta"], row_no, color=color, marker=marker, s=42, zorder=3)
    ax.set_yticks(range(len(comparisons)), [METHODS[control][0] for _, control in comparisons])
    ax.set(xlim=(-max_magnitude, max_magnitude), ylim=(len(comparisons) - .45, -.55),
           xlabel="Gradient − control (percentage points)")
    ax.tick_params(axis="y", length=0, pad=7)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True, prune="both"))
    ax.grid(axis="x", color=GRID, linewidth=.8)


def accuracy_figure(summaries, protocol, target):
    fig, axes = plt.subplots(2, 2, figsize=(13.3, 8.6), gridspec_kw={
        "width_ratios": [2.35, 1], "height_ratios": [5, 4],
    })
    comparisons = protocol["primary_contrasts"]
    endpoints = [
        abs(100 * value)
        for stage, summary in summaries.items()
        for candidate, control in comparisons
        for value in [
            *summary["contrasts"][f"{candidate} vs {control}"]["interval_95"],
            protocol["stages"][stage]["minimum_gain"],
        ]
    ]
    extent = max(5, max(endpoints) * 1.14)
    for stage, (tasks, effects) in zip(STAGES, axes):
        plot_tasks(tasks, summaries[stage], stage_tasks(protocol, stage))
        minimum_gain = protocol["stages"][stage]["minimum_gain"]
        plot_stage_effect(effects, summaries[stage], comparisons, minimum_gain, extent)
    label_figure(fig, "How accurately do models generalize beyond observed cores?",
                 "External known-class examples. Small dots show individual seeds and larger marks show task means.",
                 summaries["synthetic"]["split"])
    fig.legend(handles=legend_handles(PRIMARY), loc="upper left", bbox_to_anchor=(.035, .881),
               ncols=len(PRIMARY), frameon=False, borderaxespad=0)
    fig.text(.04, .048, "Unresolved predictions count as errors. Stage whiskers show paired seed-block 95% intervals.",
             color=MUTED, fontsize=9.5)
    fig.text(.04, .025, "Dashed lines mark the prespecified minimum gain. Public-data intervals are conditional on their data sources.",
             color=MUTED, fontsize=9.5)
    fig.subplots_adjust(left=.188, right=.964, top=.795, bottom=.127, hspace=.57, wspace=.58)
    fig.savefig(target, dpi=180, facecolor="white")
    plt.close(fig)


def cost_figure(summaries, protocol, target):
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12.5, 6))
    methods = list(PRIMARY)
    if all("mlp_compute" in row["methods"] for s in summaries.values() for row in s["replicates"]):
        methods.append("mlp_compute")
    for stage, ax in zip(STAGES, axes):
        prepare_axis(ax, STAGES[stage])
        roster = stage_tasks(protocol, stage)
        rows = [row for row in summaries[stage]["replicates"] if row["task"] in roster]
        for method in methods:
            _, color, marker = METHODS[method]
            timing = [1000 * row["methods"][method]["inference_ms_per_row"] for row in rows]
            if min(timing) <= 0:
                raise ValueError(f"Logarithmic latency plot requires positive timings for {method}")
            ax.scatter(timing, [balanced(row, method) for row in rows], s=40, color=color,
                       marker=marker, alpha=.68, linewidths=.5, edgecolors="white")
        ax.set(xscale="log", ylim=(-2, 102), xlabel="Batch inference time (µs per row, log scale)")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.grid(color=GRID, linewidth=.8)
    axes[0].set_ylabel("External balanced accuracy")
    axes[0].yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    label_figure(fig, "Accuracy and inference cost",
                 "Each point is one task and seed. Times include the common input transform.",
                 summaries["synthetic"]["split"])
    fig.legend(handles=legend_handles(methods), loc="upper left", bbox_to_anchor=(.035, .867),
               ncols=len(methods), frameon=False, borderaxespad=0)
    fig.text(.04, .044, "Absolute timing depends on the execution host.",
             color=MUTED, fontsize=9.5)
    fig.subplots_adjust(left=.087, right=.96, top=.745, bottom=.17, wspace=.16)
    fig.savefig(target, dpi=180, facecolor="white")
    plt.close(fig)


def paired_figure(summary, target):
    fig, ax = plt.subplots(figsize=(7.6, 7.1))
    prepare_axis(ax, "Complementary target labels")
    ax.plot([0, 100], [100, 0], color=MUTED, linestyle="--", linewidth=.95)
    champions = {row["seed"]: row["static_champion"] for row in summary["replicates"] if row["task"] == "paired_a"}
    for method in PRIMARY:
        _, color, marker = METHODS[method]
        results = [entry["methods"][champions[entry["seed"]] if method == "static_champion" else method]
                   for entry in summary["paired_worlds"]]
        ax.scatter([100 * row["world_a_ba"] for row in results], [100 * row["world_b_ba"] for row in results],
                   s=72, marker=marker, facecolors="none", edgecolors=color, linewidths=1.2, clip_on=False)
    ax.set(xlim=(0, 100), ylim=(0, 100), aspect="equal",
           xlabel="World A external balanced accuracy", ylabel="World B external balanced accuracy")
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(color=GRID, linewidth=.8)
    ax.legend(handles=legend_handles(PRIMARY), loc="upper right", frameon=False)
    label_figure(fig, "Two worlds with identical cores",
                 "The sum of the two accuracy scores equals resolved coverage.",
                 summary["split"], left=.08)
    fig.text(.08, .044, "The diagonal represents full coverage. The diagnostic uses the same external features in both worlds.",
             fontsize=9.2, color=MUTED)
    fig.subplots_adjust(left=.155, right=.96, top=.82, bottom=.135)
    fig.savefig(target, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "figures")
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--synthetic-summary", type=Path, help="Override the synthetic summary path")
    parser.add_argument("--real-summary", type=Path, help="Override the real summary path")
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    summaries = {
        stage: load_summary(getattr(args, f"{stage}_summary") or args.results / stage / "summary.json", stage, protocol)
        for stage in STAGES
    }
    if summaries["synthetic"]["split"] != summaries["real"]["split"]:
        parser.error("Use summaries from the same split")
    args.output.mkdir(parents=True, exist_ok=True)
    accuracy_figure(summaries, protocol, args.output / "balanced_accuracy.png")
    cost_figure(summaries, protocol, args.output / "cost_and_accuracy.png")
    paired_figure(summaries["synthetic"], args.output / "paired_worlds.png")


if __name__ == "__main__":
    main()
