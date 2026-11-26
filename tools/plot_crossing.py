# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["matplotlib==3.10.8"]
# ///
"""Make vector manuscript figures from the recorded boundary-crossing summaries."""

from argparse import ArgumentParser
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter, StrMethodFormatter


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("synthetic", "public")
STAGE_LABELS = {"synthetic": "Synthetic geometries", "public": "Public benchmarks"}
FAMILY_LABELS = {
    "affine_shells": "Affine shells",
    "wavy_shells": "Wavy shells",
    "open_chains": "Open chains",
    "local_lobes": "Separated lobes",
    "pendigits": "PenDigits",
    "seeds": "Seeds",
}
FORECAST = {
    "ap_margin": ("Static margin", "#697683", "s", "-"),
    "ap_drag": ("Drag projection", "#126687", "o", "-"),
    "ap_exponential": ("One-step ETD1", "#B64D19", "^", "-"),
}
COVERAGE = {
    "endpoint_coverage": ("Combined endpoint", "#126687", "o", "-"),
    "state_coverage": ("State endpoint", "#B64D19", "^", "--"),
    "stable_coverage": ("Whole-interval stability", "#653A89", "D", "-."),
    "ball_coverage": ("Global-ball stability", "#697683", "s", ":"),
}
ORACLE = {
    "asymmetric_binary_priors": ("Unequal priors", "#126687", "o", "-"),
    "alternating_binary_sources": ("Alternating sources", "#B64D19", "s", "--"),
    "three_class_mixtures": ("Three classes", "#653A89", "^", "-."),
}
TEXT = "#182832"
MUTED = "#465763"
GRID = "#DCE5E8"
TITLE = "#101B22"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.labelcolor": TEXT,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.7,
    "ytick.labelsize": 7.7,
    "text.color": TEXT,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def family_roster(config, stage):
    return config["families" if stage == "synthetic" else "public_families"]


def load_summary(path, stage, config):
    summary = read_json(path)
    phase = summary["phase"]
    if summary["stage"] != stage or phase not in {"development", "final"}:
        raise ValueError(f"Unexpected stage or phase in {path}")
    seeds = config[f"{phase}_seeds"]
    gaps = config["gaps"] if stage == "synthetic" else ["fixed"]
    roster = [
        (family, gap, seed, ratio, horizon)
        for family in family_roster(config, stage)
        for gap in gaps
        for seed in seeds
        for ratio in config["bandwidth_ratios"]
        for horizon in config["horizons"]
    ]
    observed = [(r["family"], r["gap"], r["seed"], r["bandwidth_ratio"], r["horizon"]) for r in summary["rows"]]
    if roster != observed:
        raise ValueError(f"{path} does not have the full prescribed roster")
    return summary


def handles(styles, *, connected=False):
    return [
        Line2D([], [], label=label, color=color, marker=marker, markersize=4.4,
               linewidth=1.15, linestyle=line if connected else "none")
        for label, color, marker, line in styles.values()
    ]


def axes_style(ax, letter=None, title=None):
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(color=GRID, labelcolor=TEXT, length=3)
    if letter is not None:
        ax.set_title(f"{letter}   {title}", loc="left", color=TITLE, pad=7, fontweight="bold")


def save(fig, path, title, phase):
    if phase == "development":
        fig.text(.975, .985, "DEVELOPMENT", ha="right", va="top",
                 color=MUTED, fontsize=7, fontweight="bold")
    fig.savefig(path, facecolor="white", metadata={"Title": title, "Creator": "plot_crossing.py",
                                                  "CreationDate": None, "ModDate": None})
    plt.close(fig)


def forecast_figure(summaries, config, path):
    fig, ax = plt.subplots(figsize=(7.05, 3.65))
    axes_style(ax)
    ratio, h = config["primary_bandwidth_ratio"], config["primary_horizon"]
    primary = [
        r for stage in STAGES for r in summaries[stage]["rows"]
        if r["bandwidth_ratio"] == ratio and r["horizon"] == h
    ]
    families = [f for stage in STAGES for f in family_roster(config, stage)]
    labels = []
    for j, family in enumerate(families):
        rows = [r for r in primary if r["family"] == family]
        mixed = [r for r in rows if r["ap_margin"] is not None]
        labels.append(FAMILY_LABELS[family])
        if j < len(config["families"]):
            ax.axhspan(j - .5, j + .5, color="#F5F8F9", zorder=0)
        ax.text(1.033, j, f"{len(mixed)}/{len(rows)}", va="center", ha="left", fontsize=7.5)
        if not mixed:
            ax.text(.03, j, "No mixed units", color=MUTED, va="center", fontsize=7.5)
            continue
        seeds = sorted({r["seed"] for r in mixed})
        for offset, (metric, (_, color, marker, _)) in zip((-.235, 0, .235), FORECAST.items(), strict=True):
            seed_scores = [mean(r[metric] for r in mixed if r["seed"] == seed) for seed in seeds]
            wiggle = [.009 * (i - (len(seeds) - 1) / 2) for i in range(len(seeds))]
            ax.scatter(seed_scores, [j + offset + d for d in wiggle], color=color, marker=marker,
                       s=11, linewidths=0, alpha=.32)
            ax.scatter(mean(r[metric] for r in mixed), j + offset, color=color, marker=marker,
                       s=28, edgecolors="white", linewidths=.55, zorder=4, clip_on=False)
    cut = len(config["families"]) - .5
    ax.axhline(cut, color="#C9D5D9", linewidth=.8)
    ax.set(yticks=range(len(families)), yticklabels=labels,
           ylim=(len(families) - .5, -.5), xlim=(0, 1.015), xlabel="Average precision for endpoint flips")
    ax.xaxis.set_major_locator(MultipleLocator(.2))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
    ax.grid(axis="x", color=GRID, linewidth=.65)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.text(1.033, -.74, "Mixed units", color=MUTED, fontsize=7, va="bottom", clip_on=False)
    fig.legend(handles=handles(FORECAST), loc="upper left", bbox_to_anchor=(.14, .99),
               ncols=3, frameon=False)
    fig.text(.145, .031, f"Bandwidth / diameter {ratio:g}, time {h:g}. Small marks are seed means. Large marks are unit means.",
             color=MUTED, fontsize=7.2)
    fig.subplots_adjust(left=.20, right=.875, top=.85, bottom=.18)
    save(fig, path, "Forecasting endpoint label flips", summaries["synthetic"]["phase"])


def coverage_figure(summaries, config, path):
    fig, axs = plt.subplots(1, 2, figsize=(7.05, 3.04), sharey=True)
    times = config["horizons"]
    ratio, main = config["primary_bandwidth_ratio"], config["primary_horizon"]
    for j, (ax, stage) in enumerate(zip(axs, STAGES, strict=True)):
        axes_style(ax, ("a", "b")[j], STAGE_LABELS[stage])
        rows = [summaries[stage]["settings"][f"{ratio}/{t}"] for t in times]
        ax.axvline(main, color="#AFBCC1", linestyle=(0, (2, 2)), linewidth=.85)
        for metric, (_, color, marker, line) in COVERAGE.items():
            ax.plot(times, [100 * r[metric] for r in rows], color=color, linestyle=line, marker=marker,
                    linewidth=1.3, markersize=4.5, markerfacecolor="white", markeredgewidth=1.15)
        ax.set(xlim=(min(times) - .08, max(times) + .08), ylim=(-2, 102), xlabel="Dimensionless flow time")
        ax.set_xticks(times, labels=[f"{t:g}" for t in times])
        ax.yaxis.set_major_locator(MultipleLocator(25))
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.grid(axis="y", color=GRID, linewidth=.65)
    axs[0].set_ylabel("Mean certified fraction")
    fig.legend(handles=handles(COVERAGE, connected=True), loc="upper left", bbox_to_anchor=(.065, .99),
               ncols=2, frameon=False, columnspacing=1.35, handlelength=2.1)
    fig.text(.082, .026, f"Bandwidth / diameter {ratio:g}. Dashed vertical guides mark the primary horizon.",
             color=MUTED, fontsize=7.2)
    fig.subplots_adjust(left=.105, right=.98, top=.75, bottom=.22, wspace=.16)
    save(fig, path, "Coverage by flow time", summaries["synthetic"]["phase"])


def oracle_figure(synthetic, path):
    oracle = synthetic["known_law_oracle"]
    fixtures = {source["name"]: source for source in oracle["fixtures"]}
    stationary = fixtures["symmetric_binary"]
    if (stationary["predicted_coefficient"] or stationary["predicted_flip_coefficient"]
            or any(r["ratio_to_horizon"] or r["ratio_to_horizon_squared"] for r in stationary["observations"])):
        raise ValueError("The plotted zero-control annotation would be incorrect")
    times = oracle["horizons"]
    fig, axs = plt.subplots(1, 2, figsize=(7.05, 3.02), sharey=True)
    specs = [
        (r"Flip mass / $t$", "ratio_to_horizon", "predicted_flip_coefficient"),
        (r"Excess Bayes risk / $t^2$", "ratio_to_horizon_squared", "predicted_coefficient"),
    ]
    seen = []
    for j, (ax, (title, observed, predicted)) in enumerate(zip(axs, specs, strict=True)):
        axes_style(ax, ("a", "b")[j], title)
        ax.axhline(1, color="#6A7981", linewidth=.9, linestyle=(0, (3, 3)), zorder=1.5)
        for family, (_, color, marker, line) in ORACLE.items():
            fixture = fixtures[family]
            rows = fixture["observations"]
            if [r["horizon"] for r in rows] != times or fixture[predicted] <= 0:
                raise ValueError(f"Invalid normalization or horizon order for {family}")
            values = [r[observed] / fixture[predicted] for r in rows]
            seen.extend(values)
            ax.plot(times, values, color=color, linestyle=line, marker=marker, linewidth=1.3,
                    markersize=4.5, markerfacecolor="white", markeredgewidth=1.15)
        ax.set(xscale="log", xlabel="Flow time (log scale)")
        ax.set_xticks(times, labels=[f"{t:g}" for t in times])
        ax.tick_params(axis="x", which="minor", length=0)
        ax.grid(axis="y", color=GRID, linewidth=.65)
    low, high = min(1, min(seen)), max(1, max(seen))
    padding = max(.0025, .1 * (high - low))
    axs[0].set(ylim=(low - padding, high + padding), ylabel="Observed / limiting coefficient")
    fig.legend(handles=handles(ORACLE, connected=True), loc="upper left",
               bbox_to_anchor=(.08, .99), ncols=3, frameon=False)
    fig.text(.082, .029, "The horizontal line marks the limit. The symmetric control has zero flips and zero excess risk.",
             color=MUTED, fontsize=7.2)
    fig.subplots_adjust(left=.116, right=.976, top=.78, bottom=.22, wspace=.17)
    save(fig, path, "Known-law calibration of flip and regret coefficients", synthetic["phase"])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "crossing_results")
    parser.add_argument("--output", type=Path, default=ROOT / "paper" / "figures")
    parser.add_argument("--config", type=Path, default=ROOT / "crossing" / "study.json")
    for stage in STAGES:
        parser.add_argument(f"--{stage}-summary", type=Path)
    args = parser.parse_args()
    config = read_json(args.config)
    summaries = {stage: load_summary(getattr(args, f"{stage}_summary") or args.results / f"crossing_{stage}" / "summary.json",
                                     stage, config) for stage in STAGES}
    if summaries["synthetic"]["phase"] != summaries["public"]["phase"]:
        parser.error("The synthetic and public summaries must use the same phase")
    args.output.mkdir(parents=True, exist_ok=True)
    forecast_figure(summaries, config, args.output / "forecast.pdf")
    coverage_figure(summaries, config, args.output / "certification.pdf")
    oracle_figure(summaries["synthetic"], args.output / "oracle.pdf")


if __name__ == "__main__":
    main()
