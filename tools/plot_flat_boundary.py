# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["matplotlib==3.10.8", "numpy==2.3.2", "scipy==1.15.2"]
# ///
"""Draw the vector figure for the affine decision boundary analysis."""

from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, PercentFormatter, StrMethodFormatter
import numpy as np
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "crossing_results" / "flat_boundary_sensitivity.json"
OUTPUT = ROOT / "paper" / "figures" / "decision_invariance.pdf"
SITES = np.array([(1.0, 0.0), (0.6, 0.8), (0.0, 1.0), (-1.0, 0.0), (-0.6, -0.8), (0.0, -1.0)])
IN_PLANE = np.array([2.0, 1.0]) / np.sqrt(5)
NORMAL = np.array([-1.0, 2.0]) / np.sqrt(5)
BETA = 1.2
HORIZON = 2.25
START = 1.45 * IN_PLANE - 0.10 * NORMAL
INK = "#20323e"
BOUNDARY = "#a4b3be"
FACET = "#126687"
TRAJECTORY = "#b44b21"
GRID = "#e3eaee"


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.labelcolor": INK,
    "axes.titlesize": 9,
    "text.color": INK,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "pdf.fonttype": 42,
})


def orbit():
    def field(_time, state):
        logits = SITES @ state / BETA  # Common terms cancel for unit-norm sites.
        weights = np.exp(logits - logits.max())
        return weights @ SITES / weights.sum() - state

    def on_facet(_time, state):
        return NORMAL @ state

    path = solve_ivp(
        field, (0, HORIZON), START, method="DOP853", rtol=2e-12, atol=2e-14, dense_output=True, events=on_facet
    )
    if not path.success or len(path.y_events[0]) != 1:
        raise RuntimeError("Expected one crossing of the highlighted ray")
    states = path.sol(np.linspace(0, HORIZON, 301))
    signs = np.argmax(SITES @ states, axis=0)
    hit = path.y_events[0][0]
    scores = SITES @ hit
    if signs[0] != 0 or signs[-1] != 1 or np.any(np.diff(signs) < 0) or scores[0] <= max(scores[2:]):
        raise RuntimeError("The illustrated trajectory did not switch across the active facet")
    return path, states, hit


def style(ax, title):
    ax.set_title(title, loc="left", fontweight="semibold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BOUNDARY)
    ax.tick_params(color=BOUNDARY, labelcolor=INK, length=3, pad=3)


def geometry(ax):
    style(ax, "(a) Crossing a Voronoi facet")
    ax.set(xlim=(-1.37, 1.62), ylim=(-1.26, 1.26), xlabel=r"$x_1$", ylabel=r"$x_2$")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_aspect("equal", adjustable="box")

    # For distinct equal-radius sites, the static Voronoi rays are the angular
    # bisectors of consecutive sites in cyclic order. SITES is in that order.
    ray_directions = SITES + np.roll(SITES, -1, axis=0)
    rays = 4 * ray_directions / np.linalg.norm(ray_directions, axis=1)[:, None]
    ax.fill([0, rays[-1, 0], rays[0, 0]], [0, rays[-1, 1], rays[0, 1]], color="#f7eddc", zorder=0)
    ax.fill([0, rays[0, 0], rays[1, 0]], [0, rays[0, 1], rays[1, 1]], color="#e2f0f6", zorder=0)
    for i, ray in enumerate(rays):
        ax.plot([0, ray[0]], [0, ray[1]], color=FACET if i == 0 else BOUNDARY, lw=1.4 if i == 0 else 0.85,
                linestyle="-" if i == 0 else (0, (3, 2)), zorder=2)

    ax.scatter(*SITES.T, marker="o", s=23, facecolors=INK, edgecolors="white", linewidths=0.6, zorder=4)
    ax.scatter(0, 0, marker="X", s=41, facecolors=INK, edgecolors="white", linewidths=0.65, zorder=6)
    path, states, hit = orbit()
    ax.plot(*states, color="white", lw=3.6, zorder=5)
    ax.plot(*states, color=TRAJECTORY, lw=1.65, zorder=6)
    ax.scatter(*hit, marker="o", s=22, facecolors="white", edgecolors=TRAJECTORY, linewidths=1.3, zorder=8)
    ax.scatter(*states[:, [0, -1]], marker="o", s=25, facecolors=TRAJECTORY, edgecolors="white", linewidths=0.6, zorder=8)
    ax.annotate("", xy=path.sol(1.5), xytext=path.sol(1.24),
                arrowprops={"arrowstyle": "-|>", "color": TRAJECTORY, "lw": 1.45, "mutation_scale": 9}, zorder=7)
    ax.annotate(r"$x_0$", START, xytext=(4, -10), textcoords="offset points", color=TRAJECTORY, fontsize=8)
    ax.annotate(r"$x_T$", states[:, -1], xytext=(0, 8), textcoords="offset points", color=TRAJECTORY, fontsize=8)
    handles = [
        Line2D([], [], color=INK, linestyle="none", marker="o", markersize=4, label="Sites"),
        Line2D([], [], color=INK, linestyle="none", marker="X", markersize=5, label="Equilibrium"),
        Line2D([], [], color=TRAJECTORY, linewidth=1.65, label="Trajectory"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.015, -0.23), ncol=3, frameon=False,
              edgecolor="none", handlelength=1.25, columnspacing=1.0, handletextpad=0.5, fontsize=7.2)


def calibration(ax, data):
    style(ax, "(b) Fixed-horizon sensitivity")
    ax.set(xlabel=r"Log-mass perturbation magnitude $|\varepsilon|$",
           ylabel="Signed relative error in mass coefficient", xlim=(0.00435, 0.092), ylim=(-0.032, 0.032))
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=0.0015, linscale=0.65)
    magnitudes = sorted(eps for eps in data["design"]["epsilons"] if eps > 0)
    ax.xaxis.set_major_locator(FixedLocator(magnitudes))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.yaxis.set_major_locator(FixedLocator([-0.02, -0.005, 0, 0.005, 0.02]))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    ax.tick_params(axis="y", which="minor", left=False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.axhline(0, lw=0.85, color=BOUNDARY, zorder=1)

    handles = []
    for entry, (color, marker) in zip(data["results"], [("#126687", "o"), ("#4f7a54", "s"), ("#714a8b", "D")], strict=True):
        limit = entry["predicted_switch_mass_coefficient"]
        for sign in (-1, 1):
            observed = {abs(row["epsilon"]): row["switch_mass_divided_by_abs_epsilon"]
                        for row in entry["measurements"] if row["epsilon"] * sign > 0}
            error = [observed[magnitude] / limit - 1 for magnitude in magnitudes]
            ax.plot(magnitudes, error, color=color, marker=marker, linestyle="--" if sign < 0 else "-", lw=1.15,
                    markersize=3.25, markerfacecolor="white" if sign < 0 else color, markeredgewidth=0.75)
        handles.append(Line2D([], [], color=color, marker=marker, markersize=3.25, lw=1.15,
                              label=rf"$T={entry['horizon']:g}$"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=3, frameon=False,
              edgecolor="none", handlelength=1.4, columnspacing=1.1, handletextpad=0.45, fontsize=7.2)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.3), gridspec_kw={"width_ratios": (1.0, 1.11)})
    fig.subplots_adjust(left=0.084, right=0.97, top=0.85, bottom=0.25, wspace=0.31)
    geometry(axes[0])
    calibration(axes[1], data)
    fig.savefig(args.output, facecolor="white", metadata={"Title": "Affine decision invariance",
                "Creator": "plot_flat_boundary.py", "CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
