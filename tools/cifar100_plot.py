# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["matplotlib==3.10.8"]
# ///
"""Draw CIFAR-100 comparisons from completed official-test records."""

from argparse import ArgumentParser
from hashlib import file_digest
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "crossing_results"
INK = "#172635"
MUTED = "#526272"
GRID = "#dfe5e9"
BASELINE = "#778495"
CNN = "#255d97"
FLOW = "#08786e"
BLUR = "#b65c37"
RIDGE = "#51649a"


def read_json(path, required=False):
    return json.loads(path.read_text(encoding="utf-8")) if required or path.is_file() else None


def decimal(value, digits):
    return f"{value:.{digits}f}".replace(".", ",")


def percent(value, digits=2):
    return f"{decimal(100 * value, digits)}%"


def archived_boost_rows(raw_boost, embedding_boost, longer, longer_embeddings):
    rows = []
    if raw_boost and raw_boost.get("status") == "complete":
        tests = raw_boost["metrics"]
        rows.extend((label, tests[key]["test"]["top1"], BASELINE) for label, key in (
            ("Логистическая регрессия\nручные признаки", "linear"),
            ("CatBoost без подбора\nручные признаки", "fixed_catboost"),
        ))
        tuned = tests["optuna_catboost"]["test"]["top1"]
        tuned_trees = raw_boost["resources"]["iterations"]
        if longer and longer.get("status") == "complete":
            tuned = longer["test"][str(longer["selected_trees"])]["top1"]
            tuned_trees = longer["selected_trees"]
        rows.append((f"CatBoost + Optuna\nручные, {tuned_trees} деревьев", tuned, BASELINE))
    if embedding_boost and embedding_boost.get("status") == "complete":
        metrics = embedding_boost["metrics"]
        top1 = metrics["optuna_catboost"]["test"]["top1"]
        tuned_trees = 160
        if longer_embeddings and longer_embeddings.get("status") == "complete":
            tuned_trees = longer_embeddings["selected_trees"]
            top1 = longer_embeddings["test"][str(tuned_trees)]["top1"]
        rows.append((f"CatBoost + Optuna\nCNN, {tuned_trees} деревьев", top1, BASELINE))
        rows.append(("Новая линейная голова\nпризнаки CNN", metrics["refit_linear"]["test"]["top1"], BASELINE))
    return rows


def neural_records(snapshot, local_run):
    if local_run is None:
        selection = read_json(snapshot, required=True)["validation_selection"]
        record = selection["candidate_runs"][selection["selected"]]
        return record["clean_test"], record["blur"]["test"], record["sha256"]
    clean = read_json(local_run / "metrics.json", required=True)["official_test"]
    blur = read_json(local_run / "blur_metrics.json", required=True)["test"]
    hashes = {}
    for name in ("head.npz", "e_test.npy", "e_test_blur.npy"):
        path = local_run / name
        if path.is_file():
            with path.open("rb") as stream:
                hashes[name] = file_digest(stream, "sha256").hexdigest()
    return clean, blur, hashes


def validate_flow_source(flow, expected_hashes, clean_top1):
    if flow["test_rows"] != 10000 or flow["clean"]["nonzero_test"]["raw"]["top1"] != clean_top1:
        raise ValueError("Flow and frozen CNN records refer to different test predictions")
    flow_hashes = flow.get("input_sha256", {})
    for name in ("head.npz", "e_test.npy", "e_test_blur.npy"):
        if name in expected_hashes and name in flow_hashes and expected_hashes[name] != flow_hashes[name]:
            raise ValueError(f"Flow source hash mismatch for {name}")


def flow_rows(flow, blur_top1):
    if flow is None:
        return [], []
    clean = flow["clean"]
    horizon = decimal(clean["best_nonzero"]["horizon"], 2)
    accuracy = [
        ("Ближайший центроид\nпризнаки CNN", clean["centroid_test"]["top1"], BASELINE),
        (f"CNN + GDM\nRK4, T = {horizon}", clean["nonzero_test"]["rk4"]["top1"], FLOW),
    ]
    blur = flow.get("gaussian_blur")
    if blur is None:
        return accuracy, []
    if blur["nonzero_test"]["raw"]["top1"] != blur_top1:
        raise ValueError("Blurred flow and CNN records refer to different test predictions")
    mse = [("Размытый вход", blur["nonzero_test"]["feature_mse"]["raw"], BLUR),
           ("GDM с отсечением", blur["label_preserving_clipped_chord_best_nonzero"]["feature_mse"], FLOW)]
    controls = blur.get("restoration_controls", {})
    for key, name, color in (("mean_shrinkage", "Сдвиг к среднему", BASELINE),
                             ("paired_ridge", "Парная Ridge", RIDGE)):
        if key in controls:
            mse.append((name, controls[key]["label_preserving_mse"], color))
    return accuracy, mse


def bar_panel(axis, rows, title, xlabel, is_accuracy):
    labels, values, fills = zip(*rows, strict=True)
    axis.barh(range(len(values)), values, height=0.57, color=fills, zorder=3)
    axis.set_yticks(range(len(labels)), labels)
    axis.invert_yaxis()
    axis.margins(y=0.08)
    axis.set_axisbelow(True)
    axis.set_title(title, loc="left", pad=15, fontweight="bold", fontsize=13, color=INK)
    axis.set_xlabel(xlabel, loc="left", labelpad=8, color=MUTED, fontsize=10)
    axis.tick_params(axis="x", color=MUTED, labelcolor=MUTED, labelsize=9)
    axis.tick_params(axis="y", length=0, pad=8, labelcolor=INK, labelsize=10)
    axis.grid(axis="x", linewidth=0.7, color=GRID)
    axis.spines[["left", "top", "right"]].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    if is_accuracy:
        span = 1.0
        axis.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
        axis.xaxis.set_major_formatter(FuncFormatter(lambda val, _: percent(val, 0)))
    else:
        span = max(values)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda val, _: decimal(val, 2)))
    axis.set_xlim(0, max(1.08 if is_accuracy else 0.0, max(values) + 0.17 * span))
    for position, value in enumerate(values):
        display = percent(value) if is_accuracy else decimal(value, 4)
        axis.text(value + 0.023 * span, position, display, ha="left", va="center", color=INK, fontsize=10)


def render(args):
    neural, blurred, neural_hashes = neural_records(args.neural_archive, args.neural)
    if neural["n"] != 10000 or blurred["n"] != 10000:
        raise ValueError("Expected all official test rows in both CNN cohorts")
    cnn_top1, blur_top1 = neural["top1_accuracy"], blurred["top1_accuracy"]
    flow = read_json(args.flow)
    if flow is not None:
        validate_flow_source(flow, neural_hashes, cnn_top1)

    baselines = archived_boost_rows(read_json(args.boost), read_json(args.embedding_boost),
                                    read_json(args.boost_horizon), read_json(args.embedding_budget))
    dynamics, restoration = flow_rows(flow, blur_top1)
    hybrid = read_json(args.catboost_flow)
    if hybrid and hybrid.get("status") == "complete":
        trees = max(hybrid["tree_counts"])
        top1 = hybrid["official_test"][str(trees)]["selected"]["metrics"]["top1"]
        dynamics.append((f"CatBoost + GDM\nCNN, {trees} деревьев", top1, FLOW))
    accuracy = sorted([*baselines, *dynamics, ("Сверточная сеть (CNN)", cnn_top1, CNN)], key=lambda row: row[1])
    paired = [("Чистые изображения", cnn_top1, CNN), ("Гауссовское размытие", blur_top1, BLUR)]

    plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 200})
    fig = plt.figure(figsize=(12.8, 7.2 if restoration else 5.7), facecolor="white")
    fig.text(0.055, 0.924, "CIFAR-100", fontsize=24, fontweight="bold", color=INK)
    subtitle = "Классификация и восстановление признаков" if restoration else "Классификация изображений"
    fig.text(0.055, 0.867, f"{subtitle} на официальном тесте из 10 000 изображений",
             fontsize=11, color=MUTED)
    first = fig.add_axes([0.245, 0.19, 0.275, 0.575])
    bar_panel(first, accuracy, "Точность на чистых изображениях", "Top-1 точность", True)

    second = fig.add_axes([0.72, 0.57, 0.24, 0.195] if restoration else [0.72, 0.19, 0.24, 0.575])
    bar_panel(second, paired, "CNN на парных входах", "Top-1 точность", True)
    if restoration:
        third = fig.add_axes([0.72, 0.19, 0.24, 0.255])
        bar_panel(third, restoration, "MSE при сохранении меток", "MSE относительно чистых признаков CNN", False)

    fig.text(0.055, 0.062, "Размытие с ядром 5 × 5 и σ = 1,25 пикселя применено к тем же тестовым изображениям.",
             fontsize=9, color=MUTED)
    if flow:
        fig.text(0.055, 0.034, "Горизонт GDM выбран по валидации.",
                 fontsize=9, color=MUTED)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"Title": "CIFAR-100. Классификация и восстановление признаков",
                                       "Software": "tools/cifar100_plot.py"})
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--neural-archive", type=Path, default=ARCHIVE / "cifar100_neural.json")
    parser.add_argument("--neural", type=Path, help="Override the archive with a local neural run directory")
    parser.add_argument("--boost", type=Path, default=ARCHIVE / "cifar100_optuna.json")
    parser.add_argument("--embedding-boost", type=Path, default=ARCHIVE / "cifar100_embedding_boost.json")
    parser.add_argument("--boost-horizon", type=Path, default=ARCHIVE / "cifar100_boost_horizon.json")
    parser.add_argument("--embedding-budget", type=Path, default=ARCHIVE / "cifar100_embedding_budget.json")
    parser.add_argument("--flow", type=Path, default=ARCHIVE / "cifar100_flow.json")
    parser.add_argument("--catboost-flow", type=Path, default=ARCHIVE / "cifar100_catboost_flow.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output/cifar100-benchmark.png")
    render(parser.parse_args())


if __name__ == "__main__":
    main()
