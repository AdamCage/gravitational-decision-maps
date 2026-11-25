"""Recompute frozen headline metrics from per-example outputs via integer counts."""

from argparse import ArgumentParser
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
from statistics import median


def recall_average(outcomes, predictions):
    if len(outcomes) != len(predictions):
        raise ValueError("Mismatch between evaluator rows and model predictions")
    total = defaultdict(int)
    correct = defaultdict(int)
    for target, prediction in zip(outcomes, predictions, strict=True):
        total[target] += 1
        correct[target] += int(target == prediction)
    return sum(Fraction(correct[c], n) for c, n in total.items()) / len(total)


def reanalyse(raw, protocol):
    stage = protocol["stages"][raw["stage"]]
    seeds = protocol["final_seeds" if raw["split"] == "final" else "development_seeds"]
    rows = {(row["task"], row["seed"]): row for row in raw["rows"]}
    expected = len(seeds) * len(stage["tasks"])
    if len(raw["rows"]) != expected or len(rows) != expected or set(rows) != {
        (task, seed) for task in stage["tasks"] for seed in seeds
    }:
        raise ValueError("Incomplete or duplicated frozen roster")
    scores = defaultdict(Fraction)
    latency_ratios = []
    group_weight = Fraction(1, len(seeds) * len(stage["primary_groups"]))
    for group in stage["primary_groups"]:
        weight = group_weight / len(group)
        for task in group:
            for seed in seeds:
                row = rows[task, seed]
                methods = row["models"]
                methods_in_order = ["gradient", row["static_champion"], "matched_static"]
                for label, method in zip(("gradient", "champion", "matched"), methods_in_order, strict=True):
                    scores[label] += weight * recall_average(row["y_external"], methods[method]["external_predictions"])
                latency_ratios.append(methods["gradient"]["inference_ms_per_row"] /
                                      methods[row["static_champion"]]["inference_ms_per_row"])

    # Recompute core-only champion admission and cross-check two worlds as recorded.
    for row in rows.values():
        selection = row["selection"]
        families = [m for m in protocol["static_families"] if m in selection]
        winner = max(families, key=lambda name: max(selection[name]["scores"]))
        if winner != row["static_champion"]:
            raise ValueError("Static champion does not follow the frozen core-only selection rule")
        for value in selection.values():
            if value["selected_trial"] != value["scores"].index(max(value["scores"])):
                raise ValueError("Candidate selection does not follow the frozen rule")
    if "paired_a" in stage["tasks"]:
        for seed in seeds:
            a, b = rows["paired_a", seed], rows["paired_b", seed]
            if any(u + v != 1 for u, v in zip(a["y_external"], b["y_external"], strict=True)):
                raise ValueError("Paired world outcome mismatch")
            if any(p["external_predictions"] != b["models"][k]["external_predictions"] for k, p in a["models"].items()):
                raise ValueError("Paired world forecast mismatch")

    primary = float(scores["gradient"] - scores["champion"])
    return {"row_count": expected, "primary_gain": primary, "delta_vs_static": primary,
            "delta_vs_matched": float(scores["gradient"] - scores["matched"]),
            "gradient_external_balanced_accuracy": float(scores["gradient"]),
            "median_latency_ratio": median(latency_ratios)}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    content = args.raw.read_bytes()
    raw = json.loads(content)
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text(encoding="utf-8"))
    metrics = reanalyse(raw, protocol)
    Path("raw.json").write_bytes(content)
    encoded = json.dumps(metrics, allow_nan=False) + "\n"
    Path("metrics.json").write_text(encoded, encoding="utf-8")
    Path("summary.json").write_text(encoded, encoding="utf-8")
    print(f"reanalysed {raw['stage']}: {metrics['row_count']} task-seed units", flush=True)


if __name__ == "__main__":
    main()
