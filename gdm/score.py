"""Fixed metrics and seed-blocked summaries for the sealed external predictions."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
import json
from pathlib import Path
from statistics import mean, median

import numpy as np


def _rates(y, pred, labels):
    present = {c: y == c for c in labels if np.any(y == c)}
    recall = {int(c): float(np.mean(pred[m] == c)) for c, m in present.items()}
    support = {int(c): int(m.sum()) for c, m in present.items()}
    return {"balanced_accuracy": mean(recall.values()) if recall else None,
            "accuracy": float(np.mean(y == pred)) if len(y) else None,
            "n": len(y), "class_recall": recall, "class_support": support}


def _summary(row, pred):
    y = np.array(row["y_external"])
    yc = np.array(row["y_core"])
    strata = np.array(row["strata"])
    a = np.array(pred["external_predictions"])
    labels = np.unique(y)
    if not len(y) or not len(yc) or len(y) != len(a) or len(y) != len(strata) or (
        len(yc) != len(pred["core_predictions"]) or set(np.unique(a)) - set(labels) - {-1}
    ):
        raise ValueError("Malformed scored prediction record")
    resolved = a != -1
    fallback = pred["fallback_predictions"]
    effective = np.where(resolved, a, fallback) if fallback is not None else a
    altered = pred["perturbed_predictions"]
    alt = np.array(altered) if altered is not None else None
    both = (resolved & (alt != -1)) if alt is not None else np.zeros_like(resolved)
    return {
        "external": _rates(y, a, labels), "core": _rates(yc, np.array(pred["core_predictions"]), labels),
        "strata": [_rates(y[strata == i], a[strata == i], labels) for i in range(3)],
        "coverage": float(np.mean(resolved)), "unresolved": int((~resolved).sum()),
        "conditional_accuracy": float(np.mean(a[resolved] == y[resolved])) if np.any(resolved) else None,
        "conditional_balanced_accuracy": _rates(y[resolved], a[resolved], labels)["balanced_accuracy"],
        "fallback_balanced_accuracy": _rates(y, effective, labels)["balanced_accuracy"],
        "perturbation_disagreement": float(np.mean(alt != a)) if alt is not None else None,
        "perturbation_resolution_flip": float(np.mean((alt != -1) != resolved)) if alt is not None else None,
        "perturbation_both_resolved_disagreement": float(np.mean(alt[both] != a[both])) if np.any(both) else None,
        "inference_ms_per_row": pred["inference_ms_per_row"], "fit_seconds": pred["fit_seconds"],
        "nfe_per_row": pred["nfe_total"] / len(y), "nfe_max": pred["nfe_max"],
        "tuning_seconds": row["selection"][pred["selection_family"]]["seconds"],
        **{k: v for k, v in pred.items() if k in {"n_parameters_", "training_field_evaluations_", "equilibrium_audit",
                                                "affine_macs_per_row", "field_mac_proxy"}
           or k.endswith("parameter_count") or k.endswith("parameter_count_")},
    }


def _add_alias(rows):
    for row in rows:
        row["methods"]["static_champion"] = row["methods"][row["static_champion"]]


def _score_at(row, model):
    return row["methods"][model]["external"]["balanced_accuracy"]


def _seed_values(rows, groups, model):
    by_key = {(r["task"], r["seed"]): r for r in rows}
    return [mean(mean(_score_at(by_key[(task, seed)], model) for task in group)
                 for group in groups) for seed in sorted({r["seed"] for r in rows})]


def _effect(lh, rh, draws):
    deltas = np.array(lh) - np.array(rh)
    ci = np.quantile(deltas[draws].mean(axis=1), (0.025, 0.975))
    return {"mean_delta": float(deltas.mean()), "interval_95": ci.tolist(), "by_seed": deltas.tolist()}


def _task_table(rows):
    output = {}
    for task in dict.fromkeys(r["task"] for r in rows):
        same = [r for r in rows if r["task"] == task]
        methods = list(same[0]["methods"])
        output[task] = {name: mean(_score_at(r, name) for r in same) for name in methods}
    return output


def _paired_worlds(raw):
    left = {r["seed"]: r for r in raw if r["task"] == "paired_a"}
    right = {r["seed"]: r for r in raw if r["task"] == "paired_b"}
    if not left and not right:
        return None
    if left.keys() != right.keys():
        raise ValueError("Incomplete paired worlds")
    results = []
    for seed in sorted(left):
        a, b = left[seed], right[seed]
        y, yy = np.array(a["y_external"]), np.array(b["y_external"])
        if not np.array_equal(y + yy, np.ones_like(y)) or np.bincount(y).tolist() != [len(y)//2] * 2:
            raise ValueError("Paired worlds must have balanced complementary external labels")
        if a["static_champion"] != b["static_champion"] or a["strata"] != b["strata"]:
            raise ValueError("Paired worlds changed the core validation or distances")
        comparisons = {}
        for method, p in a["models"].items():
            pred = np.asarray(p["external_predictions"])
            if p["external_predictions"] != b["models"][method]["external_predictions"]:
                raise ValueError(f"Paired world predictions changed: {method} at {seed}")
            ba = np.mean(pred == y)
            bb = np.mean(pred == yy)
            coverage = np.mean(pred != -1)
            if not np.isclose(ba + bb, coverage):
                raise ValueError("Invalid complementary-world identity")
            comparisons[method] = {"world_a_ba": float(ba), "world_b_ba": float(bb), "coverage": float(coverage)}
        results.append({"seed": seed, "methods": comparisons})
    return results


def _overlap(rows):
    output = {}
    for task in dict.fromkeys(row["task"] for row in rows):
        sets = [set(row["metadata"].get("external_ids", [])) for row in rows if row["task"] == task]
        if sets and all(sets):
            output[task] = {"unique_external_rows": len(set.union(*sets)),
                            "total_evaluated_rows": sum(map(len, sets)),
                            "max_pairwise_shared_rows": max((len(a & b) for a, b in combinations(sets, 2)), default=0)}
    return output


def build_report(raw, protocol, *, partial=False):
    config = protocol["stages"][raw["stage"]]
    rows = raw["rows"]
    seeds = protocol["final_seeds"] if raw["split"] == "final" else protocol["development_seeds"]
    expected = [(t, s) for t in config["tasks"] for s in seeds]
    observed = [(r["task"], r["seed"]) for r in rows]
    if observed != expected and not partial:
        raise ValueError("The complete ordered roster is required")
    summaries = [{"task": r["task"], "seed": r["seed"], "static_champion": r["static_champion"],
                  "selection": r["selection"], "stratum_thresholds": r["stratum_thresholds"],
                  "peak_rss": r["peak_rss"],
                  "methods": {key: _summary(r, p) for key, p in r["models"].items()}} for r in rows]
    _add_alias(summaries)
    result = {"stage": raw["stage"], "split": raw["split"], "tasks": _task_table(summaries),
              "replicates": summaries, "external_overlap": _overlap(rows)}
    if partial and observed != expected:
        return result

    result["paired_worlds"] = _paired_worlds(rows)
    pairs = protocol["primary_contrasts"] + [
        ["many_step_score", "matched_static"], ["one_step", "matched_static"],
        ["gradient", "gradient_fixed"], ["gradient", "gradient_relaxed"],
        ["gradient", "inertial"], ["gradient", "gradient_first_entry"],
        ["inertial", "inertial_first_entry"], ["gradient_half_step", "gradient"],
        ["gradient_half_tolerance", "gradient"],
        ["gradient_two_prototypes", "gradient_one_prototype"],
        ["gradient_anisotropic", "gradient_isotropic"], ["gradient", "sanode"],
        ["sanode_joint", "sanode"], ["sanode_joint", "sanode_joint_zero"],
        ["gradient", "mlp_capacity"], ["gradient", "mlp_wide"], ["gradient", "mlp_compute"],
    ]
    draws = np.random.default_rng(protocol["bootstrap"]["seed"]).integers(
        len(seeds), size=(protocol["bootstrap"]["resamples"], len(seeds)))
    groups = config["primary_groups"]
    method_names = list(dict.fromkeys(name for pair in pairs for name in pair))
    method_names = [m for m in method_names if all(m in r["methods"] for r in summaries)]
    vals = {name: _seed_values(summaries, groups, name) for name in method_names}
    result["aggregate_ba"] = {k: mean(v) for k, v in vals.items()}
    result["contrasts"] = {f"{lh} vs {rh}": _effect(vals[lh], vals[rh], draws) for lh, rh in pairs
                           if lh in vals and rh in vals}
    tasks = {t for group in groups for t in group}
    primary_rows = [r for r in summaries if r["task"] in tasks]
    ratio = median(r["methods"]["gradient"]["inference_ms_per_row"] /
                   r["methods"]["static_champion"]["inference_ms_per_row"] for r in primary_rows)
    result["median_latency_ratio"] = ratio
    result["champion_selection_counts"] = dict(Counter(r["static_champion"] for r in primary_rows))
    for variant in ("gradient_half_step", "gradient_half_tolerance"):
        numerical = [float(np.mean(np.asarray(r["models"]["gradient"]["external_predictions"]) !=
                                   r["models"][variant]["external_predictions"]))
                     for r in rows if r["task"] in tasks]
        result[f"median_{variant.removeprefix('gradient_')}_disagreement"] = median(numerical)
    result["decision"] = {
        "practical_gain_pass": all(result["contrasts"][f"{l} vs {r}"]["mean_delta"] >= config["minimum_gain"]
                                   for l, r in protocol["primary_contrasts"]),
        "positive_interval_pass": all(result["contrasts"][f"{l} vs {r}"]["interval_95"][0] > 0
                                      for l, r in protocol["primary_contrasts"]),
        "latency_pass": ratio <= config["maximum_latency_ratio"],
    }
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")
