"""Fit on permitted cores, seal predictions, then score against evaluator labels."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import os
from time import perf_counter
from threading import Event, Thread

import numpy as np
import psutil
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .controls import CONTROL_GRIDS, capacity_mlp_width, make_control, make_mlp
from .data import make_task
from .dynamics import AttractionClassifier


def balanced(y, pred):
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


@contextmanager
def peak_rss():
    """Sample resident process memory, including libraries and allocator caches."""
    proc = psutil.Process()
    start = proc.memory_info().rss
    stats = {"start_bytes": start, "peak_bytes": start}
    done = Event()

    def sample():
        while not done.wait(0.02):
            stats["peak_bytes"] = max(stats["peak_bytes"], proc.memory_info().rss)

    thread = Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield stats
    finally:
        done.set()
        thread.join()
        stats["peak_bytes"] = max(stats["peak_bytes"], proc.memory_info().rss)


def _details(model, x):
    if hasattr(model, "predict_details"):
        details = model.predict_details(x)
    else:
        details = {"pred": model.predict(x)}
    pred = np.asarray(details["pred"], dtype=int)
    nfe = np.broadcast_to(details.get("nfe", 0), pred.shape).astype(int)
    fallback = np.asarray(details.get("fallback", pred), dtype=int)
    if pred.shape != (len(x),) or fallback.shape != pred.shape or np.any(pred < -1) or np.any(nfe < 0):
        raise ValueError("Invalid prediction diagnostics")
    return pred, fallback, nfe


def _predict_record(model, scaler, core, external, noisy, repeats):
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        pred, fallback, nfe = _details(model, scaler.transform(external))
        durations.append(perf_counter() - start)
    core_pred, _, core_nfe = _details(model, scaler.transform(core))
    if noisy is not None:
        noisy_pred, _, noisy_nfe = _details(model, noisy)
        perturb = noisy_pred.tolist()
        perturb_nfe = int(noisy_nfe.sum())
    else:
        perturb, perturb_nfe = None, 0
    return {
        "external_predictions": pred.tolist(), "core_predictions": core_pred.tolist(),
        "fallback_predictions": fallback.tolist() if np.any(pred == -1) else None,
        "perturbed_predictions": perturb,
        "inference_ms_per_row": 1000 * float(np.median(durations)) / len(external),
        "nfe_total": int(nfe.sum()), "nfe_max": int(nfe.max()),
        "nfe_core_total": int(core_nfe.sum()), "nfe_perturbation_total": perturb_nfe,
    }


def _fit(fit_model, x, y):
    start = perf_counter()
    fit_model.fit(x, y)
    return fit_model, perf_counter() - start


def _select(factory, grids, x_fit, y_fit, x_valid, y_valid):
    scores, mean_nfe = [], []
    start = perf_counter()
    for params in grids:
        model, _ = _fit(factory(params), x_fit, y_fit)
        predictions, _, nfe = _details(model, x_valid)
        scores.append(balanced(y_valid, predictions))
        mean_nfe.append(float(nfe.mean()))
    best = int(np.argmax(scores))
    return grids[best], {"scores": scores, "selected_trial": best, "mean_validation_nfe": mean_nfe,
                         "seconds": perf_counter() - start}


def _affine_macs(widths):
    return sum(a * b for a, b in zip(widths, widths[1:]))


def _field_mac_proxy(field):
    k, d = field.centers_.shape
    if field.precisions_.ndim == 3:
        return k * (2 * d * d + 2 * d)
    return k * (3 if field._identity_precision else 5) * d


def _characterize(model):
    values = {}
    for key in ("n_parameters_", "parameter_count", "parameter_count_", "geometry_parameter_count_",
                "training_field_evaluations_"):
        value = getattr(model, key, None)
        if callable(value):
            value = value()
        if value is not None:
            values[key] = int(value)
    if hasattr(model, "coefs_"):
        values["parameter_count"] = int(sum(a.size for a in (*model.coefs_, *model.intercepts_)))
        values["affine_macs_per_row"] = int(sum(a.size for a in model.coefs_))
    if isinstance(model, AttractionClassifier):
        values["field_mac_proxy"] = _field_mac_proxy(model)
    if hasattr(model, "equilibrium_labels_"):
        values["equilibrium_audit"] = {
            "count": len(model.equilibrium_labels_),
            "min_curvature": min(model.equilibrium_curvature_, default=None),
            "max_force_residual": max(model.equilibrium_residual_, default=None),
        }
    if hasattr(model, "jacobian_max_real_"):
        values["equilibrium_audit"] = {
            "count": len(model.anchors_), "max_real_eigenvalue": model.jacobian_max_real_,
            "max_force_residual": model.attractor_residual_,
        }
    return values


def _fit_predict(config, seed, x, y, core, external, task):
    """Only ``y`` is supplied. Neither target labels nor target geometry enter fitting."""
    val = config["validation"]
    xa, xb, ya, yb = train_test_split(x, y, test_size=val["fraction"], stratify=y,
                                    random_state=seed + val["seed_offset"])
    validation_scaler = StandardScaler().fit(xa)
    xa, xb = validation_scaler.transform(xa), validation_scaler.transform(xb)
    scaler = StandardScaler().fit(x)
    xt = scaler.transform(x)
    selection, models = {}, {}
    for name in config["static_families"] + ["sanode"]:
        if name == "boosted_trees" and not task.startswith("wine_"):
            continue
        grid = CONTROL_GRIDS[name]
        if not 1 <= len(grid) <= val["max_trials_per_family"]:
            raise ValueError(f"Unbudgeted family {name}")

        def factory(params):
            return make_control(name, params, seed=seed)

        params, selected = _select(factory, grid, xa, ya, xb, yb)
        model, fit_seconds = _fit(factory(params), xt, y)
        selection[name] = {**selected, "params": params, "fit_seconds": fit_seconds}
        models[name] = (model, fit_seconds, name)

    static_rank = [name for name in config["static_families"] if name in models]
    champion = max(static_rank, key=lambda name: max(selection[name]["scores"]))
    sanode, sanode_seconds, _ = models["sanode"]
    joint, joint_seconds = _fit(make_control("sanode_joint", selection["sanode"]["params"], seed=seed), xt, y)
    models.update({
        "sanode_joint": (joint, joint_seconds, "sanode"),
        "sanode_zero": (sanode.zero_trajectory(), sanode_seconds, "sanode"),
        "sanode_joint_zero": (joint.zero_trajectory(), joint_seconds, "sanode"),
    })

    def flow_model(params, **options):
        return AttractionClassifier(**(config["flow"] | params | options), random_state=seed)

    params, selected = _select(flow_model, config["flow_trials"], xa, ya, xb, yb)
    field, fit_seconds = _fit(flow_model(params), xt, y)
    selection["gradient"] = {**selected, "params": params, "fit_seconds": fit_seconds}
    models["gradient"] = (field, fit_seconds, "gradient")
    width = capacity_mlp_width(xt.shape[1], len(np.unique(y)), field.n_parameters_)
    capacity, capacity_seconds = _fit(make_mlp(seed, (width,), selection["mlp"]["params"]["alpha"]), xt, y)
    models["mlp_capacity"] = (capacity, capacity_seconds, "mlp")
    output_width = 1 if len(np.unique(y)) == 2 else len(np.unique(y))
    wide_grid = CONTROL_GRIDS["mlp_wide"]
    wide_macs = [_affine_macs((xt.shape[1], *trial["hidden_layer_sizes"], output_width)) for trial in wide_grid]
    target_macs = max(1, _field_mac_proxy(field) * selected["mean_validation_nfe"][selected["selected_trial"]])
    compute_trial = int(np.argmin(np.abs(np.log(np.array(wide_macs) / target_macs))))
    selection["mlp_wide"]["compute_matched_trial"] = compute_trial
    selection["mlp_wide"]["compute_target_proxy"] = target_macs
    selection["mlp_wide"]["affine_macs_by_trial"] = wide_macs
    if compute_trial == selection["mlp_wide"]["selected_trial"]:
        models["mlp_compute"] = models["mlp_wide"]
    else:
        model, seconds = _fit(make_control("mlp_wide", wide_grid[compute_trial], seed=seed), xt, y)
        models["mlp_compute"] = (model, seconds, "mlp_wide")
    damping = config["inertia_damping"]
    models.update({
        "matched_static": (field.with_options(readout="static", steps=0), fit_seconds, "gradient"),
        "one_step": (field.with_options(readout="static", steps=1), fit_seconds, "gradient"),
        "many_step_score": (field.with_options(readout="static"), fit_seconds, "gradient"),
        "inertial": (field.with_options(mode="inertial", damping=damping), fit_seconds, "gradient"),
        "gradient_first_entry": (field.with_options(readout="absorbing"), fit_seconds, "gradient"),
        "inertial_first_entry": (field.with_options(mode="inertial", damping=damping,
                                                    readout="absorbing"), fit_seconds, "gradient"),
        "gradient_half_step": (field.with_options(steps=2 * config["flow"]["steps"],
                                                  step_size=config["flow"]["step_size"] / 2), fit_seconds, "gradient"),
        "gradient_half_tolerance": (field.with_options(position_tolerance=0.5 * field.position_tolerance,
                                                        force_tolerance=0.5 * field.force_tolerance),
                                    fit_seconds, "gradient"),
    })
    for name, changes in {
        "gradient_fixed": {"learned": False, "anisotropic": False},
        "gradient_one_prototype": {"n_prototypes": 1},
        "gradient_two_prototypes": {"n_prototypes": 2},
        "gradient_isotropic": {"anisotropic": False},
        "gradient_anisotropic": {"anisotropic": True},
    }.items():
        variant, seconds = _fit(flow_model(params, **changes), xt, y)
        models[name] = (variant, seconds, "gradient")
    start = perf_counter()
    relaxed = field.with_options(constrained=False)
    models["gradient_relaxed"] = (relaxed, fit_seconds + perf_counter() - start, "gradient")

    nn = NearestNeighbors(n_neighbors=2).fit(xt)
    loo = nn.kneighbors(xt, n_neighbors=2)[0][:, 1]
    thresholds = np.quantile(loo, config["strata"]["train_loo_distance_quantiles"])
    external_z = scaler.transform(external)
    distances = nn.kneighbors(external_z, n_neighbors=1)[0][:, 0]
    strata = np.searchsorted(thresholds, distances, side="left")
    perturb = config["perturbation"]
    direction = np.random.default_rng(seed + perturb["seed_offset"]).normal(size=external_z.shape)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    noisy = external_z + perturb["l2"] * direction
    perturb_models = {champion, "gradient", "gradient_relaxed", "matched_static", "inertial", "sanode"}
    outputs = {}
    for name, (model, fit_seconds, tuning) in models.items():
        outputs[name] = {
            **_predict_record(model, scaler, core, external, noisy if name in perturb_models else None,
                              config["measurement"]["batch_latency_repeats"]),
            "fit_seconds": fit_seconds, "selection_family": tuning, **_characterize(model),
        }
    return selection, champion, thresholds.tolist(), strata.tolist(), outputs


def evaluate_one(config, task, seed, split):
    with threadpool_limits(limits=1):
        dataset = make_task(task, seed, split=split)
        with peak_rss() as memory:
            selection, champion, thresholds, strata, output = _fit_predict(
                config, seed, dataset.x_train, dataset.y_train, dataset.x_core, dataset.x_external, task)
        return {"task": task, "seed": seed, "split": split, "static_champion": champion,
                "selection": selection, "strata": strata, "stratum_thresholds": thresholds,
                "peak_rss": memory, "metadata": dataset.metadata,
                "y_core": dataset.y_core.tolist(), "y_external": dataset.y_external.tolist(),
                "models": output}


def run_stage(config, stage, split="final", workers=1, tasks=None):
    tasks = config["stages"][stage]["tasks"] if tasks is None else tasks
    seeds = config["final_seeds" if split == "final" else "development_seeds"]
    jobs = [(task, seed) for task in tasks for seed in seeds]
    records = {}
    if workers == 1:
        for task, seed in jobs:
            records[task, seed] = evaluate_one(config, task, seed, split)
            print(f"finished {task} {seed}", flush=True)
    else:
        # Force one numerical thread during child import as well as during the benchmark.
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
            os.environ[name] = "1"
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(evaluate_one, config, task, seed, split): (task, seed)
                       for task, seed in jobs}
            for future in as_completed(futures):
                task, seed = futures[future]
                records[task, seed] = future.result()
                print(f"finished {task} {seed}", flush=True)
    return [records[job] for job in jobs]
