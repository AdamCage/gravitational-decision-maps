"""Run matched inertia comparisons on the existing crossing data roster."""

from hashlib import sha256
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

from crossing.data import make_case
from crossing.flow import MixtureFlow
from crossing.public import make_public
from .flow import rk4_inertial


STUDY_PATH = Path(__file__).with_name("study.json")
SPEC_PATH = STUDY_PATH.parent.parent / "crossing" / "study.json"


def settings(protocol, study):
    primary = study["primary_speed_limit_ratio"]
    base = [(ratio, horizon, primary)
            for ratio in protocol["bandwidth_ratios"] for horizon in protocol["horizons"]]
    return base + [(protocol["primary_bandwidth_ratio"], protocol["primary_horizon"], cap)
                   for cap in study["sensitivity_speed_limit_ratios"]]


def _array_hash(values, dtype):
    return sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


def _evaluate_setting(model, external, horizon, limit, study):
    methods = study["methods"]

    def predict(method, step):
        if method == "static":
            return model.predict(external), 0, external
        endpoint, nfe = rk4_inertial(model, external, horizon, step, mode=method,
                                     damping=study["damping"], speed_limit=limit)
        return model.predict(endpoint), nfe, endpoint

    # The refinement pass also warms each method on the same-size batch.
    refined = {method: predict(method, study["refined_max_step"])[0] for method in methods}
    timings = {method: [] for method in methods}
    predictions, displacements = {}, {}
    for index in range(study["timing_repeats"]):
        offset = index % len(methods)
        for method in methods[offset:] + methods[:offset]:
            start = perf_counter()
            prediction, nfe, endpoint = predict(method, study["max_step"])
            timings[method].append(perf_counter() - start)
            if method in predictions:
                if not np.array_equal(predictions[method][0], prediction):
                    raise ValueError(f"Inconsistent predictions across timed repeats for {method}")
            else:
                displacements[method] = float(np.mean(np.linalg.norm(endpoint - external, axis=1)) / model.diameter)
            predictions[method] = prediction, nfe
    return {method: {"predictions": predictions[method][0].tolist(),
                     "refined_predictions": refined[method].tolist(),
                     "latencies_s": timings[method], "nfe": predictions[method][1],
                     "mean_relative_displacement": displacements[method]}
            for method in methods}


def _trial(protocol, study, stage, family, gap, seed, phase):
    if stage == "synthetic":
        data = make_case(family, gap, seed)
    else:
        folds = tuple(protocol["final_seeds"]) if phase == "final" else ()
        data = make_public(family, seed, split=phase, fold_seeds=folds)
    scaler = StandardScaler().fit(data.x_train)
    training = scaler.transform(data.x_train)
    external = scaler.transform(data.x_test)
    models = {ratio: MixtureFlow.fit(training, data.y_train, prototypes=protocol["prototypes"],
                                    bandwidth_ratio=ratio, seed=seed)
              for ratio in protocol["bandwidth_ratios"]}
    results = []
    for ratio, horizon, cap in settings(protocol, study):
        model = models[ratio]
        limit = cap * model.diameter
        outcomes = _evaluate_setting(model, external, horizon, limit, study)
        results.append({"bandwidth_ratio": ratio, "horizon": horizon, "speed_limit_ratio": cap,
                        "speed_limit": limit, "methods": outcomes})
    train_hash = sha256(np.ascontiguousarray(data.x_train, dtype="<f8").tobytes())
    train_hash.update(np.ascontiguousarray(data.y_train, dtype="<i8").tobytes())
    return {"family": family, "gap": gap, "seed": seed, "y_test": data.y_test.tolist(),
            "test_input_sha256": _array_hash(data.x_test, "<f8"),
            "train_sha256": train_hash.hexdigest(),
            "results": results}


def run(protocol, study, stage, phase, family=None):
    families = protocol["families" if stage == "synthetic" else "public_families"]
    if family is not None and family not in families:
        raise ValueError(f"Family {family!r} is outside the {stage} roster")
    gaps = protocol["gaps"] if stage == "synthetic" else ["fixed"]
    seeds = protocol[f"{phase}_seeds"]
    start = perf_counter()
    units = []
    with threadpool_limits(limits=1):
        libraries = [{key: info[key] for key in ("internal_api", "num_threads", "version", "architecture")
                      if key in info} for info in threadpool_info()]
        for name in families:
            if family is not None and name != family:
                continue
            for gap in gaps:
                for seed in seeds:
                    units.append(_trial(protocol, study, stage, name, gap, seed, phase))
                    print(f"completed {stage} {name} {gap} {seed}", flush=True)
    return {"stage": stage, "phase": phase, "workers": 1, "units": units,
            "runtime_seconds": perf_counter() - start,
            "timing_environment": {"platform": platform.platform(), "processor": platform.processor(),
                                   "python": platform.python_version(), "threadpools": libraries}}
