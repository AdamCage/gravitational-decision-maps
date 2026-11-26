"""Run the locally fixed, leave-one-subject-out PAMAP2 transfer protocol."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import platform

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from crossing.affine_reserve import AffineReadout, GaussianReservePath
from crossing.flow import MixtureFlow
from tools.decision_safe_restoration import CELL_RESERVE, DecisionCell, affine_pairs, flow_endpoint
from tools.pamap2_pairs import load_pamap2_pairs


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "crossing" / "pamap2_protocol.json"
SOURCES = (
    "crossing/affine_certificate.py", "crossing/affine_reserve.py", "crossing/flow.py",
    "tools/decision_safe_restoration.py", "tools/pamap2_pairs.py", "tools/pamap2_restoration.py", "uv.lock",
)
NEAR_RANGE = 0.95 * 6 * 9.80665
REF_NEAR_RANGE = 0.95 * 16 * 9.80665
FRACTIONS = (1.0, 0.5, 0.25, 0.125)
SEED = 5421


def gaussian_fit(reference, labels, residual):
    centers, classes, counts = [], [], []
    squared_error = 0.0
    for label in np.unique(labels):
        rows = reference[labels == label]
        fitted = KMeans(n_clusters=2, random_state=SEED, n_init=4).fit(rows)
        centers.extend(fitted.cluster_centers_)
        classes.extend([label, label])
        counts.extend(np.bincount(fitted.labels_, minlength=2))
        squared_error += fitted.inertia_
    variance = squared_error / reference.size
    discrepancy = np.mean(np.square(residual))
    masses = np.asarray(counts, dtype=float) / len(reference)
    return (MixtureFlow(centers, classes, masses, variance + discrepancy), masses,
            float(np.log1p(discrepancy / variance)))


def exact_horizons(path, queries, horizon):
    local = np.zeros(len(queries))
    global_ = np.zeros(len(queries))
    for left in range(0, len(queries), 64):
        key = slice(left, left + 64)
        prepared = path.prepare(queries[key])
        for fraction in FRACTIONS:
            accepted, slab = path.certify(prepared, float(horizon * fraction))
            local[key] = np.where((local[key] == 0) & accepted, fraction, local[key])
            global_[key] = np.where((global_[key] == 0) & slab, fraction, global_[key])
    return local * horizon, global_ * horizon


def deploy_segment(start, endpoint, head, strict):
    accepted = head.segment(start, endpoint)
    deployed = np.where(accepted[:, None], endpoint, start)
    if not np.all(head.segment(start, deployed) == strict):
        raise ArithmeticError("Identity fallback failed to certify a previously strict start")
    return deployed, int((~accepted).sum())


def measure(candidate, classifier, scaler, reference, peak, peak_indices, initial, labels, tail):
    error = np.square(candidate - reference).mean(axis=1)
    physical = scaler.inverse_transform(candidate)
    peak_error = np.abs(physical[:, peak_indices].max(axis=1) - peak)
    decision = classifier.predict(candidate)

    def stats(subset):
        return None if not subset.any() else {
            "n": int(subset.sum()),
            "feature_mse": float(error[subset].mean()),
            "peak_mae_m_s2": float(peak_error[subset].mean()),
            "label_changes": int(np.count_nonzero(decision[subset] != initial[subset])),
            "correct": int(np.count_nonzero(decision[subset] == labels[subset])),
        }

    return {"all": stats(np.ones(len(candidate), dtype=bool)), "near_range": stats(tail)}


def evaluate(data, held_out, peak_indices):
    test = data.subjects == held_out
    train = ~test
    scaler = StandardScaler().fit(data.reference[train])
    x = scaler.transform(data.observed[train])
    ref = scaler.transform(data.reference[train])
    obs = scaler.transform(data.observed[test])
    truth = scaler.transform(data.reference[test])
    labels = data.labels[test]
    classifier = LogisticRegression(C=1, max_iter=1500, random_state=SEED).fit(ref, data.labels[train])
    initial = classifier.predict(obs)
    readout = AffineReadout(classifier.coef_, classifier.intercept_, CELL_RESERVE)
    initial_cell = readout.initial_cell(obs)
    if np.any(initial_cell.strict & (classifier.classes_[initial_cell.winner] != initial)):
        raise ArithmeticError("Classifier disagrees with the verified initial winner")
    cell = DecisionCell(classifier, obs, *affine_pairs(classifier))
    # Target a tiny interior numerical slack. Outward checks below still use
    # the fixed protocol reserve and the exact encoded classifier.
    cell.required += 1e-10 * np.abs(cell.start)
    residual = ref - x
    prior, masses, horizon = gaussian_fit(ref, data.labels[train], residual)
    flow = GaussianReservePath(readout, prior.centers, masses, prior.beta)
    local_time, slab_time = exact_horizons(flow, obs, horizon)
    ridge = Ridge(alpha=1).fit(x, residual)
    forest = ExtraTreesRegressor(n_estimators=64, min_samples_leaf=8, max_features=0.75,
                                 random_state=SEED, n_jobs=1).fit(x, residual)
    full, full_departures = flow_endpoint(prior, obs, horizon, cell)
    local_end, local_departures = flow_endpoint(prior, obs, local_time, cell)
    slab_end, slab_departures = flow_endpoint(prior, obs, slab_time, cell)
    candidates = {
        "calibration": obs + residual.mean(axis=0),
        "ridge": obs + ridge.predict(obs),
        "extra_trees": obs + forest.predict(obs),
        "prototype_mean": obs - np.expm1(-horizon) * prior.field(obs),
        "full_flow": full,
    }
    metrics = {}
    peak = data.peak_reference[test]
    tail = data.peak_observed[test] >= NEAR_RANGE

    def record(name, points, fallbacks=None):
        row = measure(points, classifier, scaler, truth, peak, peak_indices, initial, labels, tail)
        if fallbacks is not None:
            row["segment_fallbacks"] = fallbacks
            if row["all"]["label_changes"] != 0:
                raise ArithmeticError("Certified restoration changed the initial label")
        metrics[name] = row

    record("observed", obs)
    for key, candidate in candidates.items():
        record(key, candidate)
        record(f"{key}_clip", *deploy_segment(obs, cell.clip_segment(obs, candidate), readout, initial_cell.strict))
        projected, diagnostics = cell.project(obs, candidate)
        record(f"{key}_project", *deploy_segment(obs, projected, readout, initial_cell.strict))
        metrics[f"{key}_project"]["projection_solver"] = diagnostics
    record("local_flow", *deploy_segment(obs, local_end, readout, initial_cell.strict))
    record("slab_flow", *deploy_segment(obs, slab_end, readout, initial_cell.strict))
    return {
        "subject": int(held_out), "n": int(test.sum()), "reference_near_range": int(np.sum(peak >= REF_NEAR_RANGE)),
        "strict_starts": int(initial_cell.strict.sum()), "horizon": horizon,
        "controls": {"sampled_departures_full": full_departures, "sampled_departures_local": local_departures,
                     "sampled_departures_slab": slab_departures,
                     "full_local_paths": int(np.count_nonzero(local_time == horizon)),
                     "full_slab_paths": int(np.count_nonzero(slab_time == horizon)),
                     "positive_local_paths": int(np.count_nonzero(local_time > 0)),
                     "positive_slab_paths": int(np.count_nonzero(slab_time > 0)),
                     "mean_local_fraction": float(local_time.mean() / horizon),
                     "mean_slab_fraction": float(slab_time.mean() / horizon)},
        "metrics": metrics,
    }


def aggregate(folds):
    results = {}
    rng = np.random.default_rng(24617)
    for sample in ("all", "near_range"):
        available = [row["metrics"] for row in folds if row["metrics"]["observed"][sample] is not None]
        if not available:
            continue
        indices = rng.integers(0, len(available), (10000, len(available)))
        subset = {}
        for method in available[0]:
            values = [row[method][sample] for row in available]
            errors = np.array([value["feature_mse"] for value in values])
            difference = errors - [row["observed"][sample]["feature_mse"] for row in available]
            subset[method] = {
                "n": sum(value["n"] for value in values), "n_subjects": len(values),
                "macro_feature_mse": float(errors.mean()),
                "macro_mse_difference": float(difference.mean()),
                "difference_bootstrap_percentiles": np.percentile(difference[indices].mean(axis=1), [2.5, 97.5]).tolist(),
                "macro_peak_mae_m_s2": float(np.mean([value["peak_mae_m_s2"] for value in values])),
                "label_changes": sum(value["label_changes"] for value in values),
            }
        results[sample] = subset
    return results


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Verified original UCI source archive, optional if cached")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = load_pamap2_pairs(args.archive)
    peak_indices = [index for index, name in enumerate(data.metadata["feature_names"])
                    if name.startswith("max_abs_")]
    if not np.allclose(data.reference[:, peak_indices].max(axis=1), data.peak_reference, rtol=0, atol=1e-12):
        raise ArithmeticError("The declared maximum feature indices do not match the source loader")
    with threadpool_limits(limits=1):
        folds = [evaluate(data, subject, peak_indices) for subject in np.unique(data.subjects)]
    archive = {"protocol_sha256": sha256(PROTOCOL.read_bytes()).hexdigest(),
               "code_sha256": {source: sha256((ROOT / source).read_bytes()).hexdigest() for source in SOURCES},
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "scikit_learn": sklearn.__version__, "thread_limit": 1},
               "source": data.metadata, "holdouts": folds, "aggregate": aggregate(folds)}
    args.output.write_text(json.dumps(archive, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
