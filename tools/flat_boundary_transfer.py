"""Descriptive transfer of the flat-bisector bound; run as tools.flat_boundary_transfer.

Design fixed before this run. Fit one equal-mass class centroid in train-only
standardized coordinates. Use beta=(ratio*maximum centroid distance)^2, ratios
0.35 and 0.60, and horizons 0.15, 0.5 and 1.25. Iris uses three disjoint
stratified row folds after exact feature deduplication. Letter uses the authors'
first 16,000 rows for fitting and disjoint partitions of their final 4,000
rows for evaluation, excluding exact matches to any training feature vector.
HAR uses the official subject-disjoint split and three disjoint blocks of
held-out subjects, with at most 16 windows per subject/activity drawn without
replacement. All partitions and subsampling use seed 211. There is no tuning.

Report paired accuracy changes, endpoint switches, ETD1 agreement and
whole-path certificate coverage. The follow-up guard uses the same settings
and encloses the query's tangential motion before bounding the normal drift.
Check each certificate at 12 DOP853
checkpoints per horizon increment and compare terminal labels to finer RK4.
The same-source folds describe three fixed sources, with no inference over
populations of data sets. The numerical checks sample each path and provide
diagnostics, while the analytic bound concerns the exact flow.
"""

from argparse import ArgumentParser
from hashlib import sha256
from importlib.resources import files
from io import BytesIO
from itertools import combinations
import json
from pathlib import Path
import platform
from time import perf_counter_ns
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import scipy
import sklearn
from scipy.spatial.distance import cdist
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from crossing.flow import MixtureFlow, reference_path, rk4


SEED = 211
FOLDS = 3
HAR_WINDOWS_PER_SUBJECT_CLASS = 16
HORIZONS = (0.15, 0.5, 1.25)
RATIOS = (0.35, 0.60)
CHECKPOINTS_PER_INCREMENT = 12
REFERENCE_BATCH_SIZE = 64
CHECKPOINTS = np.concatenate([
    np.linspace(start, end, CHECKPOINTS_PER_INCREMENT + 1)[1:]
    for start, end in zip((0.0, *HORIZONS[:-1]), HORIZONS, strict=True)
])
IRIS_SHA256 = "f9e3809b56cc0f2f2f69554b7a0ea04e94d0e2c21343c86593eaf5581551dab6"
SOURCES = {
    "iris": {"citation": "Fisher (1936), Iris, UCI", "doi": "https://doi.org/10.24432/C56C76",
             "sha256": IRIS_SHA256, "file": "scikit-learn 1.8.0 packaged iris.csv"},
    "letter": {
        "citation": "Slate (1991), Letter Recognition, UCI",
        "doi": "https://doi.org/10.24432/C5ZP40", "file": "letter-recognition.data",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data",
        "sha256": "2b89f3602cf768d3c8355267d2f13f2417809e101fc2b5ceee10db19a60de6e2",
    },
    "har": {
        "citation": "Reyes-Ortiz et al. (2013), Human Activity Recognition Using Smartphones, UCI",
        "doi": "https://doi.org/10.24432/C54S4K",
        "file": "human+activity+recognition+using+smartphones.zip",
        "url": "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
        "sha256": "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031",
    },
}


def read_source(cache, dataset):
    source = SOURCES[dataset]
    path = cache / source["file"]
    if path.exists():
        payload = path.read_bytes()
    else:
        with urlopen(source["url"], timeout=90) as response:
            payload = response.read()
    check_hash(payload, source["sha256"], path.name)
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return payload


def check_hash(payload, expected, name):
    if sha256(payload).hexdigest() != expected:
        raise ValueError(f"Source SHA-256 mismatch for {name}")


def folds_of_rows(features, labels):
    return StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(features, labels)


def iris_cases():
    payload = files("sklearn.datasets.data").joinpath("iris.csv").read_bytes()
    check_hash(payload, IRIS_SHA256, "iris.csv")
    x, y = load_iris(return_X_y=True)
    unique = np.sort(np.unique(x, axis=0, return_index=True)[1])
    x, y = x[unique], y[unique]
    for train, test in folds_of_rows(x, y):
        yield x[train], y[train], x[test], y[test], {"n_duplicate_rows_removed": 150 - len(x)}


def letter_cases(cache):
    content = read_source(cache, "letter")
    fields = np.loadtxt(BytesIO(content), delimiter=",", dtype="S3")
    x = fields[:, 1:].astype(float)
    y = np.frombuffer(fields[:, 0].astype("S1").tobytes(), dtype=np.uint8) - ord("A")
    x_fit, y_fit = x[:16_000], y[:16_000]
    observed = {row.tobytes() for row in x_fit}
    holdout = np.asarray([row for row in range(16_000, len(x)) if x[row].tobytes() not in observed])
    for _, indices in folds_of_rows(x[holdout], y[holdout]):
        test = holdout[indices]
        yield x_fit, y_fit, x[test], y[test], {"n_test_rows_removed_as_training_matches": 4_000 - len(holdout)}


def har_cases(cache):
    payload = read_source(cache, "har")
    with ZipFile(BytesIO(payload)) as outer, ZipFile(BytesIO(outer.read("UCI HAR Dataset.zip"))) as archive:
        data = {}
        for split in ("train", "test"):
            with archive.open(f"UCI HAR Dataset/{split}/X_{split}.txt") as path:
                x = np.loadtxt(path)
            with archive.open(f"UCI HAR Dataset/{split}/y_{split}.txt") as path:
                y = np.loadtxt(path, dtype=int)
            with archive.open(f"UCI HAR Dataset/{split}/subject_{split}.txt") as path:
                subject = np.loadtxt(path, dtype=int)
            data[split] = x, y, np.unique(subject), subject
    x_fit, y_fit, train_subjects, _ = data["train"]
    x_test, y_test, test_subjects, subject = data["test"]
    if np.intersect1d(train_subjects, test_subjects).size:
        raise ValueError("UCI HAR training and evaluation subject IDs overlap")
    rng = np.random.default_rng(SEED)
    for held_out in np.array_split(rng.permutation(test_subjects), FOLDS):
        test = np.concatenate([
            rng.permutation(np.flatnonzero((subject == person) & (y_test == label)))[:HAR_WINDOWS_PER_SUBJECT_CLASS]
            for person in held_out for label in np.unique(y_test)
        ])
        yield x_fit, y_fit, x_test[test], y_test[test], {
            "n_training_subjects": len(train_subjects), "test_subject_ids": sorted(held_out.tolist()),
        }


def bisector_offsets(centers):
    """Worst inward drift on each directed bisector, canceling the two tied sites.

    On the bisector the tied equal-mass pair has equal likelihood and opposing
    normal offsets. The net normal contribution of the pair is zero. Every
    other site is bounded individually, even if some are co-projected. On
    the positive side the mixture's normal mean is increasing in the normal
    coordinate since its derivative equals a variance divided by beta.
    """
    n = len(centers)
    distances = cdist(centers, centers)
    inward = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        if distances[i, j] <= 0:
            raise ValueError("Flat bisectors require distinct centroids")
        normal = (centers[i] - centers[j]) / distances[i, j]
        midpoint = (centers[i] + centers[j]) / 2
        offset = (centers - midpoint) @ normal
        offset[[i, j]] = 0.0
        inward[i, j] = max(0.0, -offset.min())
        inward[j, i] = max(0.0, offset.max())
    np.fill_diagonal(distances, 1.0)
    return distances, inward


def bisector_stable(scores, distances, inward, horizon):
    initial = scores.argmax(axis=1)
    margins = (scores[np.arange(len(scores)), initial, None] - scores) / distances[initial]
    required = inward[initial] * np.expm1(horizon)
    clear = margins > required + 1e-10 * (1 + np.abs(margins) + required)
    clear[np.arange(len(scores)), initial] = True
    return clear.all(axis=1)


class LocalPlaneGuard:
    """Bound the normal mixture mean over a query's reachable tangential tube."""

    def __init__(self, model, distances):
        centers = model.centered
        gram = centers @ centers.T
        squared_norms = gram.diagonal()
        site_scores = gram - squared_norms / 2
        normal_offsets = (site_scores.T[:, None] - site_scores.T[None, :]) / distances[..., None]

        i, j = np.indices(distances.shape)
        half_distance = np.where(i == j, 0, distances / 2)
        normal_offsets[i, j, i] = half_distance
        normal_offsets[i, j, j] = -half_distance
        self.normal_offsets = normal_offsets
        self.drift_offsets = normal_offsets.copy()
        # Project each source onto each bisector, then compare its scores
        # to the tied pair. Axes are leader, rival, projected source, site.
        pair_scores = (site_scores.T[:, None] + site_scores.T[None, :]) / 2
        projected_logits = (site_scores - pair_scores[..., None]
                            - normal_offsets[..., :, None] * normal_offsets[..., None, :]) / model.beta
        for tied in (i, j):
            # The tied sites on the plane contribute twice the comparison mass
            # and have canceling normal drift.
            projected_logits[i, j, :, tied] = 0
            self.drift_offsets[i, j, tied] = 0
        self.lower = projected_logits.min(axis=2)
        self.upper = projected_logits.max(axis=2)
        self.beta = model.beta
        self.distances = distances

    def prepare(self, scores):
        rows = np.arange(len(scores))
        rivals = np.arange(scores.shape[1])
        leader = scores.argmax(axis=1)
        normal_distance = (scores[rows, leader, None] - scores) / self.distances[leader]
        pair_scores = (scores[rows, leader, None] + scores) / 2
        initial_logits = (scores[:, None] - pair_scores[..., None]
                          - normal_distance[..., None] * self.normal_offsets[leader]) / self.beta
        initial_logits[rows[:, None], rivals, leader[:, None]] = 0
        initial_logits[:, rivals, rivals] = 0
        return (rows, leader, normal_distance, self.drift_offsets[leader], initial_logits,
                np.minimum(0, self.lower[leader] - initial_logits),
                np.maximum(0, self.upper[leader] - initial_logits))

    @staticmethod
    def stable(prepared, horizon):
        rows, leader, normal_distance, offsets, initial_logits, lo_delta, hi_delta = prepared
        # The tangent position is in (1-h)*y0+h*conv(projected centers),
        # with h <= 1-exp(-horizon). Bound each relative Gaussian logit there.
        h = -np.expm1(-horizon)
        slack = normal_distance / np.expm1(horizon)
        terms = offsets + slack[..., None]
        adverse_logit = initial_logits + h * np.where(terms >= 0, lo_delta, hi_delta)
        adverse_logit -= adverse_logit.max(axis=-1, keepdims=True)
        scaled = terms * np.exp(adverse_logit)
        # Positive lower numerator gives plane normal drift greater than -slack.
        # This floating-point slack is a diagnostic guard, not validated rounding.
        positive = scaled.sum(axis=-1) > 1e-10 * (1 + np.abs(scaled).sum(axis=-1))
        clear = (normal_distance > 1e-10) & positive
        clear[rows, leader] = True
        return clear.all(axis=1)


def measure(model, x, y):
    distances, inward = bisector_offsets(model.centered)
    plane_guard = LocalPlaneGuard(model, distances)
    counts = [dict.fromkeys((
        "n", "static_correct", "endpoint_correct", "endpoint_switches", "etd1_agrees",
        "flat_stable", "local_flat_stable", "covariance_stable", "either_stable",
        "either_plane_stable", "any_stable", "sampled_departures",
        "flat_sampled_departures", "local_flat_sampled_departures", "covariance_sampled_departures",
        "reference_rk4_disagreements",
    ), 0) for _ in HORIZONS]
    for left in range(0, len(x), REFERENCE_BATCH_SIZE):
        points = x[left:left + REFERENCE_BATCH_SIZE]
        labels = y[left:left + REFERENCE_BATCH_SIZE]
        scores = model.scores(points)
        local_planes = plane_guard.prepare(scores)
        leading = scores.argmax(axis=1)
        static = model.classes[leading]
        path, _ = reference_path(model, points, CHECKPOINTS)
        path_scores = model.scores(path.reshape(-1, points.shape[1])).reshape(len(path), len(points), -1)
        path_labels = model.classes[path_scores.argmax(axis=2)]
        initial_class_scores = path_scores[:, np.arange(len(points)), leading].copy()
        path_scores[:, np.arange(len(points)), leading] = -np.inf
        departed_at_checkpoint = path_scores.max(axis=2) >= initial_class_scores
        for i, horizon in enumerate(HORIZONS):
            stop = (i + 1) * CHECKPOINTS_PER_INCREMENT
            endpoint = path_labels[stop - 1]
            departed = departed_at_checkpoint[:stop].any(axis=0)
            flat = bisector_stable(scores, distances, inward, horizon)
            local_flat = plane_guard.stable(local_planes, horizon)
            covariance = model.certify(points, horizon).stable
            approx = model.classes[model.exponential_scores(points, horizon)[1].argmax(axis=1)]
            check, _ = rk4(model, points, horizon, max_step=0.0125)
            metrics = {
                "n": len(points), "static_correct": (static == labels).sum(),
                "endpoint_correct": (endpoint == labels).sum(), "endpoint_switches": (endpoint != static).sum(),
                "etd1_agrees": (approx == endpoint).sum(), "flat_stable": flat.sum(),
                "local_flat_stable": local_flat.sum(), "either_plane_stable": (flat | local_flat).sum(),
                "covariance_stable": covariance.sum(), "either_stable": (flat | covariance).sum(),
                "any_stable": (flat | local_flat | covariance).sum(),
                "sampled_departures": departed.sum(), "flat_sampled_departures": (departed & flat).sum(),
                "local_flat_sampled_departures": (departed & local_flat).sum(),
                "covariance_sampled_departures": (departed & covariance).sum(),
                "reference_rk4_disagreements": (model.predict(check) != endpoint).sum(),
            }
            for name, value in metrics.items():
                counts[i][name] += int(value)
    return counts


def run_cases(cases):
    folds, rows = [], []
    for fold, (x_fit, y_fit, x_test, y_test, audit) in enumerate(cases, start=1):
        scaler = StandardScaler().fit(x_fit)
        training, external = scaler.transform(x_fit), scaler.transform(x_test)
        folds.append({"fold": fold, "n_train": len(training), "n_test": len(external),
                      "feature_dimension": training.shape[1], "n_classes": len(np.unique(y_fit)), **audit})
        for ratio in RATIOS:
            model = MixtureFlow.fit(training, y_fit, prototypes=1, bandwidth_ratio=ratio, seed=SEED)
            for horizon, counts in zip(HORIZONS, measure(model, external, y_test), strict=True):
                rows.append({"fold": fold, "ratio": ratio, "horizon": horizon, "counts": counts})
    overall = []
    for ratio in RATIOS:
        for horizon in HORIZONS:
            subset = [row["counts"] for row in rows if row["ratio"] == ratio and row["horizon"] == horizon]
            totals = {name: sum(row[name] for row in subset) for name in subset[0]}
            n = totals["n"]
            overall.append({
                "ratio": ratio, "horizon": horizon, "counts": totals,
                "endpoint_switch_rate": totals["endpoint_switches"] / n,
                "accuracy_delta": (totals["endpoint_correct"] - totals["static_correct"]) / n,
                "etd1_agreement": totals["etd1_agrees"] / n,
                "flat_stable_coverage": totals["flat_stable"] / n,
                "local_flat_stable_coverage": totals["local_flat_stable"] / n,
                "either_plane_stable_coverage": totals["either_plane_stable"] / n,
                "covariance_stable_coverage": totals["covariance_stable"] / n,
                "either_stable_coverage": totals["either_stable"] / n,
                "any_stable_coverage": totals["any_stable"] / n,
            })
    return {"folds": folds, "overall": overall, "per_fold_counts": rows}


def measure_latency(case, horizon, ratio, max_step, repeats):
    """Time warmed complete prediction batches with and without the local guard."""
    x_fit, y_fit, x_test, _, _ = case
    scaler = StandardScaler().fit(x_fit)
    model = MixtureFlow.fit(scaler.transform(x_fit), y_fit, prototypes=1, bandwidth_ratio=ratio, seed=SEED)
    points = scaler.transform(x_test)
    batches = [points[i:i + REFERENCE_BATCH_SIZE] for i in range(0, len(points), REFERENCE_BATCH_SIZE)]
    distances, inward = bisector_offsets(model.centered)
    local = LocalPlaneGuard(model, distances)

    def reference(batch):
        final, _ = rk4(model, batch, horizon, max_step)
        return model.predict(final)

    def hybrid(batch):
        scores = model.scores(batch)
        prediction = model.classes[scores.argmax(axis=1)]
        needs_solver = ~local.stable(local.prepare(scores), horizon)
        if needs_solver.any():
            prediction[needs_solver] = reference(batch[needs_solver])
        return prediction

    actions = {
        "global_plane_only": lambda: [bisector_stable(model.scores(b), distances, inward, horizon) for b in batches],
        "local_plane_only": lambda: [local.stable(local.prepare(model.scores(b)), horizon) for b in batches],
        "covariance_only": lambda: [model.certify(b, horizon).stable for b in batches],
        "full_rk4": lambda: [reference(b) for b in batches],
        "local_plane_then_rk4": lambda: [hybrid(b) for b in batches],
        "local_plane_geometry_setup": lambda: LocalPlaneGuard(model, distances),
    }
    baseline = np.concatenate(actions["full_rk4"]())
    safeguarded = np.concatenate(actions["local_plane_then_rk4"]())
    for action in actions.values():
        action()
    elapsed = {name: [] for name in actions}
    rng = np.random.default_rng(SEED)
    for _ in range(repeats):
        for name in rng.permutation(list(actions)):
            start = perf_counter_ns()
            actions[name]()
            elapsed[name].append((perf_counter_ns() - start) / 1e6)
    medians = {name: float(np.median(times)) for name, times in elapsed.items()}
    return {
        "test_rows": len(points), "median_ms_one_fold": medians,
        "hybrid_to_rk4_ratio": medians["local_plane_then_rk4"] / medians["full_rk4"],
        "hybrid_to_rk4_label_disagreements": int((safeguarded != baseline).sum()),
    }


def write_result(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, allow_nan=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".research/flat-boundary-data"))
    parser.add_argument("--output", type=Path, default=Path("crossing_results/flat_boundary_transfer.json"))
    parser.add_argument("--timings-output", type=Path,
                        help="Run the separate one-fold latency check and write machine-dependent JSON instead.")
    args = parser.parse_args()
    datasets = (("iris", iris_cases()), ("letter", letter_cases(args.cache)), ("har", har_cases(args.cache)))
    script_sha256 = sha256(Path(__file__).read_bytes()).hexdigest()
    if args.timings_output is not None:
        timing = {"horizon": 0.5, "ratio": 0.35, "max_step": 0.1, "repeats": 15}
        with threadpool_limits(limits=1):
            results = {name: measure_latency(next(cases), **timing) for name, cases in datasets}
        write_result(args.timings_output, {
            "conditions": timing | {
                "benchmark": "first held-out fold; one BLAS thread; warm-up and interleaved passes",
                "max_batch_size": REFERENCE_BATCH_SIZE,
                "setup_excluded_from_prediction_timings": True,
                "host": platform.processor(), "python": platform.python_version(),
            },
            "script_sha256": script_sha256, "timings": results,
        })
        return
    payload = {
        "status": "descriptive post-review external check",
        "source_license": "CC BY 4.0 for all source datasets",
        "sources": SOURCES,
        "script_sha256": script_sha256,
        "software": {"numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "results": {},
    }
    with threadpool_limits(limits=1):
        for name, cases in datasets:
            payload["results"][name] = run_cases(cases)
            print(f"{name}: complete", flush=True)
    write_result(args.output, payload)


if __name__ == "__main__":
    main()
