# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "numpy==2.3.2", "scipy==1.15.2", "scikit-learn==1.8.0",
#   "catboost==1.2.10", "optuna==5.1.0", "threadpoolctl==3.6.0",
# ]
# ///
"""Reproducible CIFAR-100 gradient boosting on deterministic image features."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json
import platform
from time import perf_counter

from catboost import CatBoostClassifier, Pool
import catboost
import numpy as np
import optuna
from optuna.trial import TrialState
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sklearn
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261127
VALIDATION_ROWS = 5_000
HOG_BINS = 9
FIXED = {"depth": 5, "learning_rate": 0.12, "l2_leaf_reg": 3.0, "bagging_temperature": 1.0}
FEATURES = {
    "rgb_grid_mean_8x8": 192,
    "rgb_grid_std_4x4": 48,
    "unsigned_hog_4x4": 144,
    "uniform_lbp_2x2": 40,
    "rgb_global_histogram": 24,
    "gradient_magnitude_4x4": 16,
}


def file_hash(path):
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values, dtype=None):
    return sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def within_cells(images, cell):
    n, h, w = images.shape[:3]
    return images.reshape(n, h // cell, cell, w // cell, cell, *images.shape[3:])


def pooled(images, cell):
    return within_cells(images, cell).mean(axis=(2, 4))


def grayscale_and_gradients(images):
    weight = np.asarray((0.299, 0.587, 0.114), dtype=np.float32)
    gray = (images * weight).sum(axis=-1)
    pad = np.pad(gray, ((0, 0), (1, 1), (1, 1)), mode="edge")
    gx = pad[:, 1:-1, 2:] - pad[:, 1:-1, :-2]
    gy = pad[:, 2:, 1:-1] - pad[:, :-2, 1:-1]
    return gray, pad, gx, gy


def hog(gx, gy, magnitude):
    direction = (np.arctan2(gy, gx) * np.float32(HOG_BINS / np.pi)) % HOG_BINS
    index = np.floor(direction).astype(np.int32) % HOG_BINS
    fraction = direction - np.floor(direction)
    left = magnitude * (1 - fraction)
    right = magnitude * fraction
    next_index = (index + 1) % HOG_BINS
    hist = np.stack([pooled(np.where(index == b, left, 0) + np.where(next_index == b, right, 0), 8)
                     for b in range(HOG_BINS)], axis=-1)
    scale = np.sqrt(np.sum(hist**2, axis=-1, keepdims=True) + 1e-8)
    hist = np.minimum(hist / scale, 0.25)
    return hist / np.sqrt(np.sum(hist**2, axis=-1, keepdims=True) + 1e-8)


def lbp(gray, pad):
    offsets = ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0))
    bits = [pad[:, y:y + 32, x:x + 32] >= gray for y, x in offsets]
    transitions = sum((bits[i] != bits[i - 1] for i in range(8)))
    count = sum(bits)
    pattern = np.where(transitions <= 2, count, 9)
    return np.stack([pooled(pattern == b, 16) for b in range(10)], axis=-1)


def features(images, chunk=256):
    output = np.empty((len(images), sum(FEATURES.values())), dtype=np.float32)
    for start in range(0, len(images), chunk):
        raw = np.asarray(images[start:start + chunk])
        x = raw.astype(np.float32) / np.float32(255)
        mean = pooled(x, 8)
        std = np.sqrt(np.maximum(0, pooled(x**2, 8) - mean**2))
        gray, pad, gx, gy = grayscale_and_gradients(x)
        magnitude = np.hypot(gx, gy)
        hist = np.stack([((raw >> 5) == b).mean(axis=(1, 2)) for b in range(8)], axis=-1)
        parts = (pooled(x, 4), std, hog(gx, gy, magnitude), lbp(gray, pad), hist, pooled(magnitude, 8))
        output[start:start + len(raw)] = np.concatenate([part.reshape(len(raw), -1) for part in parts], axis=-1)
    return output


def load_arrays(data, split):
    x = np.load(data / f"x_{split}.npy", mmap_mode="r")
    y = np.load(data / f"y_{split}.npy", mmap_mode="r")
    rows = 50_000 if split == "train" else 10_000
    if x.shape != (rows, 32, 32, 3) or x.dtype != np.uint8:
        raise ValueError(f"Expected {rows} RGB CIFAR-100 rows in NHWC uint8")
    if y.shape != (rows,) or np.any(y < 0) or np.any(y > 99) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Expected fine integer targets in the official order")
    return x, y


def booster(args, params):
    kwargs = {"devices": "0", "gpu_ram_part": args.gpu_memory_fraction} if args.task == "GPU" else {}
    return CatBoostClassifier(
        loss_function="MultiClass", iterations=args.iterations, thread_count=args.threads,
        border_count=32, task_type=args.task, random_seed=SEED, allow_writing_files=False, verbose=False,
        leaf_estimation_method="Gradient", leaf_estimation_iterations=1, **kwargs, **params,
    )


def evaluation(probabilities, y):
    ranked = np.argsort(-probabilities, axis=1, kind="stable")[:, :5]
    predicted = ranked[:, 0]
    return {
        "top1": float(np.mean(predicted == y)),
        "top5": float(np.mean((ranked == y[:, None]).any(axis=1))),
        "n": len(y),
    }, predicted


def trace(study):
    return [
        {"number": trial.number, "state": trial.state.name, "params": trial.params, "validation_top1": trial.value,
         "seconds": trial.duration.total_seconds() if trial.duration is not None else None}
        for trial in study.trials
    ]


def experiment(args):
    x, y = load_arrays(args.data, "train")
    started = perf_counter()
    design = {
        "status": "extracting_train_features",
        "script_sha256": file_hash(Path(__file__)),
        "feature_spec": FEATURES,
        "validation_split": {"seed": SEED, "method": "np.random.default_rng(seed).permutation(50000), first 5000",
                             "selection_metric": "validation top1 at fixed iteration count"},
        "fixed_catboost": FIXED,
        "search": {"sampler": "Optuna TPESampler", "sampler_seed": SEED, "random_startup_trials": 3,
                   "depth": [4, 7], "learning_rate_log": [0.035, 0.24], "l2_leaf_reg_log": [0.5, 32.0],
                   "bagging_temperature": [0, 3.0], "trial_cap": args.trials, "timeout_s": args.timeout},
        "resources": {"task": args.task, "threads": args.threads, "iterations": args.iterations,
                      "gpu_memory_fraction": args.gpu_memory_fraction if args.task == "GPU" else None},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
                        "optuna": optuna.__version__, "catboost": catboost.__version__, "system": platform.platform()},
        "source_sha256": {f"{key}_train.npy": file_hash(args.data / f"{key}_train.npy") for key in ("x", "y")},
    }
    atomic_json(args.output, design)
    x = features(x)
    design["train_feature_seconds"] = perf_counter() - started
    design["train_features_sha256"] = array_hash(x, "<f4")
    split = np.random.default_rng(SEED).permutation(len(x))
    val_indices, fit_indices = split[:VALIDATION_ROWS], split[VALIDATION_ROWS:]
    tune_indices = fit_indices[:args.tune_rows]
    design["split_sha256"] = {name: array_hash(ids, "<i8") for name, ids in
                              (("fit", fit_indices), ("validation", val_indices), ("tuning_fit", tune_indices))}
    design["fit_rows"], design["tune_fit_rows"] = len(fit_indices), len(tune_indices)
    y_val = y[val_indices]
    validation = Pool(x[val_indices], y_val)
    tuning = Pool(x[tune_indices], y[tune_indices])
    fitting = Pool(x[fit_indices], y[fit_indices])
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=3))

    def objective(trial):
        params = {
            "depth": trial.suggest_int("depth", 4, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.035, 0.24, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 32, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 3),
        }
        model = booster(args, params)
        model.fit(tuning)
        return float(np.mean(model.predict(validation).ravel() == y_val))

    def save_trial(current, _):
        count = len(current.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
        design.update(status="tuning", completed_trials=count, trials=trace(current))
        atomic_json(args.output, design)

    tuning_start = perf_counter()
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, callbacks=(save_trial,))
    design["tuning_seconds"] = perf_counter() - tuning_start
    best = study.best_trial
    design["best"] = {"trial": best.number, "params": best.params, "validation_top1": best.value}
    design["status"] = "fitting_final_models"
    atomic_json(args.output, design)

    models = {}
    fit_seconds = {}
    for name, params in (("fixed_catboost", FIXED), ("optuna_catboost", best.params)):
        start = perf_counter()
        model = booster(args, params)
        model.fit(fitting)
        fit_seconds[name] = perf_counter() - start
        models[name] = model
        print(f"{name} fitted in {fit_seconds[name]:.2f}s", flush=True)
    start = perf_counter()
    scaler = StandardScaler()
    linear = LogisticRegression(C=1, tol=5e-3, max_iter=180, solver="lbfgs", random_state=SEED)
    linear.fit(scaler.fit_transform(x[fit_indices]), y[fit_indices])
    fit_seconds["linear"] = perf_counter() - start
    print(f"linear fitted in {fit_seconds['linear']:.2f}s", flush=True)

    # Test files are first opened after Optuna selection and all final fits.
    x_test, y_test = load_arrays(args.data, "test")
    start = perf_counter()
    x_test = features(x_test)
    design["test_feature_seconds"] = perf_counter() - start
    design["test_features_sha256"] = array_hash(x_test, "<f4")
    design["source_sha256"].update({f"{k}_test.npy": file_hash(args.data / f"{k}_test.npy") for k in ("x", "y")})
    predict_start = perf_counter()
    test_pool = Pool(x_test)
    probabilities = {
        name: (model.predict_proba(validation, thread_count=args.threads),
               model.predict_proba(test_pool, thread_count=args.threads))
        for name, model in models.items()
    }
    probabilities["linear"] = (
        linear.predict_proba(scaler.transform(x[val_indices])), linear.predict_proba(scaler.transform(x_test)),
    )
    predictions = {}
    scores = {}
    for method, (p_val, p_test) in probabilities.items():
        val_score, _ = evaluation(p_val, y_val)
        test_score, predicted = evaluation(p_test, y_test)
        predictions[method] = predicted
        scores[method] = {"validation": val_score, "test": test_score, "fit_seconds": fit_seconds[method],
                          "test_prediction_sha256": array_hash(predicted, "<i8")}
    pairs = {}
    for name in ("fixed_catboost", "linear"):
        tuned = predictions["optuna_catboost"] == y_test
        comparison = predictions[name] == y_test
        pairs[name] = {"tuned_only_correct": int(np.count_nonzero(tuned & ~comparison)),
                       "comparator_only_correct": int(np.count_nonzero(comparison & ~tuned))}
    atomic_json(args.predictions, {"order": "official CIFAR-100 test order",
                                  "targets_sha256": array_hash(y_test, "<i8"),
                                  "predicted_fine_labels": {m: p.tolist() for m, p in predictions.items()}})
    design.update(status="complete", metrics=scores, paired_test=pairs, test_evaluation_seconds=perf_counter()-predict_start,
                  elapsed_seconds=perf_counter()-started, predictions_json=args.predictions.name,
                  predictions_sha256=file_hash(args.predictions))
    atomic_json(args.output, design)
    print(json.dumps(scores, indent=2), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / ".research/cifar100")
    parser.add_argument("--output", type=Path, default=ROOT / "crossing_results/cifar100_optuna.json")
    parser.add_argument("--predictions", type=Path, default=ROOT / "crossing_results/cifar100_optuna_predictions.json")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=220)
    parser.add_argument("--tune-rows", type=int, default=45_000)
    parser.add_argument("--task", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.12)
    args = parser.parse_args()
    if not 100 <= args.tune_rows <= 45_000 or min(args.trials, args.timeout, args.threads, args.iterations) < 1:
        parser.error("Invalid fitting budget")
    with threadpool_limits(limits=args.threads):
        experiment(args)


if __name__ == "__main__":
    main()
