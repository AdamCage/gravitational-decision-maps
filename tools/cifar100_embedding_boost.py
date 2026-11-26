# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "numpy==2.3.2", "scipy==1.15.2", "scikit-learn==1.8.0",
#   "catboost==1.2.10", "optuna==5.1.0", "threadpoolctl==3.6.0",
# ]
# ///
"""Compare CatBoost and the affine CNN head on the same frozen embeddings."""

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

from catboost import CatBoostClassifier, Pool
import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

if __package__:
    from .cifar100_optuna import FIXED, ROOT, SEED, array_hash, atomic_json, evaluation, file_hash, trace
else:
    from cifar100_optuna import FIXED, ROOT, SEED, array_hash, atomic_json, evaluation, file_hash, trace


def features(folder, split, rows):
    vectors = np.load(folder / f"e_{split}.npy", mmap_mode="r")
    if vectors.ndim != 2 or len(vectors) != rows or not np.all(np.isfinite(vectors)):
        raise ValueError(f"Expected {rows} finite ordered embeddings for {split}")
    return vectors


def labels(folder, split, rows):
    truth = np.load(folder / f"y_{split}.npy", mmap_mode="r")
    if truth.shape != (rows,) or truth.dtype != np.int64 or not np.array_equal(np.unique(truth), np.arange(100)):
        raise ValueError(f"Expected official CIFAR-100 fine labels for {split}")
    return truth


def booster(args, params):
    return CatBoostClassifier(loss_function="MultiClass", iterations=args.iterations, depth=params["depth"],
                              learning_rate=params["learning_rate"], l2_leaf_reg=params["l2_leaf_reg"],
                              bagging_temperature=params["bagging_temperature"], border_count=32,
                              task_type="GPU", devices="0", gpu_ram_part=args.gpu_memory_fraction,
                              thread_count=args.threads, random_seed=SEED, allow_writing_files=False,
                              verbose=False, leaf_estimation_method="Gradient", leaf_estimation_iterations=1)


def experiment(args):
    start = perf_counter()
    vectors = features(args.embeddings, "train", 50_000)
    truth = labels(args.data, "train", 50_000)
    order = np.random.default_rng(SEED).permutation(50_000)
    val, fit = order[:5000], order[5000:]
    for kind, indices in (("fit", fit), ("validation", val)):
        saved = np.load(args.embeddings / f"{kind}_indices.npy")
        if not np.array_equal(indices, saved):
            raise ValueError("Encoder was not trained on the prescribed split")
    train = Pool(vectors[fit], truth[fit])
    val_pool = Pool(vectors[val], truth[val])
    with np.load(args.embeddings / "head.npz") as head:
        coefficients, bias = head["coef"], head["intercept"]
        if coefficients.shape != (100, vectors.shape[1]) or not np.array_equal(head["class_ids"], np.arange(100)):
            raise ValueError("Expected the encoder's affine fine-class head")
    design = {
        "status": "tuning",
        "study": "CIFAR-100 matched frozen representation comparison",
        "scope": "Supervised encoder trained on the same fitting images and tuned on the same validation split",
        "embedding_dimension": int(vectors.shape[1]),
        "fit_rows": len(fit),
        "validation_rows": len(val),
        "split_seed": SEED,
        "selection_metric": "validation top1",
        "embedding_run": args.embeddings.name,
        "sha256": {name: file_hash(args.embeddings / name) for name in ("e_train.npy", "head.npz", "encoder.pt")},
        "script_sha256": file_hash(Path(__file__)),
        "resources": {"task": "GPU", "threads": args.threads, "gpu_memory_fraction": args.gpu_memory_fraction,
                      "iterations": args.iterations, "trial_cap": args.trials, "timeout_s": args.timeout},
        "search": {"sampler": "Optuna TPESampler", "seed": SEED, "random_startup_trials": 3,
                   "depth": [4, 7], "learning_rate_log": [0.07, 0.38], "l2_leaf_reg_log": [0.5, 24.0],
                   "bagging_temperature": [0, 2.5]},
    }
    atomic_json(args.output, design)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=3))

    def objective(trial):
        params = {"depth": trial.suggest_int("depth", 4, 7),
                  "learning_rate": trial.suggest_float("learning_rate", 0.07, 0.38, log=True),
                  "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 24.0, log=True),
                  "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 2.5)}
        model = booster(args, params)
        model.fit(train)
        return float(np.mean(model.predict(val_pool, thread_count=args.threads).ravel() == truth[val]))

    def save_trial(current, _):
        rows = trace(current)
        design.update(trials=rows, completed_trials=sum(row["state"] == "COMPLETE" for row in rows))
        atomic_json(args.output, design)

    tuning_start = perf_counter()
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, callbacks=(save_trial,))
    design["tuning_seconds"] = perf_counter() - tuning_start
    best = study.best_trial
    design["best"] = {"trial": best.number, "params": best.params, "validation_top1": best.value}
    design["status"] = "fitting_final_models"
    atomic_json(args.output, design)

    started = perf_counter()
    tuned = booster(args, best.params).fit(train)
    tuned_sec = perf_counter() - started
    started = perf_counter()
    fixed = booster(args, FIXED).fit(train)
    fixed_sec = perf_counter() - started
    started = perf_counter()
    scale = StandardScaler()
    linear = LogisticRegression(C=1, tol=5e-3, max_iter=180, solver="lbfgs", random_state=SEED)
    linear.fit(scale.fit_transform(vectors[fit]), truth[fit])
    linear_sec = perf_counter() - started

    test = features(args.embeddings, "test", 10_000)
    y_test = labels(args.data, "test", 10_000)
    design["sha256"]["e_test.npy"] = file_hash(args.embeddings / "e_test.npy")
    test_pool = Pool(test)
    records = {
        "original_affine": (vectors[val] @ coefficients.T + bias, test @ coefficients.T + bias, 0),
        "refit_linear": (linear.predict_proba(scale.transform(vectors[val])),
                         linear.predict_proba(scale.transform(test)), linear_sec),
    }
    for name, model, elapsed in (("fixed_catboost", fixed, fixed_sec), ("optuna_catboost", tuned, tuned_sec)):
        records[name] = (model.predict_proba(val_pool, thread_count=args.threads),
                         model.predict_proba(test_pool, thread_count=args.threads), elapsed)
    results = {}
    predicted = {}
    for name, (valid_scores, test_scores, fitting_sec) in records.items():
        validation, _ = evaluation(valid_scores, truth[val])
        official_test, predictions = evaluation(test_scores, y_test)
        results[name] = {"validation": validation, "test": official_test, "head_fit_seconds": fitting_sec,
                         "test_predictions_sha256": array_hash(predictions, "<i8")}
        predicted[name] = predictions.tolist()
    atomic_json(args.predictions, {"order": "official CIFAR-100 test order",
                                  "targets_sha256": array_hash(y_test, "<i8"), "predicted_fine_labels": predicted})
    design.update(status="complete", elapsed_seconds=perf_counter()-start, metrics=results,
                  predictions_json=args.predictions.name, predictions_sha256=file_hash(args.predictions))
    atomic_json(args.output, design)
    print(results, flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / ".research/cifar100")
    parser.add_argument("--embeddings", type=Path, default=ROOT / ".research/cifar100/neural-96")
    parser.add_argument("--output", type=Path, default=ROOT / "crossing_results/cifar100_embedding_boost.json")
    parser.add_argument("--predictions", type=Path, default=ROOT / "crossing_results/cifar100_embedding_boost_predictions.json")
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.12)
    args = parser.parse_args()
    if min(args.iterations, args.trials, args.threads, args.timeout) < 1:
        parser.error("Invalid fitting budget")
    with threadpool_limits(limits=args.threads):
        experiment(args)


if __name__ == "__main__":
    main()
