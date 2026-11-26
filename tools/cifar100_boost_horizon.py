# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "numpy==2.3.2", "scipy==1.15.2", "scikit-learn==1.8.0",
#   "catboost==1.2.10", "optuna==5.1.0", "threadpoolctl==3.6.0",
# ]
# ///
"""Check whether the raw-feature CatBoost comparison is tree-budget limited."""

from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

from catboost import Pool
import numpy as np
from threadpoolctl import threadpool_limits

if __package__:
    from .cifar100_optuna import ROOT, SEED, array_hash, atomic_json, booster, evaluation, features, file_hash, load_arrays
else:
    from cifar100_optuna import ROOT, SEED, array_hash, atomic_json, booster, evaluation, features, file_hash, load_arrays


CHECKPOINTS = (220, 440, 660, 880, 1100)


def experiment(args):
    start = perf_counter()
    source = json.loads(args.selection.read_bytes())
    params = source["best"]["params"]
    images, labels = load_arrays(args.data, "train")
    representation = features(images)
    if array_hash(representation, "<f4") != source["train_features_sha256"]:
        raise ValueError("Raw descriptors changed since the original Optuna run")
    order = np.random.default_rng(SEED).permutation(len(representation))
    val, fit = order[:5_000], order[5_000:]
    validation = Pool(representation[val], labels[val])
    fitting = Pool(representation[fit], labels[fit])
    result = {
        "status": "fitting",
        "study": "More trees for the original validation-selected raw-feature CatBoost parameters",
        "protocol": "Choose the highest validation top1 on the fixed checkpoint grid; earliest checkpoint wins ties",
        "original_selection_sha256": file_hash(args.selection),
        "script_sha256": file_hash(Path(__file__)),
        "optuna_trial": source["best"]["trial"],
        "params": params,
        "validation_checkpoints": list(CHECKPOINTS),
        "fit_rows": len(fit),
        "validation_rows": len(val),
        "resources": {"task": args.task, "threads": args.threads, "gpu_memory_fraction": args.gpu_memory_fraction},
    }
    atomic_json(args.output, result)
    fitted_at = perf_counter()
    model = booster(args, params).fit(fitting)
    result["fit_seconds"] = perf_counter() - fitted_at
    checkpoint_scores = {}
    for trees in CHECKPOINTS:
        probabilities = model.predict_proba(validation, ntree_end=trees, thread_count=args.threads)
        checkpoint_scores[trees], _ = evaluation(probabilities, labels[val])
    selected = max(CHECKPOINTS, key=lambda trees: (checkpoint_scores[trees]["top1"], -trees))
    result.update(status="selected", validation=checkpoint_scores, selected_trees=selected)
    atomic_json(args.output, result)

    images, truth = load_arrays(args.data, "test")
    representation = features(images)
    if source.get("test_features_sha256", array_hash(representation, "<f4")) != array_hash(representation, "<f4"):
        raise ValueError("Test representation changed")
    test = Pool(representation)
    outcomes = {}
    labels_by_checkpoint = {}
    for trees in dict.fromkeys((CHECKPOINTS[0], selected)):
        score = model.predict_proba(test, ntree_end=trees, thread_count=args.threads)
        outcomes[trees], predicted = evaluation(score, truth)
        labels_by_checkpoint[trees] = predicted.tolist()
    atomic_json(args.predictions, {"test_fine_labels": labels_by_checkpoint, "targets_sha256": array_hash(truth, "<i8")})
    result.update(status="complete", test=outcomes, predictions_sha256=file_hash(args.predictions),
                  predictions_json=args.predictions.name, elapsed_seconds=perf_counter()-start)
    atomic_json(args.output, result)
    print(result, flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / ".research/cifar100")
    parser.add_argument("--selection", type=Path, default=ROOT / "crossing_results/cifar100_optuna.json")
    parser.add_argument("--output", type=Path, default=ROOT / "crossing_results/cifar100_boost_horizon.json")
    parser.add_argument("--predictions", type=Path, default=ROOT / "crossing_results/cifar100_boost_horizon_predictions.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--task", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.12)
    args = parser.parse_args()
    args.iterations = CHECKPOINTS[-1]
    with threadpool_limits(limits=args.threads):
        experiment(args)


if __name__ == "__main__":
    main()
