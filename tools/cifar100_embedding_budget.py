# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "numpy==2.3.2", "scipy==1.15.2", "scikit-learn==1.8.0",
#   "catboost==1.2.10", "optuna==5.1.0", "threadpoolctl==3.6.0",
# ]
# ///
"""Exploratory longer tree-budget control on the frozen CNN representation."""

from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

from catboost import Pool
import numpy as np
from threadpoolctl import threadpool_limits

if __package__:
    from .cifar100_embedding_boost import booster, features, labels
    from .cifar100_optuna import ROOT, SEED, array_hash, atomic_json, evaluation, file_hash
else:
    from cifar100_embedding_boost import booster, features, labels
    from cifar100_optuna import ROOT, SEED, array_hash, atomic_json, evaluation, file_hash


CHECKPOINTS = (160, 320, 480, 640, 800, 960)


def experiment(args):
    start = perf_counter()
    prior = json.loads(args.selection.read_bytes())
    if prior["status"] != "complete" or prior["resources"]["iterations"] != CHECKPOINTS[0]:
        raise ValueError("Expected the completed 160-tree embedding Optuna run")
    if file_hash(args.embeddings / "e_train.npy") != prior["sha256"]["e_train.npy"]:
        raise ValueError("Training representation differed from the matched experiment")
    vectors = features(args.embeddings, "train", 50_000)
    truth = labels(args.data, "train", len(vectors))
    order = np.random.default_rng(SEED).permutation(len(vectors))
    val, fit = order[:5000], order[5000:]
    pool = Pool(vectors[fit], truth[fit])
    validation = Pool(vectors[val], truth[val])
    result = {
        "status": "fitting",
        "study": "Exploratory longer tree-budget control with the matched encoder and validation-selected hyperparameters",
        "checkpoint_selection": "validation top1, earliest checkpoint on ties",
        "checkpoints": list(CHECKPOINTS),
        "source_metrics_sha256": file_hash(args.selection),
        "script_sha256": file_hash(Path(__file__)),
        "embedding_run": args.embeddings.name,
        "params": prior["best"]["params"],
        "resources": {"threads": args.threads, "gpu_memory_fraction": args.gpu_memory_fraction},
    }
    atomic_json(args.output, result)
    fitted_at = perf_counter()
    model = booster(args, prior["best"]["params"]).fit(pool)
    result["fit_seconds"] = perf_counter() - fitted_at
    checkpoint_scores = {}
    for trees in CHECKPOINTS:
        probability = model.predict_proba(validation, ntree_end=trees, thread_count=args.threads)
        checkpoint_scores[trees], _ = evaluation(probability, truth[val])
    selected = max(CHECKPOINTS, key=lambda trees: (checkpoint_scores[trees]["top1"], -trees))
    result.update(status="selected", selected_trees=selected, validation=checkpoint_scores)
    atomic_json(args.output, result)

    if file_hash(args.embeddings / "e_test.npy") != prior["sha256"]["e_test.npy"]:
        raise ValueError("Test representation differed from the matched experiment")
    testing = Pool(features(args.embeddings, "test", 10_000))
    truth = labels(args.data, "test", 10_000)
    scores = {}
    predictions = {}
    for trees in dict.fromkeys((CHECKPOINTS[0], selected)):
        probability = model.predict_proba(testing, ntree_end=trees, thread_count=args.threads)
        scores[trees], predicted = evaluation(probability, truth)
        predictions[trees] = predicted.tolist()
    atomic_json(args.predictions, {"fine_labels_by_tree_count": predictions, "targets_sha256": array_hash(truth, "<i8")})
    result.update(status="complete", test=scores, predictions_json=args.predictions.name,
                  predictions_sha256=file_hash(args.predictions), elapsed_seconds=perf_counter()-start)
    atomic_json(args.output, result)
    print(result, flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / ".research/cifar100")
    parser.add_argument("--embeddings", type=Path, default=ROOT / ".research/cifar100/neural-96")
    parser.add_argument("--selection", type=Path, default=ROOT / "crossing_results/cifar100_embedding_boost.json")
    parser.add_argument("--output", type=Path, default=ROOT / "crossing_results/cifar100_embedding_budget.json")
    parser.add_argument("--predictions", type=Path, default=ROOT / "crossing_results/cifar100_embedding_budget_predictions.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.12)
    args = parser.parse_args()
    args.iterations = CHECKPOINTS[-1]
    with threadpool_limits(limits=args.threads):
        experiment(args)


if __name__ == "__main__":
    main()
