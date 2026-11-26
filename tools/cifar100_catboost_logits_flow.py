"""Test Gaussian prototype transport on centered logits of a frozen CatBoost model."""

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

from catboost import CatBoostClassifier
import numpy as np
from threadpoolctl import threadpool_limits

from .cifar100_flow import SEED, VAL, compare, digest, get_model, search
from .cifar100_optuna import atomic_json


def experiment(args):
    clock = perf_counter()
    model = CatBoostClassifier().load_model(str(args.model))
    if not np.array_equal(model.classes_, np.arange(100)):
        raise ValueError("Expected a CatBoost classifier on the 100 fine classes")

    def logits(embeddings):
        values = model.predict(embeddings, prediction_type="RawFormulaVal",
                               thread_count=args.threads)
        return values - values.mean(axis=1, keepdims=True)

    train_x = np.load(args.embeddings / "e_train.npy", mmap_mode="r")
    train_y = np.load(args.data / "y_train.npy")
    order = np.random.default_rng(SEED).permutation(len(train_y))
    fit, val = order[VAL:], order[:VAL]
    if not np.array_equal(fit, np.load(args.embeddings / "fit_indices.npy")):
        raise ValueError("Encoder fit indices did not match the CatBoost split")
    fit_logits = logits(train_x[fit])
    fit_y = train_y[fit]
    centers = np.stack([fit_logits[fit_y == label].mean(axis=0) for label in range(100)])
    variance = float(np.median(np.square(fit_logits - centers[fit_y]).sum(axis=1)))

    def identity(values):
        return values

    validation_inputs = {
        "clean": logits(train_x[val]),
        "blur": logits(np.load(args.embeddings / "e_val_blur.npy")),
    }
    selections = {name: search(centers, variance, values, train_y[val], identity)
                  for name, values in validation_inputs.items()}
    test_y = np.load(args.data / "y_test.npy")
    for condition, suffix in (("clean", ""), ("blur", "_blur")):
        values = logits(np.load(args.embeddings / f"e_test{suffix}.npy", mmap_mode="r"))
        selection = selections[condition]
        for name in ("selected", "best_nonzero"):
            point = selection[name]
            flow = get_model(centers, point["beta"] or variance)
            selection[f"{name}_test"], _ = compare(values, test_y, identity, flow, point["horizon"])

    return {
        "method": "Gaussian transport after a frozen CatBoost model, using mean-centered raw logits",
        "prototype_fit": "One equal-weight class mean of CatBoost training logits per true class",
        "selection": "Maximize validation top-1 per condition; prefer shorter horizons on ties",
        "seed": SEED, "fit_rows": len(fit), "validation_rows": len(val), "test_rows": len(test_y),
        "tree_count": model.tree_count_, "bandwidth_base": variance,
        "model_sha256": digest(args.model),
        "label_sha256": {name: digest(args.data / f"y_{name}.npy") for name in ("train", "test")},
        "embedding_sha256": {name: digest(args.embeddings / f"e_{name}.npy")
                             for name in ("train", "test", "val_blur", "test_blur")},
        **selections, "elapsed_seconds": perf_counter() - clock,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(".research/cifar100"))
    parser.add_argument("--embeddings", type=Path, default=Path(".research/cifar100/neural-96"))
    parser.add_argument("--output", type=Path, default=Path("crossing_results/cifar100_catboost_logits_flow.json"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        result = experiment(args)
    atomic_json(args.output, result)
    for condition in ("clean", "blur"):
        print(condition, result[condition]["selected"], result[condition]["selected_test"], flush=True)


if __name__ == "__main__":
    main()
