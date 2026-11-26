"""Evaluate a frozen CatBoost head before and after embedding transport."""

from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

from catboost import CatBoostClassifier, Pool
import numpy as np
from threadpoolctl import threadpool_limits

from crossing.flow import rk4
from tools.cifar100_embedding_boost import booster, features, labels
from tools.cifar100_flow import HORIZONS, STEP, get_model, metrics, paired
from tools.cifar100_optuna import ROOT, SEED, array_hash, atomic_json, file_hash


FACTORS = (0.0625, 0.125, 0.25)
TREE_COUNTS = (160, 960)


def logits(model, vectors, count, threads):
    return model.predict(vectors, prediction_type="RawFormulaVal", ntree_end=count, thread_count=threads)


def along_path(centers, beta, x, horizon):
    model = get_model(centers, beta)
    previous = 0.0
    for endpoint in HORIZONS:
        if endpoint > horizon:
            break
        x, _ = rk4(model, x, endpoint - previous, max_step=STEP)
        previous = endpoint
        yield endpoint, x


def run(args):
    start = perf_counter()
    prior = json.loads(args.selection.read_bytes())
    if prior["status"] != "complete" or file_hash(args.embeddings / "e_train.npy") != prior["sha256"]["e_train.npy"]:
        raise ValueError("Matching completed Optuna run and encoder required")
    args.iterations = TREE_COUNTS[-1]
    truth = labels(args.data, "train", 50_000)
    train = features(args.embeddings, "train", len(truth))
    split = np.random.default_rng(SEED).permutation(len(truth))
    val, fit = split[:5000], split[5000:]
    if not np.array_equal(fit, np.load(args.embeddings / "fit_indices.npy")):
        raise ValueError("Original fit split did not match")
    design = {"status": "fitting", "fit_rows": len(fit), "validation_rows": len(val),
              "embedding_run": args.embeddings.name, "tree_counts": TREE_COUNTS,
              "optuna_trial": prior["best"]["trial"], "catboost_parameters": prior["best"]["params"],
              "transport": {"grid_factors": FACTORS, "grid_horizons": (0.0, *HORIZONS), "rk4_max_step": STEP,
                            "selection_metric": "validation top1, smaller horizon on ties",
                            "geometry": "equal-mass Gaussian with fit-only class centroid per class",
                            "claim_scope": "Unconstrained transport observed on a nonlinear frozen head"},
              "train_embeddings_sha256": prior["sha256"]["e_train.npy"],
              "script_sha256": file_hash(Path(__file__))}
    if args.reuse_model:
        stored = json.loads(args.reuse_model.read_bytes())
        identity_fields = ("train_embeddings_sha256", "optuna_trial", "catboost_parameters", "embedding_run")
        if (any(design[key] != stored[key] for key in identity_fields)
                or file_hash(args.model) != stored["model_sha256"]):
            raise ValueError("Saved model does not match this experiment")
        frozen = CatBoostClassifier().load_model(args.model)
        if frozen.tree_count_ != TREE_COUNTS[-1]:
            raise ValueError("Saved model does not have the declared tree count")
        fit_seconds = stored["fit_seconds"]
    else:
        atomic_json(args.output, design)
        fit_start = perf_counter()
        frozen = booster(args, prior["best"]["params"]).fit(Pool(train[fit], truth[fit]))
        fit_seconds = perf_counter() - fit_start
        args.model.parent.mkdir(parents=True, exist_ok=True)
        pending = args.model.with_suffix(".pending.cbm")
        frozen.save_model(pending)
        pending.replace(args.model)
    design.update(status="validation", fit_seconds=fit_seconds, reused_model=bool(args.reuse_model),
                  model_sha256=file_hash(args.model))
    atomic_json(args.output, design)
    print(f"Saved {args.model}", flush=True)

    center_data = train[fit].astype(np.float64)
    fit_labels = truth[fit]
    centers = np.stack([center_data[fit_labels == k].mean(axis=0) for k in range(100)])
    variance = float(np.median(np.square(center_data - centers[fit_labels]).sum(axis=1)))
    validation = train[val].astype(np.float64)
    rows = {count: [] for count in TREE_COUNTS}

    def record(x, factor, beta, t):
        for count in TREE_COUNTS:
            predicted = logits(frozen, x, count, args.threads).argmax(axis=1)
            rows[count].append({"factor": factor, "beta": beta, "horizon": t,
                                "top1": float(np.mean(predicted == truth[val]))})

    record(validation, None, None, 0.0)
    for factor in FACTORS:
        beta = factor * variance
        for t, shifted in along_path(centers, beta, validation, HORIZONS[-1]):
            record(shifted, factor, beta, t)
    choices = {}
    for count, options in rows.items():
        ranked = sorted(options, key=lambda row: (-row["top1"], row["horizon"]))
        choices[count] = {"selected": ranked[0], "best_nonzero": next(row for row in ranked if row["horizon"] > 0)}
    design.update(status="selected", bandwidth_base=variance, validation_grid=rows, choices=choices)
    atomic_json(args.output, design)

    test = features(args.embeddings, "test", 10_000).astype(np.float64)
    truth = labels(args.data, "test", len(test))
    test_paths = {(None, 0.0): test}
    one_steps = {(None, 0.0): test}
    result = {}
    for count, pair in choices.items():
        original = logits(frozen, test, count, args.threads)
        result[count] = {"original": metrics(original, truth),
                         "original_argmax_sha256": array_hash(original.argmax(axis=1), "<i8")}
        for name, setting in pair.items():
            beta, horizon = setting["beta"], setting["horizon"]
            key = beta, horizon
            if key not in test_paths:
                for t, x in along_path(centers, beta, test, horizon):
                    test_paths[beta, t] = x
            if key not in one_steps:
                model = get_model(centers, beta)
                one_steps[key] = test - np.expm1(-horizon) * model.field(test)
            proposed = original if horizon == 0 else logits(frozen, test_paths[key], count, args.threads)
            one = original if horizon == 0 else logits(frozen, one_steps[key], count, args.threads)
            result[count][name] = {
                "metrics": metrics(proposed, truth), "paired": paired(original, proposed, truth),
                "rk4_argmax_sha256": array_hash(proposed.argmax(axis=1), "<i8"),
                "exponential_euler": {"metrics": metrics(one, truth), "paired": paired(original, one, truth),
                                      "argmax_sha256": array_hash(one.argmax(axis=1), "<i8")},
                "rk4_relative_to_exponential_euler": paired(one, proposed, truth),
            }
        design.update(status="testing", official_test=result)
        atomic_json(args.output, design)
    design.update(status="complete", test_rows=len(test), elapsed_seconds=perf_counter()-start,
                  test_embeddings_sha256=file_hash(args.embeddings / "e_test.npy"))
    atomic_json(args.output, design)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / ".research/cifar100")
    parser.add_argument("--embeddings", type=Path, default=ROOT / ".research/cifar100/neural-96")
    parser.add_argument("--selection", type=Path, default=ROOT / "crossing_results/cifar100_embedding_boost.json")
    parser.add_argument("--output", type=Path, default=ROOT / "crossing_results/cifar100_catboost_flow.json")
    parser.add_argument("--model", type=Path, default=ROOT / ".research/cifar100-boost/neural-gdm.cbm")
    parser.add_argument("--reuse-model", type=Path, metavar="PRIOR_JSON", help="Verify and reuse the model from this run")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.12)
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        run(args)
