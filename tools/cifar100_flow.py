"""Replay the frozen-head Gaussian transport probe on CIFAR-100 embeddings.

The affine head and native image encoder are held fixed. Fit one spherical
prototype per fine class on the fit rows only. Clean accuracy and paired
blur recovery select settings independently on the validation rows.
"""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from crossing.affine_reserve import AffineReadout
from crossing.flow import MixtureFlow, rk4
from crossing.reachable_hull import ReachableHullGuard

SEED = 20261127
VAL = 5000
FACTORS = (0.0625, 0.125, 0.25, 0.5, 1.0)
HORIZONS = (0.1, 0.3, 0.6, 1.0)
PROBE = 0.3
STEP = 0.05
RESERVE = 1e-4


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def scores(x, weights, bias):
    return x @ weights.T + bias


def metrics(logits, y):
    correct = logits.argmax(axis=1) == y
    five = np.argpartition(logits, -5, axis=1)[:, -5:]
    nll = np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(y)), y])
    return {"top1": float(correct.mean()), "top5": float(np.any(five == y[:, None], axis=1).mean()),
            "nll": float(nll)}


def paired(original, proposed, y):
    before, after = original.argmax(axis=1), proposed.argmax(axis=1)
    diff = (after == y).astype(int) - (before == y)
    counts = np.bincount(diff + 1, minlength=3)
    draw = np.random.default_rng(SEED).multinomial(len(y), counts / len(y), size=5000)
    delta_ci = np.percentile((draw[:, 2] - draw[:, 0]) / len(y), [2.5, 97.5]).tolist()
    return {"changes": int(np.count_nonzero(before != after)), "wrong_to_right": int(counts[2]),
            "right_to_wrong": int(counts[0]), "accuracy_delta": float(diff.mean()),
            "paired_bootstrap_interval": delta_ci}


def get_model(centers, beta):
    count = len(centers)
    return MixtureFlow(centers, np.arange(count), np.ones(count), beta)


def curve(model, x):
    last = 0.0
    result = x
    for horizon in HORIZONS:
        result, _ = rk4(model, result, horizon - last, max_step=STEP)
        last = horizon
        yield horizon, result


def loss(x, reference):
    return float(np.mean(np.square(x - reference)))


def search(centers, variance, x, y, head, reference=None):
    """All candidate settings and selection data are confined to validation."""
    objective = (lambda z: loss(z, reference)) if reference is not None else (
        lambda z: float(np.mean(head(z).argmax(axis=1) == y)))
    baseline = {"factor": None, "beta": None, "horizon": 0.0, "objective": objective(x)}
    table = [baseline]
    for factor in FACTORS:
        beta = variance * factor
        model = get_model(centers, beta)
        for horizon, z in curve(model, x):
            table.append({"factor": factor, "beta": beta, "horizon": horizon, "objective": objective(z)})
    if reference is None:
        best_first = sorted(table, key=lambda row: (-row["objective"], row["horizon"]))
    else:
        best_first = sorted(table, key=lambda row: (row["objective"], row["horizon"]))
    best_nonzero = next(row for row in best_first if row["horizon"] > 0)
    return {"grid": table, "selected": best_first[0], "best_nonzero": best_nonzero}


def compare(x, y, head, model, horizon, reference=None, step=STEP):
    original = head(x)
    if horizon:
        endpoint, evaluation_rounds = rk4(model, x, horizon, max_step=step)
        one_step = x + (-np.expm1(-horizon)) * model.field(x)
    else:
        endpoint, one_step, evaluation_rounds = x, x, 0
    result = {"raw": metrics(original, y), "rk4": metrics(head(endpoint), y),
              "exponential_euler": metrics(head(one_step), y), "paired_rk4": paired(original, head(endpoint), y),
              "paired_exponential_euler": paired(original, head(one_step), y),
              "rk4_field_evaluation_rounds": evaluation_rounds}
    if reference is not None:
        result["feature_mse"] = {"raw": loss(x, reference), "rk4": loss(endpoint, reference),
                                 "exponential_euler": loss(one_step, reference)}
    return result, endpoint


def preserve_segments(x, candidate, head, readout):
    nominal = RESERVE + 1e-9
    old, new = head(x), head(candidate)
    rows = np.arange(len(x))
    predicted = old.argmax(axis=1)
    margin0 = old[rows, predicted, None] - old
    margin1 = new[rows, predicted, None] - new
    required = nominal * margin0
    fraction = np.divide(margin0 - required, margin0 - margin1, out=np.ones_like(margin0),
                         where=margin1 < required).min(axis=1)
    endpoint = x + fraction[:, None] * (candidate - x)
    accepted = segment_batches(readout, x, endpoint)
    endpoint[~accepted] = x[~accepted]
    if np.any(head(endpoint).argmax(axis=1) != predicted):
        raise ArithmeticError("A deployed chord changed the frozen prediction")
    return endpoint, {"preserving_before_fallback": int(accepted.sum()),
                      "identity_fallbacks": int((~accepted).sum()),
                      "mean_retained_fraction_before_fallback": float(fraction.mean())}


def restoration_controls(clean_fit, val_blur, val_clean, test_blur, test_clean, y, head, readout):
    """Controls for generic shrinkage and a supervised paired correction."""
    mean = clean_fit.mean(axis=0)
    change = mean - val_blur
    strength = float(np.clip(np.sum((val_clean - val_blur) * change) / np.square(change).sum(), 0, 1))
    shrinkage = test_blur + strength * (mean - test_blur)
    order = np.random.default_rng(SEED + 1).permutation(len(val_blur))
    train, held_out = order[1000:], order[:1000]
    residual = val_clean - val_blur
    ridge_grid = {}
    for alpha in (1, 10, 100, 1000):
        model = Ridge(alpha=alpha).fit(val_blur[train], residual[train])
        ridge_grid[alpha] = loss(val_blur[held_out] + model.predict(val_blur[held_out]), val_clean[held_out])
    alpha = min(ridge_grid, key=ridge_grid.get)
    ridge = Ridge(alpha=alpha).fit(val_blur, residual)
    supervised = test_blur + ridge.predict(test_blur)
    initial = head(test_blur)
    result = {"mean_shrinkage_strength_validation": strength,
              "ridge_alpha_internal_validation_mse": ridge_grid, "ridge_selected_alpha": alpha,
              "ridge_paired_fit_rows": len(val_blur)}
    for name, candidate in (("mean_shrinkage", shrinkage), ("paired_ridge", supervised)):
        safe, audit = preserve_segments(test_blur, candidate, head, readout)
        result[name] = {"unrestricted_mse": loss(candidate, test_clean), "unrestricted_metrics": metrics(head(candidate), y),
                        "unrestricted_label_effects": paired(initial, head(candidate), y),
                        "label_preserving_mse": loss(safe, test_clean), "label_preserving_metrics": metrics(head(safe), y),
                        "certified_segment_audit": audit}
    return result


def segment_batches(readout, x, y):
    return np.concatenate([readout.segment(x[i:i + 128], y[i:i + 128]) for i in range(0, len(x), 128)])


def certificates(model, readout, head, start, endpoint, horizon, limit):
    x = start
    if limit < len(x):
        indices = np.arange(limit) * len(x) // limit
        x, endpoint = x[indices], endpoint[indices]
    clock = time.perf_counter()
    guard = ReachableHullGuard(readout, model.centers, np.ones(len(model.centers)), model.beta)
    setup = time.perf_counter() - clock
    verdicts = [guard.certify(x[i:i + 64], horizon) for i in range(0, len(x), 64)]
    strict, slab, hull = (np.concatenate([verdict[index] for verdict in verdicts]) for index in range(3))
    elapsed = time.perf_counter() - clock
    if np.any(hull & (head(x).argmax(axis=1) != head(endpoint).argmax(axis=1))):
        raise ArithmeticError("Numerical endpoint conflicts with exact path certificate")
    return {"represented_queries": len(x), "strict_starts": int(strict.sum()), "slab": int(slab.sum()),
            "reachable_hull": int(hull.sum()), "unresolved": int((~hull).sum()),
            "horizon": horizon, "setup_seconds": setup, "end_to_end_seconds": elapsed,
            "selection": "Full test, or evenly spaced floor(i*N/limit) rows when capped", "limit": limit}


def read_arrays(path):
    train = np.load(path / "e_train.npy").astype(np.float64)
    test = np.load(path / "e_test.npy").astype(np.float64)
    with np.load(path / "head.npz") as head:
        weights = np.asarray(head["coef"] if "coef" in head else head["weight"], dtype=np.float64)
        bias = np.asarray(head["intercept"] if "intercept" in head else head["bias"], dtype=np.float64)
    bias = bias.ravel()
    if train.ndim != 2 or test.shape != (10000, train.shape[1]) or len(train) != 50000:
        raise ValueError(f"Original-order train and test embeddings required: {train.shape}, {test.shape}")
    if weights.shape != (100, train.shape[1]) or bias.shape != (100,):
        raise ValueError(f"Unexpected encoder head: {weights.shape}, {bias.shape}")
    if not all(np.isfinite(array).all() for array in [train, test, weights, bias]):
        raise ValueError("Nonfinite embedding or affine coefficient")
    return train, test, weights, bias


def evaluate(dataset, path, limit, skip_blur=False):
    clock = time.perf_counter()
    train, test, weights, bias = read_arrays(path)
    train_y = np.load(dataset / "y_train.npy").ravel().astype(int)
    test_y = np.load(dataset / "y_test.npy").ravel().astype(int)
    order = np.random.default_rng(SEED).permutation(len(train_y))
    fit, val = order[VAL:], order[:VAL]
    head = lambda z: scores(z, weights, bias)
    readout = AffineReadout(weights, bias, RESERVE)
    x = train[fit]
    labels = train_y[fit]
    centers = np.stack([x[labels == c].mean(axis=0) for c in range(100)])
    variance = float(np.median(np.square(x - centers[labels]).sum(axis=1)))
    if not variance > 0:
        raise ArithmeticError("Prototype distance scale must be positive")
    clean = search(centers, variance, train[val], train_y[val], head)
    nonzero = clean["best_nonzero"]
    selected = clean["selected"]
    model = get_model(centers, nonzero["beta"])
    clean["metric"] = "maximize validation top-1; prefer shorter time on ties"
    clean["selected_test"], _ = compare(test, test_y, head, model, selected["horizon"])
    clean["nonzero_test"], _ = compare(test, test_y, head, model, nonzero["horizon"])
    clean["centroid_test"] = metrics(model.scores(test), test_y)
    clean["probe_test"], probe_end = compare(test, test_y, head, model, PROBE)
    refined_end, _ = rk4(model, test, PROBE, max_step=STEP / 2)
    clean["probe_test"]["refinement_label_disagreements"] = int(
        np.count_nonzero(head(probe_end).argmax(axis=1) != head(refined_end).argmax(axis=1)))
    clean["fixed_horizon_certificates"] = certificates(model, readout, head, test, probe_end, PROBE, limit)
    result = {"method": "One equal-mass fit-only class centroid, original neural affine head", "seed": SEED,
              "fit_rows": len(fit), "validation_rows": len(val), "test_rows": len(test), "embedding_dimension": x.shape[1],
              "bandwidth_base": variance, "margin_reserve": RESERVE, "rk4_max_step": STEP, "clean": clean}
    print(f"Clean complete, selected {selected}; testing for paired shifted embeddings.", flush=True)
    val_blur_path, test_blur_path = path / "e_val_blur.npy", path / "e_test_blur.npy"
    if not skip_blur and val_blur_path.exists() and test_blur_path.exists():
        val_blur = np.load(val_blur_path).astype(np.float64)
        test_blur = np.load(test_blur_path).astype(np.float64)
        if val_blur.shape != train[val].shape or test_blur.shape != test.shape:
            raise ValueError("Blurred embeddings must be validation-order and official test-order")
        blur = search(centers, variance, val_blur, train_y[val], head, reference=train[val])
        blur["metric"] = "minimize validation MSE to the paired clean encoder embeddings"
        selected_blur, nonzero_blur = blur["selected"], blur["best_nonzero"]
        blur_model = get_model(centers, nonzero_blur["beta"])
        blur["selected_test"], _ = compare(test_blur, test_y, head, blur_model, selected_blur["horizon"], reference=test)
        blur["nonzero_test"], candidate = compare(test_blur, test_y, head, blur_model, nonzero_blur["horizon"], reference=test)
        blur["probe_test"], _ = compare(test_blur, test_y, head, blur_model, PROBE, reference=test)
        safe, audit = preserve_segments(test_blur, candidate, head, readout)
        blur["label_preserving_clipped_chord_best_nonzero"] = {**audit, "feature_mse": loss(safe, test),
                                                                "metrics": metrics(head(safe), test_y)}
        blur["restoration_controls"] = restoration_controls(x, val_blur, train[val], test_blur, test, test_y, head, readout)
        result["gaussian_blur"] = blur
    result["input_sha256"] = {
        name: digest(path / name) for name in ("e_train.npy", "e_test.npy", "head.npz", "e_val_blur.npy", "e_test_blur.npy")
        if (path / name).exists() and (not skip_blur or "blur" not in name)
    }
    result["elapsed_seconds"] = time.perf_counter() - clock
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(".research/cifar100"))
    parser.add_argument("--embeddings", type=Path, default=Path(".research/cifar100/neural"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--certificate-rows", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-blur", action="store_true")
    args = parser.parse_args()
    if args.certificate_rows < 1:
        parser.error("--certificate-rows must be positive")
    with threadpool_limits(limits=args.threads):
        result = evaluate(args.data, args.embeddings, args.certificate_rows, args.skip_blur)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {args.output} in {result['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
