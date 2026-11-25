"""Benchmarks curated outside the learner for inductive known-class extrapolation."""

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
from io import BytesIO
from pathlib import Path
import gzip

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_digits, load_wine


FINAL_SEEDS = (101, 113, 137, 151, 181)
DEVELOPMENT_SEEDS = (17, 31)
SYNTHETIC_TASKS = (
    "rings_wedge", "curved_wedge", "anisotropic_wide", "anisotropic_narrow",
    "disconnected", "paired_a", "paired_b",
)
REAL_TASKS = ("digits_narrow", "digits_wide", "wine_narrow", "wine_wide")

_DIGITS_TRAIN_SHA = "e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd"
_DIGITS_TEST_SHA = "09f66e6debdee2cd2b5ae59e0d6abbb73fc2b0e0185d2e1957e9ebb51e23aa22"
_WINE_SHA = "a5f9961298dc86deafae12b9ee88a283508cefa03cd1f719e9dfe581ec898777"


@dataclass(frozen=True)
class Dataset:
    x_train: NDArray[np.float64]
    y_train: NDArray[np.int64]
    x_core: NDArray[np.float64]
    y_core: NDArray[np.int64]
    x_external: NDArray[np.float64]
    y_external: NDArray[np.int64]
    metadata: dict


def _distribution(labels, classes):
    return {str(int(cls)): int(np.sum(labels == cls)) for cls in classes}


def _assemble(train, core, external, metadata):
    classes = np.unique(train[1])
    for label, section in (("train", train), ("core", core), ("external", external)):
        x, y, ids = section
        if len(ids) != len(x) or len(x) != len(y) or len(set(ids)) != len(ids):
            raise ValueError(f"Malformed or duplicate records in {label} split")
        if x.ndim != 2 or x.shape[1] != train[0].shape[1] or not np.isfinite(x).all():
            raise ValueError(f"Malformed features in {label} split")
        if len(y) and not set(y).issubset(classes):
            raise ValueError(f"An evaluation class is absent from training in {label}")
        metadata[f"{label}_ids"] = ids
        metadata[f"n_{label}_by_class"] = _distribution(y, classes)
        x.flags.writeable = False
        y.flags.writeable = False
    if len(np.unique(external[1])) != len(classes):
        raise ValueError("External balanced accuracy requires all training classes")
    if len(set(train[2]) | set(core[2]) | set(external[2])) != sum(map(len, (train[2], core[2], external[2]))):
        raise ValueError("Training and evaluation IDs overlap")
    return Dataset(*train[:2], *core[:2], *external[:2], metadata)


def _rng(seed, split, family):
    return np.random.default_rng(np.random.SeedSequence([seed, int(split == "final"), family]))


def _synthetic(name, seed, split):
    family = {
        "rings_wedge": "rings", "curved_wedge": "curves",
        "anisotropic_wide": "anisotropic", "anisotropic_narrow": "anisotropic",
        "disconnected": "disconnected", "paired_a": "paired", "paired_b": "paired",
    }[name]
    family_id = ("rings", "curves", "anisotropic", "disconnected", "paired").index(family)
    rng = _rng(seed, split, family_id)
    specifications = {
        "rings": ((100, 100, 100), 2, 0.09, "abs(theta) <= 0.34*pi", 0.34),
        "curves": ((120, 75), 2, 0.07, "0.27 <= path parameter <= 0.73", 0.46),
        "disconnected": ((104, 72, 50), 2, 0.075, "latent components 0 and 1 of 3", 2 / 3),
        "paired": ((100, 100), 2, 0.05, "abs(longitudinal parameter) <= 0.9", 0.36),
    }
    radius = 0.55 if name == "anisotropic_narrow" else 0.90
    if family == "anisotropic":
        counts, dim, noise = (130, 87, 57), 8, 0.11
        rule, coverage = f"latent component radius <= {radius}", (radius / 2.2) ** 2
    else:
        counts, dim, noise, rule, coverage = specifications[family]
    classes = len(counts)

    def points(cls, n, is_external):
        sign = 2 * cls - 1
        if family == "rings":
            limit = 0.34 * np.pi
            if is_external:
                angle = rng.uniform(limit, np.pi, n) * rng.choice([-1, 1], n)
            else:
                angle = rng.uniform(-limit, limit, n)
            xy = (0.9 + 0.82 * cls + rng.normal(0.0, noise, n))[:, None]
            return xy * np.column_stack((np.cos(angle), np.sin(angle)))
        if family == "curves":
            if is_external:
                t = rng.uniform(0, 0.27, n)
                t = np.where(rng.random(n) < 0.5, t, 1 - t)
            else:
                t = rng.uniform(0.27, 0.73, n)
            xy = np.column_stack((-sign * np.cos(np.pi * t) + cls, -sign * np.sin(np.pi * t) + 0.45 * cls))
        elif family == "anisotropic":
            theta = rng.uniform(-np.pi, np.pi, n)
            low, high = (radius, 2.2) if is_external else (0, radius)
            rho = np.sqrt(rng.uniform(low ** 2, high ** 2, n))
            p = rho[:, None] * np.column_stack((1.15 * np.cos(theta), 0.32 * np.sin(theta)))
            angles = (0.42, -0.65, 0.86)
            a, b = np.cos(angles[cls]), np.sin(angles[cls])
            offset = np.asarray([[-1.5, -0.45], [0.9, -0.18], [-0.08, 1.66]])[cls]
            component = rng.choice([-1, 1], n)[:, None] * np.array([0.17, 0.30])
            xy = p @ np.array([[a, b], [-b, a]]) + offset + component
        elif family == "disconnected":
            centers = np.array([
                [[-2.1, -1.5], [-2.1, 0.05], [0.85, 0.9]],
                [[0.1, -1.6], [0.25, 0.00], [-0.95, 1.55]],
                [[2.0, -1.45], [2.0, 0.30], [1.2, 1.95]],
            ])
            component = np.full(n, 2) if is_external else rng.integers(0, 2, n)
            xy = centers[cls, component] + rng.normal(0, 0.19, (n, 2))
        else:
            if is_external:
                t = rng.uniform(0.9, 2.5, n) * rng.choice([-1, 1], n)
            else:
                t = rng.uniform(-0.9, 0.9, n)
            xy = np.column_stack((sign * (1.08 + 0.22 * t), t))
        xy += rng.normal(0, noise, xy.shape)
        if dim > 2:
            xy = np.column_stack((xy, rng.normal(0, 0.24, (n, dim - 2))))
        return xy

    def section(kind, sizes):
        x = np.concatenate([points(cls, n, kind == "external") for cls, n in enumerate(sizes)])
        y = np.repeat(np.arange(classes, dtype=np.int64), sizes)
        order = rng.permutation(len(y))
        ids = [f"{split}:{family}:{seed}:{kind}:{index}" for index in order]
        if name == "paired_b" and kind == "external":
            y = 1 - y
        return x[order], y[order], ids

    metadata = {
        "name": name, "seed": seed, "split": split, "family": family,
        "core_rule": rule, "fractional_latent_coverage": coverage,
        "feature_dim": dim, "coordinate_noise_sd": noise,
        "encoder": "identity without pretraining",
        "target_rule": "labels from the generator's latent classes, independent of the fitted field",
        "labels_complemented_outside_cores": name == "paired_b",
        "curator_only": "latent variables and generator constants",
    }
    train = section("train", counts)
    core = section("core", (45,) * classes)
    external = section("external", (160,) * classes)
    return _assemble(train, core, external, metadata)


def _verified(bytes_value, expected, label):
    if sha256(bytes_value).hexdigest() != expected:
        raise ValueError(f"Snapshot digest mismatch for {label}")
    return bytes_value


@lru_cache(maxsize=1)
def _load_digits_train():
    compressed = Path(__file__).resolve().parent.parent / "benchmark_data" / "optdigits.tra.gz"
    raw = _verified(gzip.decompress(compressed.read_bytes()), _DIGITS_TRAIN_SHA, "UCI Digits train")
    data = np.loadtxt(BytesIO(raw), delimiter=",", dtype=np.int64)
    return data[:, :-1].astype(np.float64), data[:, -1]


@lru_cache(maxsize=1)
def _load_packaged_data(family):
    if family == "digits":
        name, digest, load = "digits.csv.gz", _DIGITS_TEST_SHA, load_digits
    else:
        name, digest, load = "wine_data.csv", _WINE_SHA, load_wine
    _verified(files("sklearn.datasets.data").joinpath(name).read_bytes(), digest, name)
    x, y = load(return_X_y=True)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.int64)


def _stratified_partition(indices, labels, fraction, rng):
    left, right = [], []
    for cls in np.unique(labels[indices]):
        members = rng.permutation(indices[labels[indices] == cls])
        boundary = int(fraction * len(members))
        left.extend(members[:boundary])
        right.extend(members[boundary:])
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def _digits_eval_ids(y, seed, split):
    if split != "final" or seed not in FINAL_SEEDS:
        return np.arange(len(y))
    rng = np.random.default_rng(93563)
    fold = FINAL_SEEDS.index(seed)
    return np.concatenate([
        np.array_split(rng.permutation(np.flatnonzero(y == cls)), len(FINAL_SEEDS))[fold]
        for cls in np.unique(y)
    ])


@lru_cache(maxsize=1)
def _audit_digits_separation():
    x_train, _ = _load_digits_train()
    x_test, _ = _load_packaged_data("digits")
    if len(np.unique(np.concatenate((x_train, x_test)), axis=0)) != len(x_train) + len(x_test):
        raise ValueError("Digits snapshot has exact duplicates within or across writer cohorts")


def _public(name, seed, split):
    family, coverage_level = name.split("_")
    fraction = 0.27 if coverage_level == "narrow" else 0.50
    rng = _rng(seed, split, 20 if family == "digits" else 21)
    if family == "digits":
        x_train_source, y_train_source = _load_digits_train()
        source_files = {"optdigits.tra": _DIGITS_TRAIN_SHA}
        train_namespace = "digits:uci-train"
        indices = np.arange(len(x_train_source))
        if split == "final":
            _audit_digits_separation()
            x_eval_source, y_eval_source = _load_packaged_data("digits")
            source_files["sklearn/digits.csv.gz"] = _DIGITS_TEST_SHA
            train_pool, eval_pool = indices, _digits_eval_ids(y_eval_source, seed, split)
            eval_namespace = "digits:uci-test"
            grouping = "original UCI 30-writer training source vs separate 13-writer test source"
        else:
            train_pool, eval_pool = _stratified_partition(indices, y_train_source, 0.72, rng)
            x_eval_source, y_eval_source = x_train_source, y_train_source
            eval_namespace = train_namespace
            grouping = "development-only disjoint rows within the 30-writer training source"
        curation_transform = lambda x: x
        candidate_cap = 88
    else:
        x_train_source, y_train_source = _load_packaged_data("wine")
        if len(np.unique(x_train_source, axis=0)) != len(x_train_source):
            raise ValueError("Wine snapshot has exact duplicate feature rows")
        x_eval_source, y_eval_source = x_train_source, y_train_source
        rows = np.arange(len(x_train_source))
        train_pool, eval_pool = _stratified_partition(rows, y_train_source, 0.60, np.random.default_rng(52420))
        if split == "development":
            train_pool, eval_pool = _stratified_partition(train_pool, y_train_source, 0.65, rng)
        train_namespace = eval_namespace = "wine:uci"
        source_files = {"sklearn/wine_data.csv": _WINE_SHA}
        grouping = "fixed stratified 60/40 split by original unique row; producer/vintage IDs unavailable"
        curation_transform = np.log
        candidate_cap = 24

    train_idx, core_idx, external_idx = [], [], []
    anchor_ids, radii = {}, {}
    for cls in np.unique(y_train_source[train_pool]):
        choices = train_pool[y_train_source[train_pool] == cls]
        candidates = rng.permutation(choices)[:candidate_cap]
        anchor = int(candidates[0])
        anchor_location = curation_transform(x_train_source[anchor])
        candidate_dist = np.linalg.norm(curation_transform(x_train_source[candidates]) - anchor_location, axis=1)
        quota = max(4, round(fraction * len(candidates)))
        radius = float(np.sort(candidate_dist)[quota - 1])
        admitted = candidates[candidate_dist <= radius]
        train_idx.extend(admitted)
        relevant = eval_pool[y_eval_source[eval_pool] == cls]
        d = np.linalg.norm(curation_transform(x_eval_source[relevant]) - anchor_location, axis=1)
        core_idx.extend(relevant[d <= radius])
        external_idx.extend(relevant[d > radius])
        anchor_ids[str(int(cls))] = f"{train_namespace}:{anchor}"
        radii[str(int(cls))] = radius

    def section(indices, x, y, namespace):
        indices = rng.permutation(np.asarray(indices, dtype=np.int64))
        return x[indices], y[indices], [f"{namespace}:{idx}" for idx in indices]

    metadata = {
        "name": name, "seed": seed, "split": split, "family": family, "source_sha256": source_files,
        "source_grouping": grouping, "retained_candidate_fraction": fraction,
        "core_rule": "euclidean distance to a seeded admitted same-class anchor at most the class radius",
        "curation_coordinates": "raw pixel counts" if family == "digits" else "natural log of raw constituent values",
        "candidate_cap_per_class": candidate_cap,
        "encoder": "identity on source columns without pretraining",
        "curator_only": "anchor choice, candidate ranking and qualification against the reference class label",
        "anchor_ids": anchor_ids, "core_radii": radii, "feature_dim": x_train_source.shape[1],
        "exact_feature_duplicates": 0,
        "evaluation_fold": FINAL_SEEDS.index(seed) if family == "digits" and split == "final" and seed in FINAL_SEEDS else None,
    }
    train = section(train_idx, x_train_source, y_train_source, train_namespace)
    core = section(core_idx, x_eval_source, y_eval_source, eval_namespace)
    external = section(external_idx, x_eval_source, y_eval_source, eval_namespace)
    return _assemble(train, core, external, metadata)


def make_task(name: str, seed: int, *, split: str = "final") -> Dataset:
    """Return curator data. Pass only the two training arrays to learning routines."""
    if split not in {"development", "final"}:
        raise ValueError(f"Unknown split {split!r}")
    if name in SYNTHETIC_TASKS:
        return _synthetic(name, seed, split)
    if name in REAL_TASKS:
        return _public(name, seed, split)
    raise ValueError(f"Unknown benchmark {name!r}")
