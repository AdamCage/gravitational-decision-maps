"""Pinned public transfer cases for the attraction-crossing benchmark."""

from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import gzip

import numpy as np

from .data import Case


PUBLIC_FAMILIES = ("pendigits", "seeds")
_DATA = Path(__file__).resolve().parent.parent / "crossing_benchmark_data"
_DIGESTS = {
    "pendigits.tra": "e2b9eb9f0d0467e2b64a4816a3420edf2b8043447576f4b84337aba44a9f97d3",
    "pendigits.tes": "8bd03229c5c5291fefe43e45465dd948d2645bf23328b9d993e0b777666b2015",
    "seeds_dataset.txt": "1f3f83c0d8485ae9148061389d19628607e3f5660e3d6f40ec9102fb398bb12f",
}
_DOMAIN = 408233


@lru_cache(maxsize=3)
def _load(name):
    raw = gzip.decompress((_DATA / f"{name}.gz").read_bytes())
    if sha256(raw).hexdigest() != _DIGESTS[name]:
        raise ValueError(f"Public source digest mismatch for {name}")
    content = np.loadtxt(BytesIO(raw), delimiter="," if name.startswith("pendigits") else None, dtype=float)
    x = content[:, :-1]
    y = content[:, -1].astype(np.int64)
    if name == "seeds_dataset.txt":
        y -= 1
    ids = np.sort(np.unique(x, axis=0, return_index=True)[1])
    return x, y, ids


def _partition(labels, row_ids, fraction, rng):
    fit, evaluate = [], []
    for label in np.unique(labels[row_ids]):
        members = rng.permutation(row_ids[labels[row_ids] == label])
        cut = int(fraction * len(members))
        fit.extend(members[:cut])
        evaluate.extend(members[cut:])
    return np.array(fit, dtype=np.int64), np.array(evaluate, dtype=np.int64)


def _public_fold(labels, row_ids, seed_roster, seed):
    rng = np.random.default_rng(92093)
    fold = seed_roster.index(seed)
    return np.concatenate([
        np.array_split(rng.permutation(row_ids[labels[row_ids] == label]), len(seed_roster))[fold]
        for label in np.unique(labels[row_ids])
    ])


def _overlap_filter(x_fit, fit_pool, x_test, test_pool):
    training_features = {row.tobytes() for row in x_fit[fit_pool]}
    keep = [index for index in test_pool if x_test[index].tobytes() not in training_features]
    return np.asarray(keep, dtype=np.int64)


def make_public(family: str, seed: int, *, split: str = "development", fold_seeds=()) -> Case:
    """Return a class-balanced external probe; supply final PenDigits seeds for disjoint folds."""
    if family not in PUBLIC_FAMILIES:
        raise ValueError(f"Unknown public transfer family {family!r}")
    if split not in {"development", "final"}:
        raise ValueError(f"Unknown split {split!r}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    seed = int(seed)
    fold_seeds = tuple(fold_seeds)
    if fold_seeds and (seed not in fold_seeds or len(fold_seeds) != len(set(fold_seeds))):
        raise ValueError("The optional fold roster must contain distinct seeds including this one")
    stream = [_DOMAIN, seed, PUBLIC_FAMILIES.index(family), int(split == "final")]
    rng_fit = np.random.default_rng(np.random.SeedSequence([*stream, 0]))
    rng_test = np.random.default_rng(np.random.SeedSequence([*stream, 1]))
    if family == "pendigits":
        train_source = "pendigits.tra"
        test_source = "pendigits.tes" if split == "final" else train_source
        x_fit, y_fit, fit_pool = _load(train_source)
        if split == "development":
            fit_pool, test_pool = _partition(y_fit, fit_pool, 0.72, np.random.default_rng(32011))
            x_test, y_test = x_fit, y_fit
            groups = "row-disjoint development subsets within the original 30-writer train source"
            removed_cross = 0
        else:
            x_test, y_test, test_pool = _load(test_source)
            unique_test = _overlap_filter(x_fit, fit_pool, x_test, test_pool)
            removed_cross = len(test_pool) - len(unique_test)
            test_pool = _public_fold(y_test, unique_test, fold_seeds, seed) if fold_seeds else unique_test
            groups = "original 30-writer training source and disjoint 14-writer evaluation source"
        cap, quota, test_cap = 80, 28, 36
        transform = lambda x: x
        coordinates = "author-normalized 16 pen coordinates as published"
    else:
        train_source = test_source = "seeds_dataset.txt"
        x_fit, y_fit, all_ids = _load(train_source)
        x_test, y_test = x_fit, y_fit
        fit_pool, test_pool = _partition(y_fit, all_ids, 0.58, np.random.default_rng(32012))
        if split == "development":
            fit_pool, test_pool = _partition(y_fit, fit_pool, 0.67, np.random.default_rng(32013))
        groups = "fixed class-stratified row split; field, batch and individual source IDs unavailable"
        cap, quota, test_cap = 25, 10, 20
        transform = np.log
        coordinates = "natural log of the seven measured features, used only by curation"
        removed_cross = 0

    admitted, eligible, anchors, radii = [], [], {}, {}
    classes = np.unique(y_fit[fit_pool])
    for label in classes:
        candidates = rng_fit.permutation(fit_pool[y_fit[fit_pool] == label])[:cap]
        anchor = int(candidates[0])
        center = transform(x_fit[anchor])
        d_train = np.linalg.norm(transform(x_fit[candidates]) - center, axis=1)
        radius = float(np.sort(d_train)[quota - 1])
        admitted.extend(candidates[d_train <= radius])
        candidates_test = test_pool[y_test[test_pool] == label]
        d_test = np.linalg.norm(transform(x_test[candidates_test]) - center, axis=1)
        eligible.append(candidates_test[d_test > radius])
        anchors[str(int(label))] = f"{train_source}:{anchor}"
        radii[str(int(label))] = radius
    test_per_class = min(test_cap, *map(len, eligible))
    if test_per_class < 1:
        raise ValueError(f"No external evaluation support for at least one class in {family}/{seed}/{split}")
    selected = np.concatenate([rng_test.permutation(members)[:test_per_class] for members in eligible])
    train_idx = rng_fit.permutation(np.asarray(admitted, dtype=np.int64))
    test_idx = rng_test.permutation(selected)
    train_array, train_labels = x_fit[train_idx], y_fit[train_idx]
    test_array, test_labels = x_test[test_idx], y_test[test_idx]
    for array in (train_array, train_labels, test_array, test_labels):
        array.flags.writeable = False
    train_ids = [f"{train_source}:{index}" for index in train_idx]
    test_ids = [f"{test_source}:{index}" for index in test_idx]
    if not set(train_ids).isdisjoint(test_ids):
        raise ValueError("Training and evaluation source IDs overlap")
    metadata = {
        "family": family, "seed": seed, "split": split, "gap": "source_defined",
        "feature_dim": train_array.shape[1], "n_classes": len(classes),
        "n_train_per_class": {str(int(c)): int(np.sum(train_labels == c)) for c in classes},
        "n_test_per_class": test_per_class, "eligible_test_by_class": {str(int(c)): len(eligible[i]) for i, c in enumerate(classes)},
        "training_candidate_cap_per_class": cap, "training_core_quota_per_class": quota,
        "source_sha256": {name: _DIGESTS[name] for name in (train_source, test_source)},
        "source_grouping": groups, "curation_coordinates": coordinates,
        "core_rule": "distance at most the admitted same-class anchor radius; ties included",
        "core_radii": radii, "anchor_ids": anchors,
        "cross_source_exact_feature_duplicates_removed": removed_cross,
        "fold": fold_seeds.index(seed) if fold_seeds and family == "pendigits" and split == "final" else None,
        "train_ids": train_ids, "test_ids": test_ids, "encoder": "identity before training-only scaling",
        "curator_only": "candidate ranks, source identifiers, test class labels and metadata",
    }
    return Case(train_array, train_labels, test_array, test_labels, metadata)
