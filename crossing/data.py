"""Paired, core-only classification data for attraction trajectories.

Gap changes move the centers of two observed bands in a known generating
coordinate. The evaluation support, labels, sample sizes, random geometry and
noise draws are held fixed across gaps. Training code should receive only the
observation arrays and core labels.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FAMILIES = ("affine_shells", "wavy_shells", "open_chains", "local_lobes")
GAPS = ("small", "medium", "large")

_START = {"small": 0.48, "medium": 0.80, "large": 1.12}
_BAND_WIDTH = 0.95
_TEST_HALF_WIDTH = 0.34
_CLASSES = 3
_TRAIN_PER_CLASS = 64
_TEST_PER_CLASS = 120
_DOMAIN = 208104


@dataclass(frozen=True)
class Case:
    x_train: NDArray[np.float64]
    y_train: NDArray[np.int64]
    x_test: NDArray[np.float64]
    y_test: NDArray[np.int64]
    metadata: dict


@dataclass(frozen=True)
class _Geometry:
    matrix: NDArray[np.float64]
    shift: NDArray[np.float64]
    coefficients: NDArray[np.float64]


def _generator(seed, family, source):
    return np.random.default_rng(np.random.SeedSequence([_DOMAIN, seed, FAMILIES.index(family), source]))


def _rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def _geometry(family, seed):
    rng = _generator(seed, family, 0)
    angle = rng.uniform(-np.pi, np.pi)
    stretch = rng.uniform(1.18, 1.50)
    squeeze = rng.uniform(0.69, 0.94)
    skew = rng.uniform(-0.23, 0.23)
    matrix = _rotation(angle) @ np.array([[stretch, skew], [0.0, squeeze]])
    return _Geometry(matrix, rng.uniform(-0.40, 0.40, 2), rng.uniform(-1, 1, 8))


def _features(family, geometry, positions, labels, rng):
    n = len(labels)
    phase, curve, harmonic = geometry.coefficients[:3]
    if family in {"affine_shells", "wavy_shells"}:
        radii = np.array([0.83, 1.45, 2.16])[labels]
        if family == "wavy_shells":
            radii *= 1.0 + (0.18 + 0.02 * curve) * np.cos(3 * positions + np.pi * phase)
            radii += 0.10 * np.sin(5 * positions - np.pi * harmonic)
            radii += 0.05 * (labels - 1) * np.sin(2 * positions + np.pi * harmonic)
        direction = np.column_stack((np.cos(positions), np.sin(positions)))
        radii += rng.normal(0.0, 0.045, n)
        xy = direction * radii[:, None]
    elif family == "open_chains":
        progress = 1.15 * positions
        wave = 0.42 * np.cos(1.40 * progress + 0.72 * phase) + 0.18 * curve * progress
        offset = (labels - 1) * (0.67 + 0.10 * np.cos(progress + np.pi * harmonic))
        xy = np.column_stack((progress, wave + offset))
    else:
        centers = np.array([[-1.52, -0.70], [1.52, -0.70], [0.02, 1.60]])
        headings = np.array([0.35, -0.28, 1.48]) + 0.31 * geometry.coefficients[3:6]
        direction = np.column_stack((np.cos(headings[labels]), np.sin(headings[labels])))
        xy = centers[labels] + (0.35 * positions[:, None]) * direction
        xy += rng.normal(0.0, 0.10, xy.shape)
    xy += rng.normal(0.0, 0.047, xy.shape)
    xy = xy @ geometry.matrix.T + geometry.shift
    if family == "open_chains":
        xy = np.column_stack((xy, rng.normal(0.0, 0.21, (n, 4))))
    return xy


def _draw(family, gap, seed, geometry, test):
    rng = _generator(seed, family, 2 if test else 1)
    labels = np.repeat(np.arange(_CLASSES, dtype=np.int64), _TEST_PER_CLASS if test else _TRAIN_PER_CLASS)
    n = len(labels)
    if test:
        positions = rng.uniform(-_TEST_HALF_WIDTH, _TEST_HALF_WIDTH, n)
        part = "test"
    else:
        signs = np.tile(np.repeat([-1.0, 1.0], _TRAIN_PER_CLASS // 2), _CLASSES)
        positions = signs * (_START[gap] + _BAND_WIDTH * rng.random(n))
        part = "train"
    features = _features(family, geometry, positions, labels, rng)
    order = rng.permutation(n)
    x, y = features[order], labels[order]
    x.flags.writeable = False
    y.flags.writeable = False
    # Across gaps a train ID denotes the same latent draw after moving the band.
    # A test ID denotes the exact same measured feature row and label.
    ids = [f"{family}:{seed}:{part}:{index}" for index in order]
    return x, y, ids, (float(np.min(np.abs(positions))), float(np.max(np.abs(positions))))


def make_case(family: str, gap: str, seed: int) -> Case:
    """Generate a case with independent train/test draws and a held-out band."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown benchmark family {family!r}")
    if gap not in GAPS:
        raise ValueError(f"Unknown gap {gap!r}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    seed = int(seed)
    geometry = _geometry(family, seed)
    x_train, y_train, train_ids, train_abs = _draw(family, gap, seed, geometry, test=False)
    x_test, y_test, test_ids, test_abs = _draw(family, gap, seed, geometry, test=True)
    metadata = {
        "family": family, "gap": gap, "seed": seed,
        "coordinate": "angle_radians" if family.endswith("shells") else "longitudinal_parameter",
        "core_absolute_bounds": [_START[gap], _START[gap] + _BAND_WIDTH],
        "test_absolute_bounds": [0.0, _TEST_HALF_WIDTH],
        "realized_train_absolute_bounds": list(train_abs), "realized_test_absolute_bounds": list(test_abs),
        "nominal_gap_to_test_support": _START[gap] - _TEST_HALF_WIDTH,
        "feature_dim": x_train.shape[1], "nuisance_dim": x_train.shape[1] - 2,
        "n_classes": _CLASSES, "n_train_per_class": _TRAIN_PER_CLASS, "n_test_per_class": _TEST_PER_CLASS,
        "train_ids": train_ids, "test_ids": test_ids,
        "encoder": "identity before training-only scaling",
        "target_rule": "generator class along an affine shell, wavy shell, open chain, or local lobe",
        "curator_only": "latent sampling coordinate, geometry parameters, held-out classes, and metadata",
    }
    return Case(x_train, y_train, x_test, y_test, metadata)
