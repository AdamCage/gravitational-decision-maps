"""Outward-rounded stability and switch certificates for equal-mass Voronoi readouts.

The sites, inputs, bandwidth and time are the exact real values represented
by their float64 encodings. Assume IEEE binary64 arithmetic with gradual
underflow. Nonfinite intermediate values cause failure rather than a verdict.
"""

from collections import namedtuple
from typing import NamedTuple

import numpy as np


def down(value):
    return np.nextafter(value, -np.inf)


def up(value):
    return np.nextafter(value, np.inf)


class Bounds(namedtuple("BoundsPair", "lo hi")):
    __slots__ = ()

    @classmethod
    def exact(cls, value):
        array = np.asarray(value, dtype=np.float64)
        return cls(array, array)

    def __add__(self, other):
        return Bounds(down(self.lo + other.lo), up(self.hi + other.hi))

    def __neg__(self):
        return Bounds(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        ll, lh = self.lo * other.lo, self.lo * other.hi
        hl, hh = self.hi * other.lo, self.hi * other.hi
        lower = np.minimum(np.minimum(ll, lh), np.minimum(hl, hh))
        upper = np.maximum(np.maximum(ll, lh), np.maximum(hl, hh))
        return Bounds(down(lower), up(upper))

    def __truediv__(self, other):
        if np.any(other.lo <= 0):
            raise ArithmeticError("The denominator must have a positive lower bound")
        return self * Bounds(down(1 / other.hi), up(1 / other.lo))

    def select(self, index):
        return Bounds(self.lo[index], self.hi[index])

    @property
    def T(self):
        return Bounds(self.lo.T, self.hi.T)


ZERO = Bounds.exact(0)
ONE = Bounds.exact(1)
HALF = Bounds.exact(0.5)


def negative_exp(bounds):
    """Enclose exp for nonpositive arguments without assuming libm accuracy."""
    if np.any(bounds.hi > 0) or not np.all(np.isfinite(bounds.lo)):
        raise ArithmeticError("Expected finite nonpositive exponential arguments")
    values = np.stack(np.broadcast_arrays(bounds.lo, bounds.hi))
    # Divide by 512, bound the degree-12 alternating Taylor polynomial,
    # then square nine times. For |z| <= 1/8 the next term is < 2**-70.
    scaled = np.maximum(values, -64.0) / 512
    z = Bounds(down(scaled), np.minimum(up(scaled), 0))
    result = ONE
    for order in range(12, 0, -1):
        result = ONE + z * (result / Bounds.exact(order))
    lo = down(result.lo - 2.0**-70)
    hi = result.hi
    for _ in range(9):
        lo = down(lo * lo)
        hi = up(hi * hi)
    return Bounds(np.where(values[0] < -64, 0, lo[0]), np.minimum(hi[1], 1))


def dot_rows(left, right):
    """Enclose products of every pair of float64 rows without BLAS assumptions."""
    result = ZERO
    for x, y in zip(left.T, right.T, strict=True):
        product = x[:, None] * y
        result += Bounds(down(product), up(product))
    return result


def sum_last(terms):
    result = ZERO
    for index in range(terms.lo.shape[-1]):
        result += terms.select((..., index))
    return result


def exponential_weights(projected_logits):
    """Bound the weights after subtracting a shared, exactly encoded shift."""
    shift = projected_logits.hi.max(axis=-1, keepdims=True)
    shifted = projected_logits - Bounds.exact(shift)
    return negative_exp(Bounds(shifted.lo, np.minimum(shifted.hi, 0)))


class Verdicts(NamedTuple):
    initial: np.ndarray
    stable: np.ndarray
    switched: np.ndarray
    slab_stable: np.ndarray


class VoronoiCertificate:
    """Certify the full path or a different endpoint label without integrating.

    Uses every rival inequality. Choose modest batch sizes because the
    query arrays scale quadratically in the number of sites.
    Raises ArithmeticError if site separation or time bounds are too
    narrow to validate a positive denominator in this precision.
    """

    @np.errstate(over="raise", invalid="raise", divide="raise", under="ignore")
    def __init__(self, sites, beta):
        self.sites = np.array(sites, dtype=np.float64, copy=True)
        if (self.sites.ndim != 2 or min(self.sites.shape) < 1 or len(self.sites) < 2
                or not np.all(np.isfinite(self.sites)) or not np.isfinite(beta) or beta <= 0):
            raise ValueError("Provide finite distinct row sites and positive bandwidth")
        self.beta = Bounds.exact(beta)
        gram = dot_rows(self.sites, self.sites)
        indices = np.arange(len(self.sites))
        norm_squared = gram.select((indices, indices))
        self.bias = norm_squared * HALF
        site_scores = gram - self.bias
        # Axes are candidate leader, rival, evaluation site.
        leader = site_scores.T.select((slice(None), None))
        rival = site_scores.T.select((None, slice(None)))
        offsets = leader - rival
        pair_average = (leader + rival) * HALF
        distance = norm_squared.select((slice(None), None)) + norm_squared
        squared_distance = distance - gram * Bounds.exact(2)
        diag = np.eye(len(self.sites), dtype=bool)
        self.distance = Bounds(np.where(diag, 1, squared_distance.lo), np.where(diag, 1, squared_distance.hi))
        self.offsets = offsets
        d = self.distance.select((..., None))
        projected = (site_scores - pair_average.select((..., None))
                     - offsets.select((..., None)) * (offsets / d).select((..., None, slice(None)))) / self.beta
        k_sites = len(self.sites)
        i, j, k = np.ogrid[:k_sites, :k_sites, :k_sites]
        self.tied = (k == i) | (k == j)
        projected = Bounds(np.where(self.tied[..., None, :], 0, projected.lo),
                           np.where(self.tied[..., None, :], 0, projected.hi))
        self.projected_lo = Bounds(projected.lo.min(-2), projected.hi.min(-2))
        self.projected_hi = Bounds(projected.lo.max(-2), projected.hi.max(-2))
        span = Bounds.exact(offsets.hi.max(-1)) - Bounds.exact(offsets.lo.min(-1))
        rate = (span * span) / (self.distance * self.beta * Bounds.exact(4)) - ONE
        self.upper_rate = Bounds.exact(np.maximum(rate.hi, 0))
        self.drift = Bounds(np.where(self.tied, 0, offsets.lo), np.where(self.tied, 0, offsets.hi))

    @np.errstate(over="raise", invalid="raise", divide="raise", under="ignore")
    def classify(self, queries, horizon):
        """Return strict initial sites (or -1), local verdicts and global-slab stability."""
        queries = np.asarray(queries, dtype=np.float64)
        if (queries.ndim != 2 or queries.shape[1] != self.sites.shape[1]
                or not np.all(np.isfinite(queries)) or not np.isfinite(horizon) or horizon <= 0):
            raise ValueError("Provide finite compatible row queries and a positive horizon")
        time = Bounds.exact(horizon)
        decay = negative_exp(-time)
        move = ONE - decay
        scores = dot_rows(queries, self.sites) - self.bias
        rows = np.arange(len(queries))
        rivals = np.arange(len(self.sites))[None, :]
        winner = scores.lo.argmax(axis=-1)
        allowed = winner[:, None] != rivals
        leader = scores.select((rows, winner, None))
        margins = leader - scores
        verified = np.all((margins.lo > 0) | ~allowed, axis=-1)
        pair_average = (leader + scores) * HALF
        key = (winner, slice(None))
        distance = self.distance.select(key)
        offsets = self.offsets.select(key)
        projection = (scores.select((slice(None), None)) - pair_average.select((..., None))
                      - (margins / distance).select((..., None)) * offsets) / self.beta
        tied = self.tied[key]
        projection = Bounds(np.where(tied, 0, projection.lo), np.where(tied, 0, projection.hi))
        lower_delta = self.projected_lo.select(key) - projection
        upper_delta = self.projected_hi.select(key) - projection
        step = move * Bounds(np.minimum(lower_delta.lo, 0), np.maximum(upper_delta.hi, 0))
        weights = exponential_weights(projection + step)
        drift = self.drift.select(key)

        stable_threshold = margins * (decay / move)
        stable_sign = sum_last((drift + stable_threshold.select((..., None))) * weights)
        stable = verified & np.all((stable_sign.lo > 0) | ~allowed, axis=-1)
        slab_loss = np.maximum(0, -drift.lo.min(axis=-1))
        slab_stable = verified & np.all((stable_threshold.lo > slab_loss) | ~allowed, axis=-1)

        # Jensen bounds integral_0^T exp(-rate*t) dt from below.
        rate_time = self.upper_rate.select(key) * time * HALF
        exponent = Bounds(-rate_time.hi, np.minimum(-rate_time.lo, 0))
        crossing_time = time * negative_exp(exponent)
        flip_threshold = margins / crossing_time
        flip_sign = sum_last((drift + flip_threshold.select((..., None))) * weights)
        switched = verified & np.any((flip_sign.hi < 0) & allowed, axis=-1)
        if np.any((stable | slab_stable) & switched):
            raise ArithmeticError("Incompatible stability and switch certificates")
        return Verdicts(np.where(verified, winner, -1), stable, switched, slab_stable)
