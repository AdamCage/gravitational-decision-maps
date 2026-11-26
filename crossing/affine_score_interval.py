"""Batched binary64 interval integration in exact prototype-score coordinates.

The encoded sites, queries, bandwidth and horizon denote exact real values.
Time steps are dyadic subdivisions of the encoded horizon. All enclosures
use the outward kernels used by the facet certificate.
"""

from typing import NamedTuple

import numpy as np

from crossing.affine_certificate import (
    Bounds, HALF, ONE, dot_rows, down, exponential_weights, negative_exp, sum_last, up,
)


class ScoreVerdicts(NamedTuple):
    initial: np.ndarray
    stable: np.ndarray
    switched: np.ndarray
    row_steps: int


def tighten(first, second):
    lo = np.maximum(first.lo, second.lo)
    hi = np.minimum(first.hi, second.hi)
    if np.any(lo > hi):
        raise ArithmeticError("Disjoint validated enclosures")
    return Bounds(lo, hi)


def blend(start, target, fraction):
    """Enclose (1-u)start + u*target with the same u in both terms."""
    lowest = Bounds.exact(start.lo) + fraction * (Bounds.exact(target.lo) - Bounds.exact(start.lo))
    highest = Bounds.exact(start.hi) + fraction * (Bounds.exact(target.hi) - Bounds.exact(start.hi))
    return Bounds(lowest.lo, highest.hi)


def strict_initial(box):
    rows = np.arange(len(box.lo))
    winner = box.lo.argmax(axis=-1)
    others = np.arange(box.lo.shape[-1]) != winner[:, None]
    known = np.all((box.lo[rows, winner, None] > box.hi) | ~others, axis=-1)
    return np.where(known, winner, -1)


def occupied(box, winner):
    rows = np.arange(len(winner))
    others = np.arange(box.lo.shape[-1]) != winner[:, None]
    return np.all((box.lo[rows, winner, None] > box.hi) | ~others, axis=-1)


def posterior(logits):
    """Bound each p_i using monotone softmax over the enclosed weights."""
    weights = exponential_weights(logits)
    lower_total = sum_last(Bounds.exact(weights.lo)).lo[:, None]
    upper_total = sum_last(Bounds.exact(weights.hi)).hi[:, None]
    rest_lo = np.maximum(0, down(lower_total - weights.lo))
    rest_hi = up(upper_total - weights.hi)
    lo = np.maximum(0, down(weights.lo / up(weights.lo + rest_hi)))
    hi = np.minimum(1, up(weights.hi / down(weights.hi + rest_lo)))
    return Bounds(lo, hi)


class ScoreInterval:
    """Verify Voronoi labels of isotropic equal-mass Gaussian prototype flows.

    Use m_j(x)=s_j(x)-s_0(x), s_j(x)=a_j^T x-||a_j||^2/2. With
    B_ji=m_j(a_i), the exact transformed ODE is m'=B softmax(m/beta)-m.
    No numerical basis or perturbation of the original sites is needed.
    """

    @np.errstate(over="raise", invalid="raise", divide="raise", under="ignore")
    def __init__(self, sites, beta):
        self.sites = np.array(sites, dtype=np.float64, copy=True)
        if (self.sites.ndim != 2 or len(self.sites) < 2 or self.sites.shape[1] < 1
                or not np.all(np.isfinite(self.sites)) or not np.isfinite(beta) or beta <= 0):
            raise ValueError("Provide finite row sites and positive bandwidth")
        self.beta = Bounds.exact(beta)
        gram = dot_rows(self.sites, self.sites)
        indices = np.arange(len(self.sites))
        self.half_norms = gram.select((indices, indices)) * HALF
        center_scores = self._contrasts(gram - self.half_norms)
        self.centers = center_scores.T
        self.hull = Bounds(center_scores.lo.min(axis=0), center_scores.hi.max(axis=0))
        self.bump_low = Bounds.exact(self.hull.lo)
        self.bump_high = Bounds.exact(self.hull.hi)
        self.low_columns = self.centers - self.bump_low.select((..., None))
        self.high_columns = self.centers - self.bump_high.select((..., None))

    @staticmethod
    def _contrasts(scores):
        return scores.select((..., slice(1, None))) - scores.select((..., slice(1)))

    @staticmethod
    def _with_reference(contrasts):
        zeros = np.zeros((len(contrasts.lo), 1))
        return Bounds(np.concatenate((zeros, contrasts.lo), axis=-1),
                      np.concatenate((zeros, contrasts.hi), axis=-1))

    def _posterior(self, box):
        return posterior(self._with_reference(box) / self.beta)

    def _mean(self, probabilities):
        weights = probabilities.select((slice(None), None))
        lower = self.bump_low + sum_last(weights * self.low_columns)
        upper = self.bump_high + sum_last(weights * self.high_columns)
        return tighten(tighten(self.hull, lower), upper)

    def _step(self, box, move, remainder):
        tube_fraction = Bounds(0, move.hi)
        tube = blend(box, self.hull, tube_fraction)
        tube = tighten(tube, blend(box, self._mean(self._posterior(tube)), tube_fraction))
        probabilities = self._posterior(tube)
        average = self._mean(probabilities)
        velocity = self._with_reference(average - tube)
        anchor = Bounds.exact(velocity.lo.min(axis=-1, keepdims=True) * HALF.lo
                              + velocity.hi.max(axis=-1, keepdims=True) * HALF.lo)
        weighted = probabilities * (velocity - anchor)
        centered = self.centers - average.select((..., None))
        acceleration = sum_last(weighted.select((slice(None), None)) * centered) / self.beta
        endpoint = blend(box, self._mean(self._posterior(box)), move) + remainder * acceleration
        return tube, tighten(tube, endpoint)

    def _integrate(self, box, winner, horizon, steps):
        step = Bounds.exact(horizon / steps)
        decay = negative_exp(-step)
        move = ONE - decay
        remainder = step - move
        # Both factors denote nonnegative exact real values.
        move = Bounds(np.maximum(0, move.lo), np.minimum(1, move.hi))
        remainder = Bounds(np.maximum(0, remainder.lo), remainder.hi)
        stable = np.ones(len(winner), dtype=bool)
        for _ in range(steps):
            tube, box = self._step(box, move, remainder)
            stable &= occupied(self._with_reference(tube), winner)
        scores = self._with_reference(box)
        switched = scores.lo.max(axis=-1) > scores.hi[np.arange(len(winner)), winner]
        return stable, switched

    @np.errstate(over="raise", invalid="raise", divide="raise", under="ignore")
    def classify(self, queries, horizon, step_ladder=(1, 4, 16, 64)):
        """Refine unresolved rows on a fixed dyadic step ladder."""
        queries = np.asarray(queries, dtype=np.float64)
        if (queries.ndim != 2 or queries.shape[1] != self.sites.shape[1]
                or not np.all(np.isfinite(queries)) or not np.isfinite(horizon) or horizon <= 0):
            raise ValueError("Provide finite compatible row queries and a positive horizon")
        if not step_ladder or any(steps < 1 or steps & (steps - 1) for steps in step_ladder):
            raise ValueError("Use a nonempty ladder of positive powers of two")
        smallest_step = horizon / max(step_ladder)
        if smallest_step * max(step_ladder) != horizon:
            raise ArithmeticError("The encoded horizon does not admit this exact dyadic subdivision")
        scores = self._contrasts(dot_rows(queries, self.sites) - self.half_norms)
        initial = strict_initial(self._with_reference(scores))
        stable = np.zeros(len(initial), dtype=bool)
        switched = np.zeros(len(initial), dtype=bool)
        remaining = np.flatnonzero(initial >= 0)
        row_steps = 0
        for steps in step_ladder:
            if len(remaining) == 0:
                break
            same, change = self._integrate(scores.select(remaining), initial[remaining], horizon, steps)
            row_steps += steps * len(remaining)
            stable[remaining], switched[remaining] = same, change
            remaining = remaining[~(same | change)]
        if np.any(stable & switched):
            raise ArithmeticError("Incompatible score-space certificates")
        return ScoreVerdicts(initial, stable, switched, row_steps)
