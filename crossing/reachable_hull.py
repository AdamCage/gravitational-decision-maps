"""Outward bounds on the original Gaussian path via the reachable prototype hull."""

from typing import NamedTuple

import numpy as np

from crossing.affine_certificate import (
    Bounds, HALF, ONE, dot_rows, exponential_weights, negative_exp, sum_last,
)


class HullVerdicts(NamedTuple):
    strict: np.ndarray
    slab: np.ndarray
    hull: np.ndarray


class ReachableHullGuard:
    """Certify an affine margin reserve without a four-index site array.

    For u = 1 - exp(-t), variation of constants places the trajectory in
    (1-u)x + u conv(sites). Enclose all Gaussian logits on this swept hull,
    then bound the signed numerator of the posterior mean rival margin.
    A lower bound on the mean margin can be fed back into the same scalar
    integrating-factor formula. The strict test covers every t up to T.
    """

    @np.errstate(over="raise", invalid="raise", divide="raise", under="ignore")
    def __init__(self, readout, sites, masses, beta):
        self.readout = readout
        self.sites = np.array(readout._points(sites), copy=True)
        weights = np.array(masses, dtype=np.float64, copy=True)
        if (len(self.sites) == 0 or weights.shape != self.sites.shape[:1] or not np.isfinite(weights).all()
                or np.any(weights <= 0) or not np.isfinite(beta) or beta <= 0):
            raise ValueError("Expected finite mixture parameters with positive masses and variance")
        self.weights = Bounds.exact(weights)
        self.beta = Bounds.exact(beta)
        gram = dot_rows(self.sites, self.sites)
        idx = np.arange(len(self.sites))
        self.offset = gram.select((idx, idx)) * HALF
        logits = (gram - self.offset) / self.beta
        self.low = Bounds.exact(logits.lo.min(axis=0))
        self.high = Bounds.exact(logits.hi.max(axis=0))
        scores = readout.scores(self.sites).T
        self.site_margins = scores.select((slice(None), None)) - scores.select((None, slice(None)))

    @np.errstate(over="raise", invalid="raise", divide="raise", under="ignore")
    def certify(self, queries, horizon):
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("Expected a finite positive horizon")
        x = self.readout._points(queries)
        cell = self.readout.initial_cell(x)
        decay = negative_exp(-Bounds.exact(horizon))
        move = ONE - decay
        budget = cell.margins * ((decay - self.readout.reserve) / move)
        terms = self.site_margins.select(cell.winner) + budget.select((..., None))
        slab = cell.strict & np.all((terms.lo.min(axis=-1) > 0) | ~cell.other, axis=1)
        unresolved = np.flatnonzero(cell.strict & ~slab)
        hull = slab.copy()
        if len(unresolved):
            base = (dot_rows(x[unresolved], self.sites) - self.offset) / self.beta
            low_delta = (self.low - base).lo
            high_delta = (self.high - base).hi
            swept = base + move * Bounds(np.minimum(low_delta, 0), np.maximum(high_delta, 0))
            mass = exponential_weights(swept) * self.weights
            signs = sum_last(terms.select(unresolved) * mass.select((slice(None), None)))
            hull[unresolved] = np.all((signs.lo > 0) | ~cell.other[unresolved], axis=1)
        return HullVerdicts(cell.strict, slab, hull)
