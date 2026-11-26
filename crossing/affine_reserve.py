"""Outward-rounded path and segment checks for a frozen affine readout."""

from typing import NamedTuple

import numpy as np

from crossing.affine_certificate import (
    Bounds, HALF, ONE, dot_rows, exponential_weights, negative_exp, sum_last,
)


class InitialCell(NamedTuple):
    winner: np.ndarray
    other: np.ndarray
    margins: Bounds
    strict: np.ndarray


class AffineReadout:
    """Treat the binary64 scores and query coordinates as exact real data."""

    def __init__(self, weights, biases, reserve):
        self.weights = np.array(weights, dtype=np.float64, copy=True)
        self.biases = np.array(biases, dtype=np.float64, copy=True)
        if (self.weights.ndim != 2 or self.weights.shape[0] < 2 or self.weights.shape[1] < 1
                or self.biases.shape != self.weights.shape[:1] or not np.all(np.isfinite(self.weights))
                or not np.all(np.isfinite(self.biases)) or not 0 < reserve < 1):
            raise ValueError("Expected a finite affine readout and a reserve strictly between zero and one")
        self.reserve = Bounds.exact(reserve)

    def _points(self, points):
        rows = np.asarray(points, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != self.weights.shape[1] or not np.all(np.isfinite(rows)):
            raise ValueError("Expected a finite array of compatible queries")
        return rows

    def scores(self, points):
        return dot_rows(self._points(points), self.weights) + Bounds.exact(self.biases)

    def initial_cell(self, points):
        scores = self.scores(points)
        winner = scores.lo.argmax(axis=1)
        other = winner[:, None] != np.arange(len(self.weights))
        margins = scores.select((np.arange(len(winner)), winner, None)) - scores
        return InitialCell(winner, other, margins, np.all((margins.lo > 0) | ~other, axis=1))

    @np.errstate(over="raise", invalid="raise", divide="raise", under="ignore")
    def segment(self, start, endpoint):
        """Check the reserve on every point of the actual straight segment."""
        if np.shape(start) != np.shape(endpoint):
            raise ValueError("Start and endpoint arrays must have the same shape")
        cell = self.initial_cell(start)
        scores = self.scores(endpoint)
        margin = scores.select((np.arange(len(cell.winner)), cell.winner, None)) - scores
        excess = margin - self.reserve * cell.margins
        return cell.strict & np.all((excess.lo > 0) | ~cell.other, axis=1)


class ProjectedInput(NamedTuple):
    cell: InitialCell
    logits: Bounds
    lower_delta: np.ndarray
    upper_delta: np.ndarray


class GaussianReservePath:
    """Bound every margin along the prescribed spherical Gaussian flow."""

    @np.errstate(over="raise", invalid="raise", divide="raise", under="ignore")
    def __init__(self, readout, sites, masses, beta):
        self.readout = readout
        self.sites = np.array(readout._points(sites), copy=True)
        masses = np.asarray(masses, dtype=np.float64)
        if (len(self.sites) < 1 or masses.shape != self.sites.shape[:1] or not np.all(np.isfinite(masses))
                or np.any(masses <= 0) or not np.isfinite(beta) or beta <= 0):
            raise ValueError("Expected finite mixture parameters with positive masses and bandwidth")
        self.masses = Bounds.exact(masses.copy())
        self.beta = Bounds.exact(beta)
        gram = dot_rows(self.sites, self.sites)
        i = np.arange(len(self.sites))
        self.offset = gram.select((i, i)) * HALF
        site_logits = (gram - self.offset) / self.beta

        weights = readout.weights
        literal = Bounds.exact(weights)
        directions = literal.select((slice(None), None)) - literal.select((None, slice(None)))
        self.constant = np.all(weights[:, None] == weights[None, :], axis=2)
        norm2 = sum_last(directions * directions)
        norm2 = Bounds(np.where(self.constant, 1, norm2.lo), np.where(self.constant, 1, norm2.hi))
        slopes = dot_rows(weights, self.sites)
        slopes = slopes.select((slice(None), None)) - slopes.select((None, slice(None)))
        self.slopes = slopes / (norm2 * self.beta).select((..., None))
        site_scores = readout.scores(self.sites).T
        self.coordinates = site_scores.select((slice(None), None)) - site_scores.select((None, slice(None)))
        projection = site_logits - self.coordinates.select((..., None)) * self.slopes.select((..., None, slice(None)))
        self.lower_projection = projection.lo.min(axis=-2)
        self.upper_projection = projection.hi.max(axis=-2)

    @np.errstate(over="raise", invalid="raise", divide="raise", under="ignore")
    def prepare(self, queries):
        cell = self.readout.initial_cell(queries)
        base = (dot_rows(self.readout._points(queries), self.sites) - self.offset) / self.beta
        logits = base.select((slice(None), None)) - cell.margins.select((..., None)) * self.slopes.select(cell.winner)
        lower = Bounds.exact(self.lower_projection[cell.winner]) - logits
        upper = Bounds.exact(self.upper_projection[cell.winner]) - logits
        return ProjectedInput(cell, logits, np.minimum(0, lower.lo), np.maximum(0, upper.hi))

    @np.errstate(over="raise", invalid="raise", divide="raise", under="ignore")
    def certify(self, prepared, horizon):
        """Return local and global certificates for the same positive scalar horizon."""
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("Expected a finite positive horizon")
        cell = prepared.cell
        decay = negative_exp(-Bounds.exact(horizon))
        move = ONE - decay
        projected = prepared.logits + move * Bounds(prepared.lower_delta, prepared.upper_delta)
        mass = self.masses * exponential_weights(projected)
        budget = cell.margins * ((decay - self.readout.reserve) / move)
        terms = self.coordinates.select(cell.winner) + budget.select((..., None))
        exempt = self.constant[cell.winner]
        global_valid = cell.strict & np.all((terms.lo.min(axis=-1) > 0) | exempt, axis=1)
        local_valid = cell.strict & np.all((sum_last(terms * mass).lo > 0) | exempt, axis=1)
        return local_valid | global_valid, global_valid
