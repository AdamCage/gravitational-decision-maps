"""Gaussian class scores, their attraction flow, and finite-time label certificates."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from sklearn.cluster import KMeans


def _norm(vectors):
    return np.linalg.norm(vectors, axis=-1)


def _integrals(rate, time):
    """Return integrals of exp(rate*s) and (time-s)*exp(rate*s)."""
    z = np.asarray(rate) * time
    remainder = np.expm1(z)
    safe = np.where(np.abs(z) < 1e-4, 1.0, z)
    small = np.abs(z) < 1e-4
    first = np.where(small, 1 + z * (0.5 + z * (1 / 6 + z / 24)), remainder / safe)
    second = np.where(small, 0.5 + z * (1 / 6 + z * (1 / 24 + z / 120)), (remainder - z) / safe**2)
    return time * first, time**2 * second


def _relaxed_integral(rate, time):
    if time < 1e-4 and np.max(np.abs(rate * time), initial=0) < 1e-3:
        return time**2 * (0.5 + (rate - 1) * time / 6 + (rate**2 - rate + 1) * time**2 / 24)
    return np.maximum(0, _integrals(rate, time)[0] - np.exp(-time) * _integrals(rate + 1, time)[0])


@dataclass(frozen=True)
class Certificate:
    initial: np.ndarray
    endpoint: np.ndarray
    stable: np.ndarray
    forced_crossing: np.ndarray
    ball_stable: np.ndarray
    taylor_endpoint: np.ndarray
    drag_endpoint: np.ndarray
    state_endpoint: np.ndarray
    linear_label: np.ndarray
    linear_strength: np.ndarray
    drag_strength: np.ndarray
    static_margin: np.ndarray
    radius: np.ndarray


class MixtureFlow:
    """Attraction to a mixture of class shells with shared spherical precision.

    The score uses beta times a log sum with the common quadratic term removed.
    Its pairwise differences are unchanged. Physical velocity is mixture mean minus x.
    """

    def __init__(self, centers, labels, masses, beta):
        self.centers = np.asarray(centers, dtype=float)
        self.source_labels = np.asarray(labels)
        weights = np.asarray(masses, dtype=float)
        if self.centers.ndim != 2 or len(self.centers) < 2 or not np.isfinite(self.centers).all():
            raise ValueError("Expected at least two finite prototype vectors")
        if (self.source_labels.shape != (len(self.centers),) or weights.shape != self.source_labels.shape
                or not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isfinite(beta) or beta <= 0):
            raise ValueError("Each source needs a label and positive mass; beta must be positive")
        self.classes = np.unique(self.source_labels)
        if len(self.classes) < 2:
            raise ValueError("Expected at least two classes")
        self.groups = tuple(np.flatnonzero(self.source_labels == c) for c in self.classes)
        sizes = {len(g) for g in self.groups}
        self.group_index = np.stack(self.groups) if len(sizes) == 1 else None
        self.beta = float(beta)
        self.origin = self.centers[0].copy()
        self.centered = self.centers - self.origin
        self.centers_t = np.ascontiguousarray(self.centered.T)
        self.mean_square = np.einsum("kd,kd->k", self.centered, self.centered)
        self.offsets = np.log(weights) - self.mean_square / (2 * self.beta)
        distances = cdist(self.centers, self.centers)
        self.group_distance2 = tuple(distances[np.ix_(g, g)]**2 for g in self.groups)
        self.diameter = float(distances.max())
        self.group_distances = np.array([[distances[np.ix_(a, b)].max() for b in self.groups] for a in self.groups])
        self.group_diameters = self.group_distances.diagonal().copy()
        self.global_covariance = self.diameter**2 / 4
        self.global_group_covariance = self.group_diameters**2 / 4
        self.global_rate = self.global_covariance / self.beta - 1
        self.other = ~np.eye(len(self.classes), dtype=bool)

    @classmethod
    def fit(cls, x, y, *, prototypes, bandwidth_ratio, seed):
        """Fit equal class masses and equally weighted sources within each class."""
        all_centers, all_labels, all_weights = [], [], []
        for label in np.unique(y):
            points = x[y == label]
            n = min(prototypes, len(np.unique(points, axis=0)))
            centers = (KMeans(n_clusters=n, n_init=4, random_state=seed).fit(points).cluster_centers_
                       if n > 1 else points.mean(axis=0, keepdims=True))
            all_centers.extend(centers)
            all_labels.extend([label] * n)
            all_weights.extend([1 / n] * n)
        diameter = float(cdist(all_centers, all_centers).max())
        return cls(all_centers, all_labels, all_weights, max(bandwidth_ratio * diameter, 1e-4) ** 2)

    def _rel_logits(self, x_relative):
        return x_relative @ self.centers_t / self.beta + self.offsets

    def scores(self, x):
        return self._class_scores(self._rel_logits(x - self.origin))

    def _class_scores(self, logits):
        if self.group_index is not None:
            grouped = logits[:, self.group_index]
            maximum = grouped.max(axis=2, keepdims=True)
            return self.beta * (maximum[:, :, 0] + np.log(np.exp(grouped - maximum).sum(axis=2)))
        return self.beta * np.column_stack([logsumexp(logits[:, g], axis=1) for g in self.groups])

    def predict(self, x):
        return self.classes[self.scores(x).argmax(axis=1)]

    def field(self, x):
        return self._rel_field(x - self.origin)

    def _rel_field(self, relative):
        return self._posterior(self._rel_logits(relative)) @ self.centered - relative

    @staticmethod
    def _posterior(logits):
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        return weights / weights.sum(axis=1, keepdims=True)

    def moments(self, x):
        """Class scores and means plus total and per-class covariance traces."""
        relative = x - self.origin
        logits = self._rel_logits(relative)
        means, scores, traces = [], [], []
        for g, distance2 in zip(self.groups, self.group_distance2, strict=True):
            part = logits[:, g]
            maximum = part.max(axis=1, keepdims=True)
            weights = np.exp(part - maximum)
            total = weights.sum(axis=1, keepdims=True)
            weights /= total
            mean = weights @ self.centered[g]
            means.append(mean)
            scores.append((maximum + np.log(total))[:, 0])
            traces.append(0.5 * np.einsum("nk,nk->n", weights @ distance2, weights))
        class_means = np.stack(means, axis=1)
        log_scores = np.stack(scores, axis=1)
        class_traces = np.column_stack(traces)
        post = np.exp(log_scores - log_scores.max(axis=1, keepdims=True))
        post /= post.sum(axis=1, keepdims=True)
        mixture_mean = np.einsum("nc,ncd->nd", post, class_means)
        centered = class_means - mixture_mean[:, None, :]
        mixture_trace = np.einsum("nc,nc->n", post, class_traces + np.einsum("ncd,ncd->nc", centered, centered))
        return self.beta * log_scores, class_means, mixture_mean - relative, class_traces, mixture_trace

    @staticmethod
    def _variance_envelope(trace, cap, exponent):
        # Clamping prevents overflow. The mathematical result is unchanged where
        # the exponential is above the diameter cap and the trace is positive.
        log_trace = np.log(np.maximum(trace + 1e-14, np.finfo(float).tiny))
        return np.exp(np.minimum(np.log(np.maximum(cap, np.finfo(float).tiny)), log_trace + exponent))

    def _tube(self, x, velocity, trace, time):
        speed = _norm(velocity)
        source_distances2 = cdist(x, self.centers, metric="sqeuclidean")
        convex_radius = -np.expm1(-time) * np.sqrt(source_distances2.max(axis=1))
        radius = np.minimum(convex_radius, speed * _integrals(self.global_rate, time)[0])
        global_radius = radius
        covariance_rate = np.full(len(x), self.global_covariance / self.beta)
        for _ in range(2):
            variance = self._variance_envelope(trace, self.global_covariance, self.diameter * radius / self.beta)
            covariance_rate = np.minimum(covariance_rate, variance / self.beta)
            radius = np.minimum(radius, speed * _integrals(covariance_rate - 1, time)[0])
        return speed, global_radius, radius, covariance_rate

    def _endpoint_labels(self, lower):
        positive = np.all((lower > 0) | ~self.other, axis=2)
        if np.any(positive.sum(axis=1) > 1):
            raise ArithmeticError("Inconsistent class bounds")
        return np.where(positive.any(axis=1), self.classes[positive.argmax(axis=1)], -1)

    def _state_lower(self, x, velocity, speed, covariance_rate, time):
        predicted_state = (x - self.origin) - np.expm1(-time) * velocity
        scores = self._class_scores(self._rel_logits(predicted_state))
        margins = scores[:, :, None] - scores[:, None, :]
        uncertainty = covariance_rate * speed * _relaxed_integral(covariance_rate - 1, time)
        error = uncertainty[:, None, None] * self.group_distances
        return margins - error - 1e-10 * (1 + np.abs(margins) + error), scores

    def certify_endpoint(self, x, time):
        """Fast state-ball certificate used in the selective forward solver."""
        if not np.isfinite(time) or time <= 0 or self.classes.min() < 0:
            raise ValueError("Use a positive finite time and nonnegative labels")
        relative = x - self.origin
        p = self._posterior(self._rel_logits(relative))
        mean = p @ self.centered
        trace = np.einsum("nk,nk->n", p, cdist(mean, self.centered, metric="sqeuclidean"))
        velocity = mean - relative
        speed, _, _, covariance_rate = self._tube(x, velocity, trace, time)
        lower, _ = self._state_lower(x, velocity, speed, covariance_rate, time)
        return self._endpoint_labels(lower)

    def exponential_scores(self, x, time):
        """Uncertified one-step ETD1 scores, reusing starting logits."""
        rel = x - self.origin
        logits = self._rel_logits(rel)
        velocity = self._posterior(logits) @ self.centered - rel
        end = self._class_scores(self._rel_logits(rel - np.expm1(-time) * velocity))
        return self._class_scores(logits), end

    def certify(self, x, time):
        """Certify a stable initial class or a different class at the endpoint.

        `endpoint=-1` denotes an uncertified endpoint (class labels are nonnegative).
        Forced crossing can be certified independently of the exact endpoint class.
        Theory applies to the exact autonomous flow, with a conservative floating
        margin guard in this implementation. It is not an interval arithmetic solver.
        """
        if not np.isfinite(time) or time <= 0 or self.classes.min() < 0:
            raise ValueError("Use a positive finite time and nonnegative labels")
        scores, means, velocity, class_trace, trace = self.moments(x)
        deriv = np.einsum("ncd,nd->nc", means, velocity)
        lead = scores.argmax(axis=1)
        row = np.arange(len(x))
        pairs = scores[:, :, None] - scores[:, None, :]
        leading = pairs[row, lead].copy()
        leading[row, lead] = np.inf
        static_margin = leading.min(axis=1)

        relax = -np.expm1(-time)
        speed, global_r, radius, covariance_rate = self._tube(x, velocity, trace, time)
        rate = covariance_rate - 1

        cv = self._variance_envelope(class_trace, self.global_group_covariance[None, :],
                                     radius[:, None] * self.group_diameters / self.beta) / self.beta
        hessian = np.maximum(cv[:, :, None], cv[:, None, :])
        gradient = np.minimum(self.group_distances, _norm(means[:, :, None] - means[:, None, :]) +
                              hessian * radius[:, None, None])
        _, i1 = _integrals(rate, time)
        _, i2 = _integrals(2 * rate, time)
        jacobian = np.maximum(1, rate)
        error = hessian * (speed**2 * i2)[:, None, None] + gradient * (jacobian * speed * i1)[:, None, None]
        linear = scores + time * deriv
        approximate_margin = linear[:, :, None] - linear[:, None, :]
        guard = 1e-10 * (1 + np.abs(approximate_margin) + error)
        taylor_low = approximate_margin - error - guard

        # Integrate known exponential relaxation exactly and bound only covariance curvature.
        drag_error = (hessian * (speed**2 * _relaxed_integral(2 * rate, time))[:, None, None]
                      + gradient * (covariance_rate * speed * _relaxed_integral(rate, time))[:, None, None])
        drift = scores + relax * deriv
        drag_margin = drift[:, :, None] - drift[:, None, :]
        drag_low = drag_margin - drag_error - 1e-10 * (1 + np.abs(drag_margin) + drag_error)

        low = np.maximum(taylor_low, drag_low)
        state_low, _ = self._state_lower(x, velocity, speed, covariance_rate, time)
        endpoint = self._endpoint_labels(np.maximum(low, state_low))
        stable = (static_margin > 1e-10) & (self._endpoint_labels(low) == self.classes[lead])
        crosses = (static_margin > 1e-10) & np.any((np.maximum(low, state_low)[row, :, lead] > 0) & self.other[lead], axis=1)
        # Baseline certificate with only a global-ball displacement and a Lipschitz margin.
        ball_low = leading - global_r[:, None] * self.group_distances[lead]
        ball_stable = np.all(ball_low > 1e-10, axis=1)
        current_lin = linear[row, lead]
        rival = linear.copy()
        rival[row, lead] = -np.inf
        current_drag = drift[row, lead].copy()
        drift[row, lead] = -np.inf
        return Certificate(self.classes[lead], endpoint, stable, crosses, ball_stable,
                           self._endpoint_labels(taylor_low), self._endpoint_labels(drag_low), self._endpoint_labels(state_low),
                           self.classes[linear.argmax(axis=1)],
                           rival.max(axis=1) - current_lin, drift.max(axis=1) - current_drag,
                           static_margin, radius)


def reference_path(model, x, checkpoints):
    """DOP853 continuous-flow reference at explicitly declared positive times."""
    times = np.asarray(checkpoints, dtype=float)
    shape = x.shape
    result = solve_ivp(lambda _, state: model._rel_field(state.reshape(shape)).ravel(),
                       (0, times[-1]), (x - model.origin).ravel(), method="DOP853", t_eval=times,
                       rtol=2e-10, atol=3e-12, max_step=0.125)
    if not result.success or len(result.t) != len(times):
        raise RuntimeError(f"Reference integration failed: {result.message}")
    return result.y.T.reshape(len(times), *shape) + model.origin, result.nfev


def rk4(model, x, time, max_step):
    """Fixed-step RK4 endpoint with deterministic per-row field evaluation count."""
    n = int(np.ceil(time / max_step))
    if n < 1:
        raise ValueError("Positive horizon and step required")
    step = time / n
    state = x - model.origin
    for _ in range(n):
        k1 = model._rel_field(state)
        k2 = model._rel_field(state + 0.5 * step * k1)
        k3 = model._rel_field(state + 0.5 * step * k2)
        k4 = model._rel_field(state + step * k3)
        state += (step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return state + model.origin, 4 * n


def hybrid(model, x, time, max_step):
    """Use certified endpoint labels, applying the same solver only on the rest."""
    prediction = model.certify_endpoint(x, time)
    pending = prediction < 0
    nfe = 0
    if np.any(pending):
        state, nfe = rk4(model, x[pending], time, max_step)
        prediction[pending] = model.predict(state)
    return prediction, len(x) + int(pending.sum()) * nfe


def triage(model, x, time, max_step, buffer):
    """Empirical shortcut. Integrate predicted changes and small one-step margins."""
    start, end = model.exponential_scores(x, time)
    initial = start.argmax(axis=1)
    indices = np.arange(len(x))
    projected_margin = end[indices, initial].copy()
    end[indices, initial] = -np.inf
    pending = projected_margin - end.max(axis=1) <= model.beta * buffer
    predicted = model.classes[initial]
    nfe = 0
    if np.any(pending):
        state, nfe = rk4(model, x[pending], time, max_step)
        predicted[pending] = model.predict(state)
    return predicted, len(x) + int(pending.sum()) * nfe
