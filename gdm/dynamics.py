"""Core-fitted softened attraction with matched static and dynamical readouts."""

from copy import copy
from math import ceil

import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans


def _squared_distance(points, centers, center_norms):
    distance = np.einsum("nd,nd->n", points, points)[:, None] - 2 * points @ centers.T
    return np.maximum(distance + center_norms, 0)


class AttractionClassifier:
    """Learn a mixture potential and classify with its sources or trajectories.

    Readout "static" scores source responsibilities by class after exactly
    `steps` updates. The same readout with zero steps is the parameter-matched
    static classifier. "absorbing" detects the first core-ball crossing of the
    numerical path. "destination" checks association with a validated minimum,
    distance, field magnitude, velocity (if applicable), and dwell time.
    Unresolved predictions have label -1. Class labels are nonnegative integers.
    """

    _runtime_options = {
        "constrained",
        "mode",
        "readout",
        "steps",
        "step_size",
        "damping",
        "position_tolerance",
        "force_tolerance",
        "velocity_tolerance",
        "dwell_time",
    }

    def __init__(
        self,
        n_prototypes=2,
        anisotropic=True,
        learned=True,
        constrained=True,
        mode="gradient",
        readout="destination",
        steps=80,
        step_size=0.15,
        bandwidth=1.0,
        damping=1.4,
        position_tolerance=0.06,
        force_tolerance=0.08,
        velocity_tolerance=0.08,
        dwell_time=0.30,
        random_state=0,
    ):
        self.n_prototypes = n_prototypes
        self.anisotropic = anisotropic
        self.learned = learned
        self.constrained = constrained
        self.mode = mode
        self.readout = readout
        self.steps = steps
        self.step_size = step_size
        self.bandwidth = bandwidth
        self.damping = damping
        self.position_tolerance = position_tolerance
        self.force_tolerance = force_tolerance
        self.velocity_tolerance = velocity_tolerance
        self.dwell_time = dwell_time
        self.random_state = random_state

    def _validate_options(self):
        if self.mode not in {"gradient", "inertial"}:
            raise ValueError("mode must be gradient or inertial")
        if self.readout not in {"static", "absorbing", "destination"}:
            raise ValueError("readout must be static, absorbing, or destination")
        if not isinstance(self.n_prototypes, (int, np.integer)) or self.n_prototypes < 1:
            raise ValueError("n_prototypes must be a positive integer")
        if not isinstance(self.steps, (int, np.integer)) or self.steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        positive = (
            self.bandwidth,
            self.damping,
            self.step_size,
            self.position_tolerance,
            self.force_tolerance,
            self.velocity_tolerance,
            self.dwell_time,
        )
        if not all(np.isfinite(value) and value > 0 for value in positive):
            raise ValueError("bandwidth, damping, step_size and tolerances must be positive and finite")

    def fit(self, X, y):
        self._validate_options()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or not np.all(np.isfinite(X)):
            raise ValueError("X must be a finite nonempty matrix")
        if y.shape != (len(X),) or not np.issubdtype(y.dtype, np.integer) or np.any(y < 0):
            raise ValueError("y must contain one nonnegative integer per row")
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self._fit_geometry(X, y)
        self._initialize_destination()
        return self

    def _fit_geometry(self, X, y):
        d = self.n_features_in_
        centers, cluster_points, class_indices, weights = [], [], [], []
        for class_index, label in enumerate(self.classes_):
            members = X[y == label]
            count = min(self.n_prototypes, len(np.unique(members, axis=0)))
            if count == 1:
                groups = np.zeros(len(members), dtype=int)
            else:
                groups = KMeans(
                    n_clusters=count, random_state=self.random_state, n_init=4
                ).fit_predict(members)
            for cluster in np.unique(groups):
                points = members[groups == cluster]
                center = points.mean(axis=0)
                if count > 1:
                    # Keep multimodal sources on observed core support.
                    center = points[np.argmin(np.sum((points - center) ** 2, axis=1))]
                centers.append(center)
                cluster_points.append(points)
                class_indices.append(class_index)
                part = (0.5 / count + 0.5 * len(points) / len(members)) if self.learned else 1 / count
                weights.append(part / len(self.classes_))

        self.centers_ = np.asarray(centers)
        self.center_norms_ = np.einsum("kd,kd->k", self.centers_, self.centers_)
        self.source_class_idx_ = np.asarray(class_indices)
        self.source_labels_ = self.classes_[self.source_class_idx_]
        self.source_classes_ = np.eye(len(self.classes_))[self.source_class_idx_]
        self.weights_ = np.asarray(weights)
        self.log_weights_ = np.log(self.weights_)
        self._identity_precision = not self.anisotropic and not self.learned
        self._diagonal = d > 8 or not self.anisotropic

        spacing2 = _squared_distance(self.centers_, self.centers_, self.center_norms_)
        np.fill_diagonal(spacing2, np.inf)
        nearest = np.sqrt(spacing2.min(axis=1))
        valid_gaps = nearest[np.isfinite(nearest) & (nearest > 1e-10)]
        typical_gap = np.median(valid_gaps) if len(valid_gaps) else 1.0
        residual = [points - center for points, center in zip(cluster_points, self.centers_)]
        radii = [np.sqrt(np.einsum("nd,nd->n", offset, offset)) for offset in residual]
        radial = max(float(np.median(np.concatenate(radii))), 0.10 * typical_gap, 1e-8)
        self.radial_scale_ = radial
        self.kernel_scale_ = self.bandwidth * max(radial / np.sqrt(d), 0.08 * typical_gap)
        self.kernel_variance_ = self.kernel_scale_**2
        self.patch_radii_ = np.minimum(0.38 * nearest, 1.5 * radial)
        self.core_radii_ = np.minimum(
            0.38 * nearest, [max(np.quantile(r, 0.65), 0.12 * radial) for r in radii]
        )

        precisions = []
        for offset, r in zip(residual, radii):
            scale = np.clip(np.median(r) / radial, 0.75, 1.30) if self.learned else 1.0
            if self.anisotropic and len(offset) >= 3:
                variance = np.mean(offset**2, axis=0)
                mean_variance = float(variance.mean())
                if mean_variance > 1e-20:
                    if self._diagonal:
                        shape = np.clip(0.55 + 0.45 * variance / mean_variance, 0.40, 2.50)
                        precisions.append(1 / (scale**2 * shape))
                    else:
                        covariance = offset.T @ offset / len(offset)
                        eigenvalues, vectors = np.linalg.eigh(covariance / mean_variance)
                        shape = np.clip(0.55 + 0.45 * eigenvalues, 0.40, 2.50)
                        precisions.append((vectors / (scale**2 * shape)) @ vectors.T)
                    continue
            precisions.append((np.ones(d) if self._diagonal else np.eye(d)) / scale**2)
        self.precisions_ = np.asarray(precisions)
        if self._diagonal:
            self.precision_center_ = self.precisions_ * self.centers_
            self.source_curvature_ = self.precisions_.min(axis=1)
        else:
            self.precision_center_ = np.einsum("kij,kj->ki", self.precisions_, self.centers_)
            self.source_curvature_ = np.linalg.eigvalsh(self.precisions_)[:, 0]
        self.quadratic_center_ = np.einsum("kd,kd->k", self.centers_, self.precision_center_)
        self.source_potential_ = self._base_potential(self.centers_)
        self._relaxed_equilibria_ = None

        k = len(self.centers_)
        covariance = d if self._diagonal else d * (d + 1) // 2
        self.n_parameters_ = (
            k * d
            + (k * covariance if self.anisotropic else 0)
            + (k + k - len(self.classes_) if self.learned else 0)
            + 1
        )

    def with_options(self, **options):
        """Reuse fitted geometry while changing dynamics, readout, or solver."""
        invalid = set(options) - self._runtime_options
        if invalid:
            raise ValueError(f"cannot change fitted geometry options {sorted(invalid)}")
        model = copy(self)
        for key, value in options.items():
            setattr(model, key, value)
        model._validate_options()
        model._initialize_destination()
        return model

    def _logits(self, points):
        if self._identity_precision:
            quadratic = _squared_distance(points, self.centers_, self.center_norms_)
        elif self._diagonal:
            quadratic = (points**2) @ self.precisions_.T - 2 * points @ self.precision_center_.T
            quadratic = np.maximum(quadratic + self.quadratic_center_, 0.0)
        else:
            delta = points[:, None, :] - self.centers_
            quadratic = np.einsum("nki,kij,nkj->nk", delta, self.precisions_, delta)
        return self.log_weights_ - quadratic / (2 * self.kernel_variance_)

    def _base_potential(self, points):
        return -self.kernel_variance_ * np.logaddexp.reduce(self._logits(points), axis=1)

    def _responsibilities(self, points):
        logits = self._logits(points)
        maximum = np.max(logits, axis=1)
        weights = np.exp(logits - maximum[:, None])
        total = weights.sum(axis=1)
        return weights / total[:, None], -self.kernel_variance_ * (maximum + np.log(total))

    def _base_field(self, points):
        weights, potential = self._responsibilities(points)
        bias = weights @ self.precision_center_
        if self._identity_precision:
            field = bias - points
        elif self._diagonal:
            field = bias - points * (weights @ self.precisions_)
        else:
            field = bias - np.einsum("nk,kij,nj->ni", weights, self.precisions_, points)
        return potential, field

    def _local_quadratic(self, offsets, sources):
        if self._diagonal:
            gradient = self.precisions_[sources] * offsets
        else:
            gradient = np.einsum("nij,nj->ni", self.precisions_[sources], offsets)
        return np.einsum("nd,nd->n", offsets, gradient), gradient

    def _potential_and_field(self, points):
        potential, force = self._base_field(points)
        if not self.constrained:
            return potential, force
        q = _squared_distance(points, self.centers_, self.center_norms_)
        nearby, source = np.where(q < self.patch_radii_**2)
        if len(nearby) == 0:
            return potential, force

        delta = points[nearby] - self.centers_[source]
        distance = np.linalg.norm(delta, axis=1)
        radius = self.patch_radii_[source]
        # Quintic smoothstep across the annulus between r/2 and r.
        t = np.clip(2 * distance / radius - 1, 0, 1)
        blend = 1 - t**3 * (10 - 15 * t + 6 * t**2)
        derivative = -60 * t**2 * (1 - t) ** 2 / radius
        gradient_blend = np.divide(
            derivative, distance, out=np.zeros_like(distance), where=distance > 0
        )[:, None] * delta

        quadratic, local_gradient = self._local_quadratic(delta, source)
        local_potential = self.source_potential_[source] + 0.5 * quadratic
        difference = local_potential - potential[nearby]
        force[nearby] = (
            (1 - blend[:, None]) * force[nearby]
            - blend[:, None] * local_gradient
            - difference[:, None] * gradient_blend
        )
        potential[nearby] += blend * difference
        return potential, force

    def _minimum_curvature(self, point):
        weights, _ = self._responsibilities(point[None, :])
        weights = weights[0]
        if self._diagonal:
            gradients = self.precisions_ * (point - self.centers_)
            component_hessian = np.diag(weights @ self.precisions_)
        else:
            gradients = np.einsum("kij,kj->ki", self.precisions_, point - self.centers_)
            component_hessian = np.einsum("k,kij->ij", weights, self.precisions_)
        centered = gradients - weights @ gradients
        covariance = (centered.T * weights) @ centered
        hessian = component_hessian - covariance / self.kernel_variance_
        return float(np.linalg.eigvalsh((hessian + hessian.T) / 2)[0])

    def _find_relaxed_equilibria(self):
        positions, source_indices, eigenvalues, residuals = [], [], [], []

        def objective(point):
            value, force = self._base_field(point[None, :])
            return value[0], -force[0]

        for i, (center, ownership_radius) in enumerate(zip(self.centers_, self.patch_radii_)):
            if ownership_radius <= 1e-10:
                continue
            result = minimize(
                objective,
                center,
                method="L-BFGS-B",
                jac=True,
                options={"gtol": 1e-9, "ftol": 1e-15, "maxiter": 80},
            )
            point = result.x
            if not np.all(np.isfinite(point)) or np.linalg.norm(point - center) >= ownership_radius:
                continue
            residual = float(np.linalg.norm(objective(point)[1]))
            minimum_curvature = self._minimum_curvature(point)
            if residual <= 1e-7 * max(1.0, self.radial_scale_) and minimum_curvature > 1e-5:
                positions.append(point)
                source_indices.append(i)
                eigenvalues.append(minimum_curvature)
                residuals.append(residual)
        return (
            np.asarray(positions).reshape(-1, self.n_features_in_),
            np.asarray(source_indices, dtype=int),
            np.asarray(eigenvalues),
            np.asarray(residuals),
        )

    def _initialize_destination(self):
        if self.readout != "destination":
            return
        if self.constrained:
            indices = np.flatnonzero(self.patch_radii_ > 1e-10)
            self.equilibria_ = self.centers_[indices]
            self.equilibrium_curvature_ = self.source_curvature_[indices]
            self.equilibrium_residual_ = np.zeros(len(indices))
        else:
            if self._relaxed_equilibria_ is None:
                self._relaxed_equilibria_ = self._find_relaxed_equilibria()
            self.equilibria_, indices, self.equilibrium_curvature_, self.equilibrium_residual_ = (
                self._relaxed_equilibria_
            )
        self.equilibrium_sources_ = indices
        self.equilibrium_labels_ = self.source_labels_[indices]
        self.equilibrium_norms_ = np.einsum("kd,kd->k", self.equilibria_, self.equilibria_)
        self.equilibrium_inner_radii_ = self.patch_radii_[indices] / 2
        self.equilibrium_tolerances_ = np.minimum(
            self.position_tolerance * self.radial_scale_, 0.4 * self.equilibrium_inner_radii_
        )

    def potential(self, X):
        """Return the actual mixture potential, including any stability patches."""
        return self._potential_and_field(self._check_X(X))[0]

    def field(self, X):
        """Return the gradient vector field F = -grad U."""
        return self._potential_and_field(self._check_X(X))[1]

    def _check_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.all(np.isfinite(X)):
            raise ValueError("X must be a finite matrix with the fitted feature dimension")
        return X

    def _static(self, points):
        weights, _ = self._responsibilities(points)
        pred = self.classes_[(weights @ self.source_classes_).argmax(axis=1)]
        return np.where(np.isfinite(weights).all(axis=1), pred, -1)

    def _advance(self, points, velocity, force):
        if velocity is None:
            return points + self.step_size * force, None
        # Semi-implicit Euler with damping implicit in the velocity update.
        updated_velocity = (velocity + self.step_size * force) / (1 + self.step_size * self.damping)
        return points + self.step_size * updated_velocity, updated_velocity

    def _first_core_hit(self, start, segment=None):
        q = _squared_distance(start, self.centers_, self.center_norms_)
        squared_radius = self.core_radii_**2
        positive_radius = squared_radius > 0
        if segment is None:
            crossing = np.where((q <= squared_radius) & positive_radius, 0.0, np.inf)
        else:
            # Solve a*t² + 2*b*t + c = 0, accepting the earliest entry in [0, 1].
            a = np.einsum("nd,nd->n", segment, segment)[:, None]
            b = np.einsum("nd,nd->n", start, segment)[:, None] - segment @ self.centers_.T
            c = q - squared_radius
            discriminant = b**2 - a * c
            denominator = -b + np.sqrt(np.maximum(discriminant, 0))
            root = np.divide(c, denominator, out=np.full_like(q, np.inf), where=denominator > 0)
            valid = (
                positive_radius
                & (a > 0)
                & (b < 0)
                & (discriminant >= 0)
                & (root >= 0)
                & (root <= 1)
            )
            crossing = np.where(c <= 0, np.where(positive_radius, 0.0, np.inf), np.where(valid, root, np.inf))
        sources = crossing.argmin(axis=1)
        hit = np.isfinite(crossing[np.arange(len(start)), sources])
        return np.where(hit, self.source_labels_[sources], -1)

    def _integrate_static(self, X, nfe):
        points = X
        velocity = np.zeros_like(X) if self.mode == "inertial" else None
        for _ in range(self.steps):
            _, force = self._potential_and_field(points)
            nfe += 1
            points, velocity = self._advance(points, velocity, force)
        return self._static(points)

    def _integrate_absorbing(self, X, nfe):
        pred = self._first_core_hit(X)
        active = np.flatnonzero(pred < 0)
        points = X[active]
        velocity = np.zeros_like(points) if self.mode == "inertial" else None
        for _ in range(self.steps):
            if len(active) == 0:
                break
            _, force = self._potential_and_field(points)
            nfe[active] += 1
            updated, velocity = self._advance(points, velocity, force)
            labels = self._first_core_hit(points, updated - points)
            pred[active] = labels
            unresolved = labels < 0
            active, points = active[unresolved], updated[unresolved]
            if velocity is not None:
                velocity = velocity[unresolved]
        return pred

    def _integrate_destination(self, X, nfe):
        pred = np.full(len(X), -1, dtype=int)
        if len(self.equilibria_) == 0:
            return pred
        active = np.arange(len(X))
        points = X
        velocity = np.zeros_like(points) if self.mode == "inertial" else None
        prior = np.full(len(X), -1)
        dwell = np.zeros(len(X), dtype=int)
        dwell_intervals = ceil(self.dwell_time / self.step_size - 1e-12)
        force_bound2 = (self.force_tolerance * self.radial_scale_) ** 2
        speed_bound2 = (self.velocity_tolerance * self.radial_scale_) ** 2
        for step in range(self.steps + 1):
            if len(active) == 0:
                break
            _, force = self._potential_and_field(points)
            nfe[active] += 1
            distance2 = _squared_distance(points, self.equilibria_, self.equilibrium_norms_)
            closest = distance2.argmin(axis=1)
            selected_distance2 = distance2[np.arange(len(points)), closest]
            near = selected_distance2 <= self.equilibrium_tolerances_[closest] ** 2
            near &= np.einsum("nd,nd->n", force, force) <= force_bound2
            if velocity is not None:
                speed2 = np.einsum("nd,nd->n", velocity, velocity)
                near &= speed2 <= speed_bound2
                if self.constrained:
                    # An energy below the sphere's lowest potential gives
                    # an invariant region of the local damped oscillator.
                    candidate = closest[near]
                    offsets = points[near] - self.equilibria_[candidate]
                    quadratic, _ = self._local_quadratic(offsets, self.equilibrium_sources_[candidate])
                    barrier = self.equilibrium_curvature_[candidate] * self.equilibrium_inner_radii_[candidate] ** 2
                    near[near] = quadratic + speed2[near] < barrier
            current = np.where(near, closest, -1)
            dwell = np.where(near & (current == prior), dwell + 1, 0)
            done = near & (dwell >= dwell_intervals)
            pred[active[done]] = self.equilibrium_labels_[closest[done]]
            if step == self.steps:
                break
            pending = ~done
            active, prior, dwell = active[pending], current[pending], dwell[pending]
            remaining_velocity = None if velocity is None else velocity[pending]
            points, velocity = self._advance(points[pending], remaining_velocity, force[pending])
        return pred

    def predict_details(self, X):
        """Return predictions, original static scores used as fallback, and NFE."""
        X = self._check_X(X)
        nfe = np.zeros(len(X), dtype=int)
        with np.errstate(over="ignore", invalid="ignore"):
            if self.readout == "static":
                pred = self._integrate_static(X, nfe)
            elif self.readout == "absorbing":
                pred = self._integrate_absorbing(X, nfe)
            else:
                pred = self._integrate_destination(X, nfe)
            fallback = pred.copy()
            unresolved = pred < 0
            if np.any(unresolved):
                fallback[unresolved] = self._static(X[unresolved])
        return {
            "pred": pred,
            "fallback": fallback,
            "resolved": ~unresolved,
            "nfe": nfe,
        }

    def predict(self, X):
        return self.predict_details(X)["pred"]
