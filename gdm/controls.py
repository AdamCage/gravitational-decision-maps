"""Core-only static controls and the feature-space SA-nODE comparator."""

import numpy as np
from scipy.linalg import hadamard
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


CONTROL_GRIDS = {
    "knn": (
        {"n_neighbors": 1, "weights": "uniform"},
        {"n_neighbors": 3, "weights": "distance"},
        {"n_neighbors": 7, "weights": "uniform"},
        {"n_neighbors": 15, "weights": "distance"},
    ),
    "centroid": ({},),
    "prototypes": tuple({"count": count} for count in (2, 3, 4, 6)),
    "mahalanobis": tuple({"shrinkage": value} for value in (0.15, 0.40, 0.70, 0.95)),
    "rbf_svm": (
        {"C": 0.5, "gamma_scale": 0.5},
        {"C": 4.0, "gamma_scale": 0.25},
        {"C": 8.0, "gamma_scale": 1.0},
        {"C": 32.0, "gamma_scale": 4.0},
    ),
    "gmm": (
        {"components": 1, "covariance_type": "diag", "reg_covar": 0.03},
        {"components": 1, "covariance_type": "full", "reg_covar": 0.08},
        {"components": 2, "covariance_type": "diag", "reg_covar": 0.03},
        {"components": 3, "covariance_type": "diag", "reg_covar": 0.08},
    ),
    "mlp": (
        {"hidden_layer_sizes": (8,), "alpha": 0.10},
        {"hidden_layer_sizes": (16,), "alpha": 1.0},
        {"hidden_layer_sizes": (32,), "alpha": 0.01},
        {"hidden_layer_sizes": (32, 16), "alpha": 0.10},
    ),
    "mlp_wide": (
        {"hidden_layer_sizes": (64, 32), "alpha": 0.10},
        {"hidden_layer_sizes": (96, 64), "alpha": 1.0},
        {"hidden_layer_sizes": (128, 64), "alpha": 0.01},
        {"hidden_layer_sizes": (256, 128), "alpha": 0.10},
    ),
    "boosted_trees": (
        {"max_leaf_nodes": 7, "max_iter": 80, "learning_rate": 0.10, "min_samples_leaf": 5},
        {"max_leaf_nodes": 15, "max_iter": 160, "learning_rate": 0.07, "min_samples_leaf": 8},
        {"max_leaf_nodes": 31, "max_iter": 130, "learning_rate": 0.10, "min_samples_leaf": 5},
        {"max_leaf_nodes": 15, "max_iter": 200, "learning_rate": 0.04, "min_samples_leaf": 12},
    ),
    "sanode": (
        {"train_steps": 24, "epochs": 72, "learning_rate": 0.003},
        {"train_steps": 40, "epochs": 72, "learning_rate": 0.003},
        {"train_steps": 24, "epochs": 72, "learning_rate": 0.008},
        {"train_steps": 40, "epochs": 72, "learning_rate": 0.008},
    ),
}

class KNearestClassifier:
    """kNN with its requested neighborhood capped by the available core rows."""

    def __init__(self, n_neighbors, weights):
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y):
        self.model_ = KNeighborsClassifier(
            n_neighbors=min(self.n_neighbors, len(X)), weights=self.weights,
        ).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class CentroidClassifier:
    """Euclidean nearest class mean with uniform priors."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centers_ = np.stack([X[y == label].mean(axis=0) for label in self.classes_])
        self.parameter_count_ = self.centers_.size
        return self

    def predict(self, X):
        indices = np.argmin(cdist(X, self.centers_, metric="sqeuclidean"), axis=1)
        return self.classes_[indices]


class PrototypeClassifier:
    """Nearest class-conditional k-means prototype with uniform class weights."""

    def __init__(self, count, seed):
        self.count = count
        self.seed = seed

    def fit(self, X, y):
        centers = []
        labels = []
        self.classes_ = np.unique(y)
        for label in self.classes_:
            members, multiplicities = np.unique(X[y == label], axis=0, return_counts=True)
            count = min(self.count, len(members))
            kmeans = KMeans(n_clusters=count, n_init=8, random_state=self.seed)
            centers.append(kmeans.fit(members, sample_weight=multiplicities).cluster_centers_)
            labels.extend([label] * count)
        self.centers_ = np.concatenate(centers, axis=0)
        self.center_labels_ = np.asarray(labels)
        self.parameter_count_ = self.centers_.size
        return self

    def predict(self, X):
        indices = np.argmin(cdist(X, self.centers_, metric="sqeuclidean"), axis=1)
        return self.center_labels_[indices]


class MahalanobisClassifier:
    """Nearest centroid under a regularized pooled within-class metric."""

    def __init__(self, shrinkage):
        self.shrinkage = shrinkage

    def fit(self, X, y):
        self.classes_, labels = np.unique(y, return_inverse=True)
        means = np.stack([X[labels == index].mean(axis=0) for index in range(len(self.classes_))])
        residuals = X - means[labels]
        dimension = X.shape[1]
        covariance = residuals.T @ residuals / max(len(X) - len(self.classes_), 1)
        ridge = max(np.trace(covariance) / dimension, 1e-8)
        covariance = (1 - self.shrinkage) * covariance + self.shrinkage * ridge * np.eye(dimension)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        self.transform_ = eigenvectors / np.sqrt(eigenvalues)[None, :]
        self.centers_ = means @ self.transform_
        self.parameter_count_ = means.size + dimension * (dimension + 1) // 2
        return self

    def predict(self, X):
        indices = np.argmin(cdist(X @ self.transform_, self.centers_, metric="sqeuclidean"), axis=1)
        return self.classes_[indices]


class RBFSVM:
    """RBF support-vector classification with core-only scale and balanced weights."""

    def __init__(self, C, gamma_scale):
        self.C = C
        self.gamma_scale = gamma_scale

    def fit(self, X, y):
        gamma = self.gamma_scale / max(X.shape[1] * X.var(), 1e-8)
        self.model_ = SVC(C=self.C, gamma=gamma, class_weight="balanced").fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class GaussianMixtureClassifier:
    """Class-conditional GMM likelihood classifier with uniform class priors."""

    def __init__(self, components, covariance_type, reg_covar, seed):
        self.components = components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.seed = seed

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.mixtures_ = []
        for label in self.classes_:
            members = X[y == label]
            components = min(self.components, len(np.unique(members, axis=0)))
            mixture = GaussianMixture(
                n_components=components, covariance_type=self.covariance_type,
                reg_covar=self.reg_covar, n_init=3, random_state=self.seed,
            ).fit(members)
            self.mixtures_.append(mixture)
        dimension = X.shape[1]
        covariance_parameters = dimension
        if self.covariance_type == "full":
            covariance_parameters = dimension * (dimension + 1) // 2
        self.parameter_count_ = sum(
            mixture.n_components * (dimension + covariance_parameters + 1) - 1
            for mixture in self.mixtures_
        )
        return self

    def predict(self, X):
        scores = np.column_stack([mixture.score_samples(X) for mixture in self.mixtures_])
        return self.classes_[np.argmax(scores, axis=1)]


def make_mlp(seed, hidden_layer_sizes, alpha=0.10):
    """Construct an MLP used by the standard and larger feedforward controls."""
    return MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layer_sizes), alpha=alpha, activation="tanh",
        solver="lbfgs", max_iter=350, max_fun=6500, tol=0.0001, random_state=seed,
    )


def capacity_mlp_width(n_features, n_classes, target_parameters):
    """Choose the closest one-hidden-layer size by weight and bias count."""
    output_dimension = 1 if n_classes == 2 else n_classes
    return max(1, round((target_parameters - output_dimension) / (n_features + output_dimension + 1)))


class SANode:
    """Feature-space adaptation of Marino et al. (arXiv:2311.10387v2).

    The trainable spectral coupling uses the paper's analytic cubic reaction,
    planted sign equilibria, Euler recurrence, and endpoint regression loss.
    Adam updates all non-planted spectral coordinates and the potential scale.
    """

    amplitude = 0.5
    onset_offset = 0.6
    train_step_size = 0.10
    evaluation_horizon = 12.0
    clip_state = 5.0
    distance_tolerance = 0.12
    coordinate_tolerance = 0.25
    residual_tolerance = 0.025

    def __init__(self, train_steps, epochs, learning_rate, seed, learn_encoder=False):
        self.train_steps = train_steps
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.learn_encoder = learn_encoder

    def _map_features(self, X, encoder=None):
        if encoder is not None:
            weights, bias = encoder
            X = X @ weights + bias
        return np.tanh(X + self.onset_offset)

    def _pad(self, mapped):
        return np.pad(mapped, ((0, 0), (0, self.state_dimension_ - mapped.shape[1])))

    def _initial_state(self, X):
        # The offset avoids imposing the field's antipodal symmetry on centered inputs.
        return self._pad(self.amplitude * self._map_features(X, self.encoder_))

    def _coupling(self, basis, eigenvalues, gamma, planted):
        inverse = np.linalg.inv(basis)
        limit = 8 * self.amplitude**2 * gamma - 0.01
        # The fitted eigenvalues absorb the paper's fixed factor beta.
        frequencies = np.concatenate((np.zeros(planted), np.clip(eigenvalues, -4.0, limit)))
        coupling = (basis * frequencies[None, :]) @ inverse
        return coupling, frequencies, inverse

    def _field(self, state, coupling, gamma):
        return 4 * gamma * state * (self.amplitude**2 - state**2) + state @ coupling.T

    def _loss_gradient(self, X, targets, basis, eigenvalues, gamma):
        planted = len(self.classes_)
        coupling, frequencies, inverse = self._coupling(basis, eigenvalues, gamma, planted)
        states = [X]
        clip_derivatives = []
        step = self.train_step_size
        for _ in range(self.train_steps):
            next_state = states[-1] + step * self._field(states[-1], coupling, gamma)
            clip_derivatives.append(np.abs(next_state) < self.clip_state)
            states.append(np.clip(next_state, -self.clip_state, self.clip_state))
        residual = states[-1] - targets
        loss = np.mean(residual**2)
        adjoint = (2 / residual.size) * residual
        grad_coupling = np.zeros_like(coupling)
        grad_gamma = 0.0
        for state, keep_gradient in zip(reversed(states[:-1]), reversed(clip_derivatives)):
            adjoint = adjoint * keep_gradient
            local = self.amplitude**2 - state**2
            grad_coupling += step * (adjoint.T @ state)
            grad_gamma += 4 * step * np.sum(adjoint * state * local)
            adjoint += step * (4 * gamma * (local - 2 * state**2) * adjoint + adjoint @ coupling)

        scaled_gradient = grad_coupling @ inverse.T
        basis_gradient = scaled_gradient * frequencies[None, :] - coupling.T @ scaled_gradient
        frequency_gradient = np.sum(basis * scaled_gradient, axis=0)[planted:]
        upper_limit = 8 * self.amplitude**2 * gamma - 0.01
        grad_gamma += 8 * self.amplitude**2 * np.sum(frequency_gradient[eigenvalues > upper_limit])
        frequency_gradient *= (eigenvalues >= -4.0) & (eigenvalues <= upper_limit)

        gram_error = basis.T @ basis - np.eye(len(basis))
        regularization = 0.01 / len(basis)
        loss += regularization * np.sum(gram_error**2)
        basis_gradient += 4 * regularization * (basis @ gram_error)
        return loss, (basis_gradient[:, planted:], frequency_gradient, np.array([grad_gamma])), adjoint

    def _joint_loss_gradient(self, X, targets, basis, parameters):
        eigenvalues, gamma, weights, bias = parameters
        mapped = self._map_features(X, (weights, bias))
        initial = self._pad(self.amplitude * mapped)
        loss, gradients, initial_gradient = self._loss_gradient(initial, targets, basis, eigenvalues, gamma[0])
        transformed_gradient = initial_gradient[:, :X.shape[1]] * self.amplitude * (1 - mapped**2)
        weight_error = weights - np.eye(len(weights))
        regularization = 0.001 / len(weights)
        loss += regularization * (np.sum(weight_error**2) + np.sum(bias**2))
        weight_gradient = X.T @ transformed_gradient + 2 * regularization * weight_error
        bias_gradient = transformed_gradient.sum(axis=0) + 2 * regularization * bias
        return loss, (*gradients, weight_gradient, bias_gradient)

    def fit(self, X, y):
        self.classes_, target_indices = np.unique(y, return_inverse=True)
        planted = len(self.classes_)
        smallest_dimension = max(8, X.shape[1], planted + 2)
        self.state_dimension_ = 2 ** (int(smallest_dimension) - 1).bit_length()
        rng = np.random.default_rng(self.seed)
        self.encoder_ = None
        initial = self._initial_state(X)

        # Choose an orthogonal sign codebook with a core-only assignment to classes.
        codebook = self.amplitude * hadamard(self.state_dimension_)[:, rng.permutation(self.state_dimension_)]
        codebook *= rng.choice([-1, 1], size=self.state_dimension_)[None, :]
        codebook = codebook[rng.permutation(self.state_dimension_)]
        means = np.stack([initial[target_indices == label].mean(axis=0) for label in range(planted)])
        overlap = means @ codebook.T
        rows, columns = linear_sum_assignment(np.abs(overlap), maximize=True)
        signs = np.where(overlap[rows, columns] < 0, -1, 1)
        self.anchors_ = codebook[columns] * signs[:, None]
        targets = self.anchors_[target_indices]

        class_axes = self.anchors_.T / (self.amplitude * np.sqrt(self.state_dimension_))
        complement = np.linalg.qr(class_axes, mode="complete")[0][:, planted:]
        remainder = complement.shape[1]
        rotation = np.linalg.qr(rng.normal(size=(remainder, remainder)))[0]
        parameters = [
            complement @ rotation,
            rng.normal(-5, 1, size=remainder) / np.sqrt(self.state_dimension_),
            np.array([0.4]),
        ]
        if self.learn_encoder:
            parameters.extend((np.eye(X.shape[1]), np.zeros(X.shape[1])))
        first_moments = [np.zeros_like(parameter) for parameter in parameters]
        second_moments = [np.zeros_like(parameter) for parameter in parameters]
        self.rejected_updates_ = 0
        self.training_field_evaluations_ = 0
        self.training_loss_ = []
        iteration = 0
        for _ in range(self.epochs):
            order = rng.permutation(len(X))
            batch_losses = []
            for start in range(0, len(X), 128):
                sample = order[start:start + 128]
                basis = np.column_stack((class_axes, parameters[0]))
                if self.learn_encoder:
                    loss, gradients = self._joint_loss_gradient(X[sample], targets[sample], basis, parameters[1:])
                else:
                    loss, gradients, _ = self._loss_gradient(
                        initial[sample], targets[sample], basis, parameters[1], parameters[2][0]
                    )
                self.training_field_evaluations_ += len(sample) * self.train_steps
                batch_losses.append(loss * len(sample))
                iteration += 1
                updates = []
                for idx, (parameter, gradient) in enumerate(zip(parameters, gradients)):
                    first_moments[idx] = 0.9 * first_moments[idx] + 0.1 * gradient
                    second_moments[idx] = 0.999 * second_moments[idx] + 0.001 * gradient**2
                    mean = first_moments[idx] / (1 - 0.9**iteration)
                    variance = second_moments[idx] / (1 - 0.999**iteration)
                    updates.append(parameter - self.learning_rate * mean / (np.sqrt(variance) + 1e-8))
                updates[1] = np.clip(updates[1], -4.0, 2.4)
                updates[2] = np.clip(updates[2], 0.25, 1.20)
                new_basis = np.column_stack((class_axes, updates[0]))
                if np.linalg.cond(new_basis) > 25:
                    updates[0] = parameters[0]
                    self.rejected_updates_ += 1
                parameters = updates
            self.training_loss_.append(sum(batch_losses) / len(X))

        self.gamma_ = float(parameters[2][0])
        self.basis_ = np.column_stack((class_axes, parameters[0]))
        self.coupling_, self.frequencies_, _ = self._coupling(self.basis_, parameters[1], self.gamma_, planted)
        self.encoder_ = tuple(parameters[3:]) if self.learn_encoder else None
        self.parameter_count_ = sum(parameter.size for parameter in parameters)
        self.attractor_residual_ = float(np.max(np.abs(self._field(self.anchors_, self.coupling_, self.gamma_))))
        self.jacobian_max_real_ = float(self.frequencies_.max() - 8 * self.amplitude**2 * self.gamma_)
        return self

    def predict_details(self, X, step_scale=1.0):
        """Run a finite-horizon destination check with a nearest-anchor fallback."""
        if not np.isfinite(step_scale) or step_scale <= 0:
            raise ValueError("step_scale must be positive and finite")
        count = max(1, int(np.ceil(self.evaluation_horizon / (self.train_step_size * step_scale))))
        step = self.evaluation_horizon / count
        state = self._initial_state(X)
        guarded = np.zeros(len(X), dtype=bool)
        for _ in range(count):
            state += step * self._field(state, self.coupling_, self.gamma_)
            guarded |= np.max(np.abs(state), axis=1) >= self.clip_state
            np.clip(state, -self.clip_state, self.clip_state, out=state)
        indices = np.argmin(cdist(state, self.anchors_, metric="sqeuclidean"), axis=1)
        error = state - self.anchors_[indices]
        distance = np.sqrt(np.mean(error**2, axis=1))
        residual = np.sqrt(np.mean(self._field(state, self.coupling_, self.gamma_)**2, axis=1))
        converged = (
            (~guarded) & (distance <= self.distance_tolerance)
            & (np.max(np.abs(error), axis=1) <= self.coordinate_tolerance)
            & (residual <= self.residual_tolerance)
        )
        fallback = self.classes_[indices]
        return {
            "pred": np.where(converged, fallback, -1),
            "fallback": fallback,
            "converged": converged,
            "nfe": np.full(len(X), count + 1, dtype=np.int64),
            "distance": distance,
            "residual": residual,
            "guarded": guarded,
            "exact_sign": np.all(np.sign(state) == np.sign(self.anchors_[indices]), axis=1),
        }

    def predict(self, X):
        return self.predict_details(X)["pred"]

    def zero_trajectory(self):
        """Read the same anchors immediately after the fitted input map."""
        return SANodeInitialReadout(self)


class SANodeInitialReadout:
    def __init__(self, model):
        self.model = model
        self.parameter_count_ = model.parameter_count_

    def predict(self, X):
        initial = self.model._initial_state(X)
        indices = np.argmin(cdist(initial, self.model.anchors_, metric="sqeuclidean"), axis=1)
        return self.model.classes_[indices]


def make_control(name, params, seed=0):
    """Build an unfitted estimator for one frozen candidate configuration."""
    if name == "knn":
        return KNearestClassifier(**params)
    if name == "centroid":
        return CentroidClassifier(**params)
    if name == "prototypes":
        return PrototypeClassifier(**params, seed=seed)
    if name == "mahalanobis":
        return MahalanobisClassifier(**params)
    if name == "rbf_svm":
        return RBFSVM(**params)
    if name == "gmm":
        return GaussianMixtureClassifier(**params, seed=seed)
    if name in {"mlp", "mlp_wide"}:
        return make_mlp(**params, seed=seed)
    if name == "boosted_trees":
        return HistGradientBoostingClassifier(**params, class_weight="balanced", early_stopping=False, random_state=seed)
    if name in {"sanode", "sanode_joint"}:
        return SANode(**params, seed=seed, learn_encoder=name == "sanode_joint")
    raise ValueError(f"Unknown control model: {name}")
