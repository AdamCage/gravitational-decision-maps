import unittest

import numpy as np

from gdm.controls import CONTROL_GRIDS, SANode, make_control


class StaticControlTests(unittest.TestCase):
    def test_small_core_neighborhood_and_constant_feature(self):
        X = np.column_stack((np.arange(12), np.zeros(12)))
        y = np.repeat([0, 1], 6)
        knn = make_control("knn", CONTROL_GRIDS["knn"][-1]).fit(X, y)
        self.assertEqual(knn.model_.n_neighbors, 12)
        np.testing.assert_array_equal(knn.predict(X), y)
        centroid = make_control("centroid", {}).fit(X, y)
        np.testing.assert_array_equal(centroid.predict(X), y)


class SANodeTests(unittest.TestCase):
    def test_spectral_and_encoder_gradients_match_finite_differences(self):
        rng = np.random.default_rng(2311)
        model = SANode(8, 2, 0.01, 17, learn_encoder=True)
        model.classes_ = np.arange(2)
        model.state_dimension_ = 4
        inputs = rng.normal(scale=0.4, size=(5, 3))
        targets = rng.choice([-0.5, 0.5], size=(5, 4))
        basis = np.linalg.qr(rng.normal(size=(4, 4)))[0] + rng.normal(scale=0.05, size=(4, 4))
        parameters = [np.array([-1.0, 0.81]), np.array([0.4]), np.eye(3), np.zeros(3)]
        _, gradients = model._joint_loss_gradient(inputs, targets, basis, parameters)
        for values, gradient in zip((basis[:, 2:], *parameters), gradients):
            for index in np.ndindex(values.shape):
                value = values[index]
                values[index] = value + 1e-6
                plus = model._joint_loss_gradient(inputs, targets, basis, parameters)[0]
                values[index] = value - 1e-6
                minus = model._joint_loss_gradient(inputs, targets, basis, parameters)[0]
                values[index] = value
                self.assertAlmostEqual(gradient[index], (plus - minus) / 2e-6, places=7)

    def test_frozen_and_joint_share_anchors_and_count_solver_work(self):
        rng = np.random.default_rng(19)
        X = rng.normal(size=(30, 3))
        y = np.asarray(X[:, 0] > 0, dtype=np.int64)
        params = {"train_steps": 8, "epochs": 3, "learning_rate": 0.01}
        frozen = make_control("sanode", params, 11).fit(X, y)
        joint = make_control("sanode_joint", params, 11).fit(X, y)
        np.testing.assert_array_equal(frozen.anchors_, joint.anchors_)
        self.assertEqual(joint.parameter_count_, frozen.parameter_count_ + X.shape[1] * (X.shape[1] + 1))
        for model in (frozen, joint):
            self.assertLess(model.attractor_residual_, 1e-12)
            self.assertLess(model.jacobian_max_real_, 0)
            details = model.predict_details(X, step_scale=0.5)
            np.testing.assert_array_equal(details["nfe"], np.full(len(X), 241))
            np.testing.assert_array_equal(details["pred"] >= 0, details["converged"])
            expected_indices = np.argmax(model._initial_state(X) @ model.anchors_.T, axis=1)
            np.testing.assert_array_equal(model.zero_trajectory().predict(X), model.classes_[expected_indices])


if __name__ == "__main__":
    unittest.main()
