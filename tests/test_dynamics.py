import unittest

import numpy as np

from gdm.dynamics import AttractionClassifier


class AttractionClassifierTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array(
            [[-1.12, -0.09], [-1.0, 0.0], [-0.88, 0.09], [0.88, -0.09], [1.0, 0.0], [1.12, 0.09]]
        )
        self.y = np.repeat([0, 1], 3)

    def fit_symmetric(self, **options):
        return AttractionClassifier(n_prototypes=1, anisotropic=False, learned=False, **options).fit(
            self.X, self.y
        )

    def test_constrained_force_matches_potential_and_pins_centers(self):
        model = AttractionClassifier(n_prototypes=1, bandwidth=2.5).fit(self.X, self.y)
        # Include local quadratic, transition annulus, outside, and symmetry plane.
        points = np.array([[-1.01, 0.02], [-0.80, 0.08], [-0.68, 0.13], [0.0, 0.01], [1.7, 0.8]])
        h = 1e-6
        gradient = np.column_stack(
            [(model.potential(points + h * v) - model.potential(points - h * v)) / (2 * h) for v in np.eye(2)]
        )
        np.testing.assert_allclose(model.field(points), -gradient, atol=1e-8)
        np.testing.assert_allclose(model.field(model.equilibria_), 0, atol=1e-10)
        self.assertTrue(np.all(model.equilibrium_curvature_ > 0))

    def test_relaxed_roots_have_low_residual_and_positive_curvature(self):
        model = self.fit_symmetric(constrained=False)
        self.assertEqual(len(model.equilibria_), 2)
        np.testing.assert_allclose(model.field(model.equilibria_), 0, atol=1e-7)
        self.assertTrue(np.all(model.equilibrium_curvature_ > 0))

    def test_symmetric_gradient_flow_and_static_resolved_labels_agree(self):
        model = self.fit_symmetric()
        rng = np.random.default_rng(128)
        points = rng.uniform([-2, -1], [2, 1], size=(400, 2))
        static = model.with_options(readout="static", steps=0)
        trajectory = model.predict_details(points)
        self.assertIs(static.centers_, model.centers_)
        self.assertTrue(trajectory["resolved"].all())
        np.testing.assert_array_equal(static.predict(points), trajectory["pred"])
        refined = model.with_options(step_size=0.075, steps=160)
        np.testing.assert_array_equal(trajectory["pred"], refined.predict(points))

    def test_static_step_ablation_counts_each_force_evaluation(self):
        model = self.fit_symmetric()
        points = np.array([[-2, 0.0], [2, 0.0]])
        for steps in [0, 1, 9]:
            result = model.with_options(readout="static", steps=steps).predict_details(points)
            np.testing.assert_array_equal(result["nfe"], np.full(len(points), steps))

    def test_unresolved_destination_uses_original_static_fallback(self):
        model = self.fit_symmetric(steps=0)
        points = np.array([[-2, 0.0], [2, 0.0]])
        result = model.predict_details(points)
        np.testing.assert_array_equal(result["pred"], [-1, -1])
        np.testing.assert_array_equal(result["fallback"], [0, 1])
        np.testing.assert_array_equal(result["nfe"], [1, 1])

    def test_core_intersection_detects_first_crossing_between_solver_positions(self):
        model = self.fit_symmetric(readout="absorbing")
        starts = np.array([[-3.0, 0.0], [3.0, 0.0], [-3.0, 1.0]])
        changes = np.array([[6.0, 0.0], [-6.0, 0.0], [6.0, 0.0]])
        np.testing.assert_array_equal(model._first_core_hit(starts, changes), [0, 1, -1])
        result = model.predict_details(model.centers_)
        np.testing.assert_array_equal(result["pred"], [0, 1])
        np.testing.assert_array_equal(result["nfe"], [0, 0])


if __name__ == "__main__":
    unittest.main()
