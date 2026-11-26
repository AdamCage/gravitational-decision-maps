import unittest

import numpy as np

from crossing.flow import MixtureFlow
from relativistic.flow import inertial_velocity, rk4_inertial


class _ConstantField:
    origin = np.array([0.0, 0.0])
    force = np.array([0.0, 4.0])

    def _rel_field(self, position):
        return np.broadcast_to(self.force, position.shape)


class RelativisticFlowTests(unittest.TestCase):
    def test_velocity_maps_share_a_cap_and_the_classical_limit(self):
        momentum = np.array([[0, 0], [0.3, 0.4], [3, 4], [3000, 4000]])
        limit = 2.0
        relativistic = inertial_velocity(momentum, "relativistic", limit)
        clipped = inertial_velocity(momentum, "clipped_inertial", limit)
        np.testing.assert_array_equal(inertial_velocity(momentum, "inertial"), momentum)
        np.testing.assert_allclose(relativistic[2], momentum[2] * limit / np.sqrt(29))
        np.testing.assert_allclose(clipped, [[0, 0], [0.3, 0.4], [1.2, 1.6], [1.2, 1.6]])
        self.assertTrue(np.all(np.linalg.norm(relativistic, axis=1) < limit))
        np.testing.assert_allclose(inertial_velocity(momentum, "relativistic", 1e9), momentum)

    def test_classical_damping_matches_constant_force_solution(self):
        model = _ConstantField()
        x = np.array([[1.0, -2.0], [0.0, 3.0]])
        time, damping = 0.625, 1.4
        endpoint, nfe = rk4_inertial(model, x, time, 0.01, mode="inertial", damping=damping)
        expected = x + model.force * (time + np.expm1(-damping * time) / damping) / damping
        np.testing.assert_allclose(endpoint, expected, rtol=0, atol=2e-9)
        self.assertEqual(nfe, 252)

    def test_capped_trajectories_match_undamped_constant_force_solutions(self):
        model = _ConstantField()
        x = np.zeros((1, 2))
        limit, time, force = 2.0, 1.0, 4.0
        expected = {
            "clipped_inertial": limit * time - limit**2 / (2 * force),
            "relativistic": (np.hypot(limit, force * time) - limit) * limit / force,
        }
        for mode, displacement in expected.items():
            with self.subTest(mode=mode):
                endpoint, nfe = rk4_inertial(model, x, time, 1 / 64, mode=mode,
                                             damping=0, speed_limit=limit)
                np.testing.assert_allclose(endpoint, [[0.0, displacement]], rtol=0, atol=1e-8)
                self.assertEqual(nfe, 256)

    def test_capped_rk4_endpoints_obey_travel_bound_on_mixture_field(self):
        model = MixtureFlow([[-1.0, 0.0], [1.0, 0.0]], [0, 1], [0.5, 0.5], 1.0)
        x = np.array([[-2.0, 8.0], [3.0, 4.0], [0.0, 0.0]])
        time, limit = 1.25, 0.25
        for mode in ["clipped_inertial", "relativistic"]:
            with self.subTest(mode=mode):
                endpoint, nfe = rk4_inertial(model, x, time, 0.1, mode=mode,
                                             damping=1.4, speed_limit=limit)
                self.assertTrue(np.all(np.linalg.norm(endpoint - x, axis=1) <= limit * time + 1e-12))
                np.testing.assert_array_equal(endpoint[-1], x[-1])
                self.assertEqual(nfe, 52)


if __name__ == "__main__":
    unittest.main()
