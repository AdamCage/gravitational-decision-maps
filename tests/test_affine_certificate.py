import unittest
from decimal import Context, Decimal

import numpy as np

from crossing.affine_certificate import Bounds, VoronoiCertificate, negative_exp
from tools.flat_boundary_interval import FIXTURES, Flow, point, STEP


class AffineCertificateTests(unittest.TestCase):
    def test_exponential_encloses_high_precision_values(self):
        x = np.array([-1000, -70, -64, -2, -0.125, -1e-13, -np.finfo(float).tiny, 0])
        values = negative_exp(Bounds.exact(x))
        reference = Context(prec=90)
        for arg, lower, upper in zip(x, values.lo, values.hi, strict=True):
            enclosed = reference.exp(Decimal.from_float(float(arg)))
            self.assertLessEqual(Decimal.from_float(float(lower)), enclosed)
            self.assertLessEqual(enclosed, Decimal.from_float(float(upper)))

    def test_reflection_preserves_and_exact_tie_abstains(self):
        certificate = VoronoiCertificate([[-1.0], [1.0]], 1.2)
        initial, stable, switch, slab = certificate.classify([[-0.1], [0.0], [0.1]], 1.25)
        np.testing.assert_array_equal(initial, [0, -1, 1])
        np.testing.assert_array_equal(stable, [True, False, True])
        np.testing.assert_array_equal(slab, stable)
        self.assertFalse(switch.any())

    def test_switch_against_interval_integration_of_same_encoded_inputs(self):
        original = FIXTURES["centered_cospherical"]
        sites = np.array(original["sites"], dtype=float)
        queries = np.array(original["queries"], dtype=float)
        beta = float(original["beta"])
        certificate = VoronoiCertificate(sites, beta)
        encoded = {"sites": [[Decimal.from_float(float(value)) for value in row] for row in sites],
                   "beta": Decimal.from_float(beta)}
        integrator = Flow(encoded)
        duration = 0.125
        initial, stable, switch, _ = certificate.classify(queries, duration)
        self.assertTrue(switch.any())
        for index in np.flatnonzero(switch | stable):
            box = tuple(point(Decimal.from_float(float(value))) for value in queries[index])
            self.assertEqual(integrator.winner(box), initial[index])
            for _ in range(int(Decimal.from_float(duration) / STEP)):
                _, box = integrator.step(box)
                if stable[index]:
                    self.assertIn(integrator.winner(box), (None, initial[index]))
            if switch[index]:
                rival = integrator.winner(box)
                self.assertIsNotNone(rival)
                self.assertNotEqual(rival, initial[index])


if __name__ == "__main__":
    unittest.main()
