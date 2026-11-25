import unittest

import numpy as np

from crossing.flow import MixtureFlow, reference_path, rk4, triage


class CrossingFlowTests(unittest.TestCase):
    def test_exponential_and_certified_labels_are_translation_consistent(self):
        x = np.array([[-2.0], [-1.0], [-0.25], [0.25], [1.0], [2.0]])
        base = MixtureFlow([[-1.0], [1.0]], [0, 1], [0.5, 0.5], 1.0)
        shifted = MixtureFlow([[1e12 - 1.0], [1e12 + 1.0]], [0, 1], [0.5, 0.5], 1.0)
        np.testing.assert_array_equal(base.predict(x), shifted.predict(x + 1e12))
        np.testing.assert_allclose(base.field(x), shifted.field(x + 1e12), rtol=0, atol=0)
        np.testing.assert_array_equal(base.certify_endpoint(x, 0.5), shifted.certify_endpoint(x + 1e12, 0.5))

    def test_endpoint_certificate_does_not_rule_out_a_recrossing(self):
        b = 2 * np.sqrt(2) * np.exp(-0.05) * np.cosh(0.005)
        model = MixtureFlow([[-1], [1], [0]], [0, 0, 1], [1.0, 2.0, b], 10)
        x = np.array([[-5 * np.log(2) - 0.1]])
        path, _ = reference_path(model, x, np.linspace(0.005, 0.15, 40))
        labels = np.array([model.predict(z)[0] for z in path])
        self.assertTrue(np.any(labels == 1))
        self.assertEqual(labels[-1], 0)
        self.assertEqual(model.certify_endpoint(x, 0.15)[0], 0)
        self.assertFalse(model.certify(x, 0.15).stable[0])

    def test_runtime_paths_do_not_require_evaluator_labels(self):
        x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        model = MixtureFlow([[-1.0], [1.0]], [0, 1], [0.5, 0.5], 1.0)
        certified = model.certify(x, 0.15)
        target, calls = rk4(model, x, 0.15, 0.1)
        heuristic, triage_calls = triage(model, x, 0.15, 0.1, 0.1)
        np.testing.assert_array_equal(model.predict(target), heuristic)
        self.assertTrue(certified.stable.all())
        self.assertLess(triage_calls, len(x) * calls)


if __name__ == "__main__":
    unittest.main()
