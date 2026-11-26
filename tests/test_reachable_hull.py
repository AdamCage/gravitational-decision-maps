import unittest

import numpy as np

from crossing.affine_reserve import AffineReadout
from crossing.reachable_hull import ReachableHullGuard


class ReachableHullTests(unittest.TestCase):
    def test_single_site_exact_path_threshold_and_tie(self):
        head = AffineReadout([[0.0], [1.0]], [0.0, 0.0], 0.25)
        guard = ReachableHullGuard(head, [[-1.0]], [1.0], 1.2)
        start = np.array([[1.0], [2.0], [0.0]])
        for t in [0.4, 0.5, 0.8]:
            expected = np.r_[(-1 + (start[:2, 0] + 1) * np.exp(-t)) > 0.25 * start[:2, 0], False]
            verdict = guard.certify(start, t)
            np.testing.assert_array_equal(verdict.strict, [True, True, False])
            np.testing.assert_array_equal(verdict.slab, expected)
            np.testing.assert_array_equal(verdict.hull, expected)

    def test_posterior_tightens_global_slab_on_imbalanced_mixture(self):
        head = AffineReadout([[0.0], [1.0]], [0.0, 0.0], 0.25)
        guard = ReachableHullGuard(head, [[-4.0], [1.0]], [0.01, 1.0], 0.1)
        verdict = guard.certify([[1.0]], 0.3)
        np.testing.assert_array_equal(verdict.slab, [False])
        np.testing.assert_array_equal(verdict.hull, [True])


if __name__ == "__main__":
    unittest.main()
