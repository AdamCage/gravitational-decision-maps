import unittest

import numpy as np

from crossing.affine_reserve import AffineReadout, GaussianReservePath


class AffineReserveTests(unittest.TestCase):
    def test_straight_segment_reserve_and_ties(self):
        head = AffineReadout([[0.0], [1.0]], [0.0, 0.0], 0.25)
        verdict = head.segment([[1.0], [1.0], [1.0], [0.0], [-1.0]], [[0.3], [0.25], [-1.0], [1.0], [-0.3]])
        np.testing.assert_array_equal(verdict, [True, False, False, False, True])

    def test_single_site_exact_margin_decay_threshold(self):
        head = AffineReadout([[0.0], [1.0]], [0.0, 0.0], 0.25)
        path = GaussianReservePath(head, [[-1.0]], [1.0], 1.2)
        state = path.prepare([[1.0], [2.0]])
        for horizon in [0.4, 0.5, 0.8]:
            exact = -1 + np.array([2.0, 3.0]) * np.exp(-horizon) > [0.25, 0.5]
            local, global_ = path.certify(state, horizon)
            np.testing.assert_array_equal(local, exact)
            np.testing.assert_array_equal(global_, exact)

    def test_constant_rival_margin_is_exempt(self):
        head = AffineReadout([[1.0], [1.0], [1.0]], [1.0, 0.0, -1.0], 0.25)
        path = GaussianReservePath(head, [[-2.0], [1.0]], [0.2, 0.8], 0.5)
        query = [[-1.0], [2.0]]
        self.assertTrue(head.segment(query, [[10.0], [-20.0]]).all())
        local, global_ = path.certify(path.prepare(query), 5.0)
        self.assertTrue(local.all() and global_.all())


if __name__ == "__main__":
    unittest.main()
