import unittest
from unittest.mock import patch

import numpy as np

from crossing.data import FAMILIES, GAPS, make_case
from crossing import public


class CrossingDataTests(unittest.TestCase):
    def test_gap_pairs_keep_exact_evaluation_arrays_and_move_the_core(self):
        for family in FAMILIES:
            cases = [make_case(family, gap, 900001) for gap in GAPS]
            small = cases[0]
            for case in cases:
                np.testing.assert_array_equal(case.x_test, small.x_test)
                np.testing.assert_array_equal(case.y_test, small.y_test)
                np.testing.assert_array_equal(case.y_train, small.y_train)
                self.assertEqual(case.metadata["test_ids"], small.metadata["test_ids"])
                self.assertEqual(case.metadata["train_ids"], small.metadata["train_ids"])
                self.assertGreater(case.metadata["realized_train_absolute_bounds"][0],
                                   case.metadata["realized_test_absolute_bounds"][1])
                np.testing.assert_array_equal(np.bincount(case.y_train), [64, 64, 64])
                np.testing.assert_array_equal(np.bincount(case.y_test), [120, 120, 120])
                self.assertEqual(case.x_test.shape[1], 6 if family == "open_chains" else 2)
                self.assertTrue(np.isfinite(case.x_train).all() and np.isfinite(case.x_test).all())
            for nearer, further in zip(cases[:-1], cases[1:], strict=True):
                self.assertGreater(further.metadata["nominal_gap_to_test_support"],
                                   nearer.metadata["nominal_gap_to_test_support"])
                self.assertFalse(np.array_equal(nearer.x_train, further.x_train))

    def test_independent_seeds_and_curation_have_disjoint_identifiers(self):
        for family in FAMILIES:
            a = make_case(family, GAPS[0], 900001)
            b = make_case(family, GAPS[0], 900002)
            self.assertTrue(set(a.metadata["test_ids"]).isdisjoint(b.metadata["test_ids"]))
            self.assertTrue(set(a.metadata["train_ids"]).isdisjoint(a.metadata["test_ids"]))
            self.assertFalse(np.array_equal(a.x_test, b.x_test))
            self.assertFalse(a.y_train.flags.writeable)
            self.assertFalse(a.y_test.flags.writeable)

    def test_public_development_has_class_balance_and_never_opens_writer_test(self):
        original_load = public._load

        def permitted_load(name):
            self.assertNotEqual(name, "pendigits.tes")
            return original_load(name)

        with patch.object(public, "_load", side_effect=permitted_load):
            for family, dimensions, classes in (("pendigits", 16, 10), ("seeds", 7, 3)):
                case = public.make_public(family, 900001, split="development")
                self.assertEqual(case.x_train.shape[1], dimensions)
                self.assertEqual(len(np.unique(case.y_train)), classes)
                self.assertEqual(len(np.unique(case.y_test)), classes)
                self.assertEqual(len(set(np.bincount(case.y_test))), 1)
                self.assertTrue(set(case.metadata["train_ids"]).isdisjoint(case.metadata["test_ids"]))
                self.assertTrue(set(case.metadata["anchor_ids"].values()) <= set(case.metadata["train_ids"]))


if __name__ == "__main__":
    unittest.main()
