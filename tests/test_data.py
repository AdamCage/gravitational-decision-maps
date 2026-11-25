import unittest
from unittest.mock import patch

import numpy as np

from gdm.data import DEVELOPMENT_SEEDS, make_task


class BenchmarkTests(unittest.TestCase):
    def test_paired_world_differs_only_in_external_labels(self):
        for seed in DEVELOPMENT_SEEDS:
            a = make_task("paired_a", seed, split="development")
            b = make_task("paired_b", seed, split="development")
            for attr in ("x_train", "y_train", "x_core", "y_core", "x_external"):
                np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))
            np.testing.assert_array_equal(a.y_external, 1 - b.y_external)
            self.assertEqual(a.metadata["external_ids"], b.metadata["external_ids"])

    def test_digits_development_does_not_load_writer_test_set(self):
        with patch("gdm.data._load_packaged_data", side_effect=AssertionError("opens held-out Digits")):
            benchmark = make_task("digits_narrow", DEVELOPMENT_SEEDS[0], split="development")
        self.assertEqual(len(np.unique(benchmark.y_external)), 10)

    def test_pub_coverages_retain_nested_cores_and_same_evaluation_entities(self):
        for family in ("digits", "wine"):
            small = make_task(f"{family}_narrow", DEVELOPMENT_SEEDS[0], split="development").metadata
            large = make_task(f"{family}_wide", DEVELOPMENT_SEEDS[0], split="development").metadata
            self.assertEqual(small["anchor_ids"], large["anchor_ids"])
            self.assertTrue(set(small["train_ids"]) < set(large["train_ids"]))
            self.assertTrue(set(large["external_ids"]) < set(small["external_ids"]))
            self.assertEqual(
                set(small["core_ids"]) | set(small["external_ids"]),
                set(large["core_ids"]) | set(large["external_ids"]),
            )


if __name__ == "__main__":
    unittest.main()
