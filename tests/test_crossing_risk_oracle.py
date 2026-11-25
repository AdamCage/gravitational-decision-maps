import json
import unittest

from crossing.risk_oracle import HORIZONS, oracle_experiments


class GaussianRiskOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = oracle_experiments()

    def test_complete_rosters_and_exact_symmetric_null(self):
        cases = self.report["fixtures"]
        self.assertEqual([len(case["boundaries"]) for case in cases], [1, 1, 3, 5])
        self.assertEqual([sum(len(pair["fine"]) for pair in case["pairwise_roots"]) for case in cases], [1, 1, 3, 9])
        for case in cases:
            for pair in case["pairwise_roots"]:
                self.assertEqual(len(pair["sign_change_witnesses"]), pair["sign_variation_bound"])
                for witness in pair["sign_change_witnesses"]:
                    a, b = witness["score_differences_b_minus_a"]
                    self.assertLess(a * b, 0)
        self.assertEqual(cases[0]["predicted_coefficient"], 0.0)
        self.assertTrue(all(row["excess_quadrature"] == 0.0 for row in cases[0]["observations"]))

    def test_local_quadrature_agrees_with_global_cdf_and_small_time_coefficient(self):
        for case in self.report["fixtures"][1:]:
            coefficient = case["predicted_coefficient"]
            observations = case["observations"]
            self.assertGreater(coefficient, 0.0)
            self.assertEqual([row["horizon"] for row in observations], list(HORIZONS))
            for row in observations:
                self.assertGreater(row["excess_quadrature"], 0)
                self.assertAlmostEqual(row["excess_cdf"], row["excess_quadrature"], delta=2e-13)
                self.assertLess(row["round_trip_max_error"], 2e-11)
            ratio = observations[0]["ratio_to_horizon_squared"]
            self.assertLess(abs(ratio / coefficient - 1), 0.0035)
        json.dumps(self.report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
