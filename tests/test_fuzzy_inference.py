import unittest

import numpy as np

from src.fuzzy.fuzzy_inference import (
    MamdaniReliabilityFIS,
    mean_pairwise_total_variation,
)
from src.fuzzy.fuzzy_rules import build_rule_base


class FuzzyInferenceTests(unittest.TestCase):
    def setUp(self):
        self.fis = MamdaniReliabilityFIS(universe_resolution=501)

    def test_rule_base_exhaustively_covers_four_inputs(self):
        self.assertEqual(len(build_rule_base()), 81)

    def test_total_variation_disagreement_bounds(self):
        identical = [[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]]
        maximally_split = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        self.assertAlmostEqual(mean_pairwise_total_variation(identical), 0.0)
        self.assertAlmostEqual(
            mean_pairwise_total_variation(maximally_split), 2.0 / 3.0
        )

    def test_weights_are_normalized_and_deterministic(self):
        probabilities = np.array(
            [[0.93, 0.07], [0.54, 0.46], [0.89, 0.11]], dtype=float
        )
        first = self.fis.weights_from_probabilities(probabilities)
        second = self.fis.weights_from_probabilities(probabilities)
        np.testing.assert_allclose(first, second)
        self.assertAlmostEqual(float(first.sum()), 1.0)
        self.assertTrue(np.all(first >= 0.0))
        self.assertGreater(first[0], first[1])
        self.assertGreater(first[2], first[1])

    def test_symmetric_inputs_produce_symmetric_weights(self):
        weights = self.fis.weights_from_probabilities(
            [[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]]
        )
        np.testing.assert_allclose(weights, np.full(3, 1.0 / 3.0), atol=1e-8)

    def test_invalid_probabilities_are_rejected(self):
        with self.assertRaises(ValueError):
            self.fis.weights_from_probabilities(
                [[0.8, 0.3], [0.6, 0.4], [0.7, 0.3]]
            )


if __name__ == "__main__":
    unittest.main()
