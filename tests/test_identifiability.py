import unittest
import numpy as np

from nad.identifiability import (
    orthogonal_novelty_fraction,
    residualized_disagreement,
)


class IdentifiabilityTests(unittest.TestCase):
    def test_aliased_candidate_is_zero_novelty(self):
        J = np.array([[1.0], [0.0], [0.0]])
        g = np.array([3.0, 0.0, 0.0])
        r = orthogonal_novelty_fraction(J, g)
        self.assertLess(r["novelty_fraction"], 1e-12)

    def test_orthogonal_candidate_is_full_novelty(self):
        J = np.array([[1.0], [0.0], [0.0]])
        g = np.array([0.0, 2.0, 0.0])
        r = orthogonal_novelty_fraction(J, g)
        self.assertGreater(r["novelty_fraction"], 1.0 - 1e-12)

    def test_disagreement_can_be_large_but_unidentifiable(self):
        J = np.array([[1.0], [0.0]])
        a = np.array([100.0, 0.0])
        b = np.array([0.0, 0.0])
        r = residualized_disagreement(J, a, b)
        self.assertGreater(r["raw_disagreement_norm"], 10.0)
        self.assertLess(r["residualized_disagreement_norm"], 1e-10)


if __name__ == "__main__":
    unittest.main()
