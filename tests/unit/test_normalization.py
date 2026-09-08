import unittest
import numpy as np
from satellite_srm.preprocessing.normalization import RadiometricNormalizer

class TestNormalization(unittest.TestCase):
    def test_scale_factor_normalization(self):
        norm = RadiometricNormalizer(scale_factor=10000.0)
        raw = np.array([[[0.0, 5000.0, 10000.0]]], dtype=np.float32)
        res = norm.normalize(raw)
        self.assertAlmostEqual(res[0, 0, 0], 0.0)
        self.assertAlmostEqual(res[0, 0, 1], 0.5)
        self.assertAlmostEqual(res[0, 0, 2], 1.0)

    def test_reversibility(self):
        norm = RadiometricNormalizer()
        raw = np.random.uniform(500, 9500, (4, 16, 16)).astype(np.float32)
        n = norm.normalize(raw)
        recovered = norm.unnormalize(n, to_dn=True)
        # Verify within roundoff error
        np.testing.assert_allclose(raw, recovered, atol=2.0)

if __name__ == "__main__":
    unittest.main()
