import unittest
import numpy as np
from satellite_srm.compat import torch
from satellite_srm.models.multispectral_swinir import MultispectralSwinIR
from satellite_srm.inference.uncertainty import MonteCarloUncertaintyEstimator

class TestUncertainty(unittest.TestCase):
    def test_mc_dropout_uncertainty(self):
        model = MultispectralSwinIR(scale_factor=2.0, in_channels=4, out_channels=4, drop_rate=0.2)
        estimator = MonteCarloUncertaintyEstimator(model, mc_passes=3)
        x = torch.from_numpy(np.random.rand(1, 4, 16, 16).astype(np.float32))
        res = estimator.predict_with_uncertainty(x)

        self.assertEqual(res.mean_prediction.shape, (4, 32, 32))
        self.assertEqual(res.pixel_stddev.shape, (1, 32, 32))
        self.assertGreaterEqual(res.mean_uncertainty, 0.0)
        self.assertIn("p50", res.percentiles)

if __name__ == "__main__":
    unittest.main()
