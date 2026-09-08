import unittest
import numpy as np
from satellite_srm.validation.psnr import calculate_psnr
from satellite_srm.validation.ssim import calculate_ssim
from satellite_srm.validation.sam import calculate_sam
from satellite_srm.validation.ergas import calculate_ergas
from satellite_srm.validation.ndvi import evaluate_ndvi_fidelity

class TestMetrics(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.target = np.random.rand(4, 32, 32).astype(np.float32)
        self.pred_identical = self.target.copy()
        self.pred_perturbed = np.clip(self.target + np.random.normal(0, 0.05, self.target.shape), 0.0, 1.0).astype(np.float32)

    def test_psnr_identical(self):
        psnr = calculate_psnr(self.target, self.pred_identical)
        self.assertGreaterEqual(psnr, 99.0)

    def test_psnr_perturbed(self):
        psnr = calculate_psnr(self.target, self.pred_perturbed)
        self.assertGreater(psnr, 20.0)
        self.assertLess(psnr, 40.0)

    def test_ssim_identical(self):
        ssim = calculate_ssim(self.target, self.pred_identical)
        self.assertAlmostEqual(ssim, 1.0, places=3)

    def test_sam_identical(self):
        sam = calculate_sam(self.target, self.pred_identical)
        self.assertAlmostEqual(sam, 0.0, places=3)

    def test_ergas_identical(self):
        ergas = calculate_ergas(self.target, self.pred_identical)
        self.assertAlmostEqual(ergas, 0.0, places=3)

    def test_ndvi_fidelity(self):
        res = evaluate_ndvi_fidelity(self.target, self.pred_identical)
        self.assertAlmostEqual(res["ndvi_mae"], 0.0, places=4)
        self.assertAlmostEqual(res["ndvi_pearson_r"], 1.0, places=3)

if __name__ == "__main__":
    unittest.main()
