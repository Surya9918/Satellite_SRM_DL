import unittest
import numpy as np
from satellite_srm.inference.overlap import generate_blending_window
from satellite_srm.inference.blending import OverlapBlender

class TestBlending(unittest.TestCase):
    def test_hann_window_symmetry(self):
        w = generate_blending_window(32, 32, method="weighted_hann")
        self.assertEqual(w.shape, (32, 32))
        self.assertAlmostEqual(w[16, 16], 1.0, places=1)
        self.assertLess(w[0, 0], 0.1)

    def test_overlap_blender_reconstruction(self):
        blender = OverlapBlender(channels=1, out_height=48, out_width=48)
        tile = np.ones((1, 32, 32), dtype=np.float32)
        # Add overlapping tiles
        blender.add_tile(0, 0, tile)
        blender.add_tile(0, 16, tile)
        blender.add_tile(16, 0, tile)
        blender.add_tile(16, 16, tile)
        recon = blender.finalize()
        self.assertEqual(recon.shape, (1, 48, 48))
        # Constant inputs should blend to approximately 1.0
        np.testing.assert_allclose(recon[:, 8:40, 8:40], 1.0, atol=0.05)

if __name__ == "__main__":
    unittest.main()
