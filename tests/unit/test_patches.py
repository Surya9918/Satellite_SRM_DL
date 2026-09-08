import unittest
import numpy as np
import tempfile
import os
from satellite_srm.preprocessing.patches import PatchExtractor
from tests.fixtures.generate_fixtures import get_test_raster

class TestPatchExtraction(unittest.TestCase):
    def test_patch_scale_dimensions(self):
        lr_raster = get_test_raster(channels=4, height=64, width=64)
        hr_raster = get_test_raster(channels=4, height=192, width=192)

        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = PatchExtractor(lr_patch_size=32, scale_factor=3.0, stride=32)
            patches = extractor.extract_pairs(lr_raster, hr_raster, "test_scene", "TestAOI", tmpdir)

            self.assertGreater(len(patches), 0)
            p0 = patches[0]
            self.assertEqual(p0.lr_size, 32)
            self.assertEqual(p0.hr_size, 96)
            self.assertEqual(p0.scale_factor, 3.0)

            # Check saved arrays
            lr_arr = np.load(os.path.join(tmpdir, "lr", f"{p0.patch_id}.npy"))
            hr_arr = np.load(os.path.join(tmpdir, "hr", f"{p0.patch_id}.npy"))
            self.assertEqual(lr_arr.shape, (4, 32, 32))
            self.assertEqual(hr_arr.shape, (4, 96, 96))

if __name__ == "__main__":
    unittest.main()
