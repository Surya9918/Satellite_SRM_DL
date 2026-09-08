import unittest
import numpy as np
import tempfile
import os
from satellite_srm.geospatial.geotiff import write_geotiff, read_geotiff
from tests.fixtures.generate_fixtures import get_test_raster

class TestGeoTIFF(unittest.TestCase):
    def test_write_and_read_metadata_preservation(self):
        raster = get_test_raster(channels=4, height=32, width=32)
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "sample.tif")
            write_geotiff(fpath, raster.data, raster.metadata)
            recovered = read_geotiff(fpath)

            self.assertEqual(recovered.bands, 4)
            self.assertEqual(recovered.height, 32)
            self.assertEqual(recovered.width, 32)
            self.assertAlmostEqual(recovered.metadata.gsd_x, 10.0, places=2)

if __name__ == "__main__":
    unittest.main()
