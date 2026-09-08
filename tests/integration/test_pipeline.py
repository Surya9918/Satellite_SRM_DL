import unittest
import tempfile
import os
import numpy as np
from satellite_srm.geospatial.geotiff import write_geotiff, read_geotiff
from satellite_srm.models.multispectral_swinir import MultispectralSwinIR
from satellite_srm.inference.pipeline import FullSceneSRMPipeline
from tests.fixtures.generate_fixtures import get_test_raster

class TestIntegrationPipeline(unittest.TestCase):
    def test_end_to_end_inference_pipeline(self):
        lr_raster = get_test_raster(channels=4, height=48, width=48)
        model = MultispectralSwinIR(scale_factor=2.0, in_channels=4, out_channels=4)

        config = {
            "inference": {
                "tile_size": 32,
                "overlap": 8,
                "scale_factor": 2.0,
                "uncertainty": {"mc_passes": 2},
                "blending_method": "weighted_hann"
            }
        }
        pipeline = FullSceneSRMPipeline(model, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_tif = os.path.join(tmpdir, "input.tif")
            output_sr = os.path.join(tmpdir, "output_sr.tif")
            output_unc = os.path.join(tmpdir, "output_unc.tif")

            write_geotiff(input_tif, lr_raster.data, lr_raster.metadata)
            res = pipeline.run(input_tif, output_sr, output_unc)

            self.assertTrue(os.path.exists(output_sr))
            self.assertTrue(os.path.exists(output_unc))

            sr_raster = read_geotiff(output_sr)
            self.assertEqual(sr_raster.bands, 4)
            self.assertEqual(sr_raster.height, 96)
            self.assertEqual(sr_raster.width, 96)
            self.assertAlmostEqual(sr_raster.metadata.gsd_x, 5.0, places=2)

if __name__ == "__main__":
    unittest.main()
