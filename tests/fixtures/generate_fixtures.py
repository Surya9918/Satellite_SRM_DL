"""Generates miniature synthetic multi-band test fixtures."""
import numpy as np
from satellite_srm.geospatial.metadata import GeoMetadata
from satellite_srm.geospatial.raster import GeoRaster
from satellite_srm.compat import Affine, CRS

def get_test_raster(channels: int = 4, height: int = 32, width: int = 32) -> GeoRaster:
    data = (np.random.rand(channels, height, width) * 10000.0).astype(np.float32)
    meta = GeoMetadata(
        width=width,
        height=height,
        count=channels,
        crs=CRS.from_epsg(32644),
        transform=Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0),
        band_names=["B02", "B03", "B04", "B08"][:channels],
        gsd_x=10.0,
        gsd_y=10.0
    )
    return GeoRaster(data, meta)
