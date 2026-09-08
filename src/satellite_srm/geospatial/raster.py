"""In-memory multi-band geospatial raster representation."""
import numpy as np
from typing import Tuple, Optional
from satellite_srm.geospatial.metadata import GeoMetadata

class GeoRaster:
    """
    Encapsulates a multi-spectral raster array of shape (bands, height, width)
    with strict geographic georeferencing metadata.
    """
    def __init__(self, data: np.ndarray, metadata: GeoMetadata):
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        assert data.ndim == 3, f"Data must have 3 dimensions (C, H, W), got {data.shape}"
        self.data = data
        self.metadata = metadata

    @property
    def bands(self) -> int:
        return self.data.shape[0]

    @property
    def height(self) -> int:
        return self.data.shape[1]

    @property
    def width(self) -> int:
        return self.data.shape[2]

    def pixel_to_geo(self, col: float, row: float) -> Tuple[float, float]:
        """Maps pixel coordinates (col, row) to geographic projection coordinates (X, Y)."""
        t = self.metadata.transform
        x = t.c + col * t.a + row * t.b
        y = t.f + col * t.d + row * t.e
        return x, y

    def geo_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Maps geographic projection coordinates (X, Y) to integer pixel indices (col, row)."""
        t = self.metadata.transform
        det = t.a * t.e - t.b * t.d
        if abs(det) < 1e-12:
            raise ValueError("Singular affine transform matrix.")
        col = (t.e * (x - t.c) - t.b * (y - t.f)) / det
        row = (-t.d * (x - t.c) + t.a * (y - t.f)) / det
        return int(round(col)), int(round(row))

    def extract_window(self, col_off: int, row_off: int, width: int, height: int) -> "GeoRaster":
        """Extracts a sub-window and updates its geographic affine transform."""
        col_end = min(col_off + width, self.width)
        row_end = min(row_off + height, self.height)
        patch_data = self.data[:, row_off:row_end, col_off:col_end]

        t = self.metadata.transform
        new_c = t.c + col_off * t.a + row_off * t.b
        new_f = t.f + col_off * t.d + row_off * t.e
        new_transform = Affine(t.a, t.b, new_c, t.d, t.e, new_f)

        new_meta = GeoMetadata(
            width=patch_data.shape[2],
            height=patch_data.shape[1],
            count=self.bands,
            crs=self.metadata.crs,
            transform=new_transform,
            dtype=str(patch_data.dtype),
            nodata=self.metadata.nodata,
            band_names=list(self.metadata.band_names),
            gsd_x=self.metadata.gsd_x,
            gsd_y=self.metadata.gsd_y
        )
        return GeoRaster(patch_data, new_meta)
