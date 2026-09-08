"""Geospatial metadata representation and transform scaling."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from satellite_srm.compat import Affine, CRS

@dataclass
class GeoMetadata:
    width: int
    height: int
    count: int
    crs: CRS
    transform: Affine
    dtype: str = "float32"
    nodata: Optional[float] = None
    bounds: Optional[Tuple[float, float, float, float]] = None  # (minx, miny, maxx, maxy)
    band_names: List[str] = field(default_factory=lambda: ["B02", "B03", "B04", "B08"])
    gsd_x: float = 10.0
    gsd_y: float = 10.0

    def __post_init__(self):
        if self.bounds is None and self.transform is not None:
            minx = self.transform.c
            maxy = self.transform.f
            maxx = minx + self.width * self.transform.a
            miny = maxy + self.height * self.transform.e
            self.bounds = (min(minx, maxx), min(miny, maxy), max(minx, maxx), max(miny, maxy))

    def scale_resolution(self, scale_factor: float) -> "GeoMetadata":
        """
        Updates the affine transform and raster dimensions to reflect enhanced super-resolution.
        For an upscaling factor S (e.g. 3.0), the pixel size is divided by S and dimensions are multiplied by S.
        """
        new_width = int(round(self.width * scale_factor))
        new_height = int(round(self.height * scale_factor))
        new_a = self.transform.a / scale_factor
        new_e = self.transform.e / scale_factor
        new_transform = Affine(new_a, self.transform.b, self.transform.c,
                               self.transform.d, new_e, self.transform.f)
        return GeoMetadata(
            width=new_width,
            height=new_height,
            count=self.count,
            crs=self.crs,
            transform=new_transform,
            dtype=self.dtype,
            nodata=self.nodata,
            bounds=self.bounds,
            band_names=list(self.band_names),
            gsd_x=self.gsd_x / scale_factor,
            gsd_y=self.gsd_y / scale_factor
        )
