"""Agricultural analysis: crop canopy health classification, zonal metrics, and field edge delineation."""
import numpy as np
import cv2
from typing import Dict, Any
from satellite_srm.applications.indices import SpectralIndices

class AgricultureAnalyzer:
    """Extracts agronomic intelligence from super-resolved Sentinel-2 imagery."""

    def __init__(self):
        pass

    def analyze(self, data: np.ndarray) -> Dict[str, Any]:
        ndvi = SpectralIndices.ndvi(data)
        # Zonal classification
        barren = ndvi < 0.20
        moderate_veg = (ndvi >= 0.20) & (ndvi < 0.50)
        vigorous_crop = ndvi >= 0.50

        # Field edge detection using morphological gradient on high-resolution NIR
        nir_norm = cv2.normalize(data[3], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        edges = cv2.Canny(nir_norm, 50, 150)

        total_px = ndvi.size
        return {
            "mean_ndvi": float(np.mean(ndvi)),
            "barren_soil_percent": float(np.sum(barren) / total_px * 100),
            "moderate_vegetation_percent": float(np.sum(moderate_veg) / total_px * 100),
            "vigorous_canopy_percent": float(np.sum(vigorous_crop) / total_px * 100),
            "field_boundary_edges": edges,
            "ndvi_map": ndvi
        }
