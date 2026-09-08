"""Disaster & flood extent analysis: NDWI waterbody mapping and inundation boundary delineation."""
import numpy as np
import cv2
from typing import Dict, Any
from satellite_srm.applications.indices import SpectralIndices

class DisasterAnalyzer:
    """Delineates flood water extents and surface inundation zones."""

    def analyze(self, data: np.ndarray, threshold: float = 0.0) -> Dict[str, Any]:
        ndwi = SpectralIndices.ndwi(data)
        water_mask = ndwi > threshold

        # Delineate water boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(water_mask.astype(np.uint8), kernel)
        boundaries = dilated - water_mask.astype(np.uint8)

        water_area_ratio = float(np.mean(water_mask))
        return {
            "water_mask": water_mask,
            "water_boundaries": boundaries,
            "inundation_percentage": water_area_ratio * 100.0,
            "ndwi_map": ndwi
        }
