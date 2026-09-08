"""Cloud and cirrus masking using Sentinel-2 Scene Classification Layer (SCL) or spectral thresholding."""
import numpy as np
from scipy.ndimage import binary_dilation
from typing import Optional

class CloudMasker:
    """Detects and masks cloud/cirrus pixels using SCL or Blue/SWIR reflectance thresholds."""

    def __init__(self, scl_cloud_classes=(3, 8, 9, 10, 11), dilation_radius: int = 3, threshold: float = 0.35):
        self.scl_cloud_classes = set(scl_cloud_classes)
        self.dilation_radius = dilation_radius
        self.threshold = threshold

    def mask_from_scl(self, scl_data: np.ndarray) -> np.ndarray:
        """Creates boolean cloud mask (True = cloud/invalid) from SCL raster."""
        mask = np.isin(scl_data, list(self.scl_cloud_classes))
        if self.dilation_radius > 0:
            structure = np.ones((2 * self.dilation_radius + 1, 2 * self.dilation_radius + 1))
            mask = binary_dilation(mask, structure=structure)
        return mask

    def mask_from_reflectance(self, b02: np.ndarray, b04: np.ndarray, b08: np.ndarray) -> np.ndarray:
        """
        Spectral thresholding fallback when SCL band is unavailable.
        Clouds exhibit elevated visible reflectance and flat spectral response.
        """
        # Blue reflectance threshold (DN > 2500 or normalized > 0.25)
        bright_blue = (b02 > 2500) if np.max(b02) > 1.0 else (b02 > 0.25)
        bright_red = (b04 > 2200) if np.max(b04) > 1.0 else (b04 > 0.22)
        cloud_candidate = bright_blue & bright_red
        if self.dilation_radius > 0:
            structure = np.ones((2 * self.dilation_radius + 1, 2 * self.dilation_radius + 1))
            cloud_candidate = binary_dilation(cloud_candidate, structure=structure)
        return cloud_candidate
