"""Cloud shadow detection and morphological dilation."""
import numpy as np
from scipy.ndimage import binary_dilation

class ShadowMasker:
    """Identifies cloud shadow candidates using low NIR reflectance and geographic adjacency."""

    def __init__(self, dilation_size: int = 5, nir_shadow_threshold: float = 1200.0):
        self.dilation_size = dilation_size
        self.nir_shadow_threshold = nir_shadow_threshold

    def mask_shadows(self, b08_nir: np.ndarray, cloud_mask: np.ndarray) -> np.ndarray:
        """Generates shadow mask based on low NIR signature combined with cloud proximity."""
        thresh = self.nir_shadow_threshold if np.max(b08_nir) > 1.0 else (self.nir_shadow_threshold / 10000.0)
        low_nir = (b08_nir < thresh) & ~cloud_mask

        structure = np.ones((self.dilation_size, self.dilation_size))
        cloud_vicinity = binary_dilation(cloud_mask, structure=structure)
        shadow_mask = low_nir & cloud_vicinity
        return shadow_mask
