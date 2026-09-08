"""Standard remote sensing index calculations (NDVI, NDWI, NDBI, SAVI, EVI)."""
import numpy as np

class SpectralIndices:
    """Calculates vegetative, water, and built-up indices from 4-band Sentinel-2 (B02, B03, B04, B08)."""

    @staticmethod
    def ndvi(data: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red)."""
        red = data[2].astype(np.float32)
        nir = data[3].astype(np.float32)
        return (nir - red) / (nir + red + eps)

    @staticmethod
    def ndwi(data: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Normalized Difference Water Index (McFeeters): (Green - NIR) / (Green + NIR)."""
        green = data[1].astype(np.float32)
        nir = data[3].astype(np.float32)
        return (green - nir) / (green + nir + eps)

    @staticmethod
    def savi(data: np.ndarray, L: float = 0.5, eps: float = 1e-6) -> np.ndarray:
        """Soil-Adjusted Vegetation Index."""
        red = data[2].astype(np.float32)
        nir = data[3].astype(np.float32)
        return ((nir - red) / (nir + red + L + eps)) * (1.0 + L)
