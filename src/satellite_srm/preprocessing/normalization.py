"""Radiometric normalization preserving physical surface reflectance units."""
import numpy as np
from typing import Tuple, Dict, Any

class RadiometricNormalizer:
    """
    Normalizes Sentinel-2 L2A Digital Numbers (0-10000) into physical reflectance [0.0, 1.0],
    with optional percentile clipping and reversible unnormalization for GeoTIFF generation.
    """

    def __init__(self, scale_factor: float = 10000.0, clip_range: Tuple[float, float] = (0.0, 1.0),
                 use_percentiles: bool = False, p_min: float = 1.0, p_max: float = 99.0):
        self.scale_factor = scale_factor
        self.clip_range = clip_range
        self.use_percentiles = use_percentiles
        self.p_min = p_min
        self.p_max = p_max
        self.stats: Dict[str, Any] = {}

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Converts raw DN or reflectance array into [0.0, 1.0] normalized array."""
        norm = data.astype(np.float32)
        if np.max(norm) > 50.0:  # Raw Digital Numbers (e.g. 0-10000)
            norm = norm / self.scale_factor

        if self.use_percentiles:
            c = norm.shape[0]
            self.stats = {"mins": [], "maxs": []}
            for b in range(c):
                vmin = float(np.percentile(norm[b], self.p_min))
                vmax = float(np.percentile(norm[b], self.p_max))
                self.stats["mins"].append(vmin)
                self.stats["maxs"].append(vmax)
                norm[b] = np.clip((norm[b] - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
            return norm

        return np.clip(norm, self.clip_range[0], self.clip_range[1])

    def unnormalize(self, data: np.ndarray, to_dn: bool = False) -> np.ndarray:
        """Reverses normalization to return true physical surface reflectance or DN."""
        out = data.astype(np.float32)
        if self.use_percentiles and "mins" in self.stats:
            c = out.shape[0]
            for b in range(c):
                vmin = self.stats["mins"][b]
                vmax = self.stats["maxs"][b]
                out[b] = out[b] * (vmax - vmin) + vmin

        if to_dn:
            out = np.clip(out * self.scale_factor, 0, 10000).astype(np.uint16)
        else:
            out = np.clip(out, 0.0, 1.0)
        return out
