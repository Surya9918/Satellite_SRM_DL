"""Analytical bicubic baseline interpolation engine."""
import numpy as np
import cv2

class BicubicBaseline:
    """Standard interpolation baseline without learned deep priors."""

    def __init__(self, scale_factor: float = 3.0):
        self.scale_factor = scale_factor

    def upsample(self, lr_multispectral: np.ndarray) -> np.ndarray:
        c, h, w = lr_multispectral.shape
        new_h, new_w = int(round(h * self.scale_factor)), int(round(w * self.scale_factor))
        out = np.zeros((c, new_h, new_w), dtype=lr_multispectral.dtype)

        for b in range(c):
            out[b] = cv2.resize(lr_multispectral[b], (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return out
