"""Sub-pixel co-registration between LR Sentinel-2 and HR reference imagery using phase correlation."""
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple
from satellite_srm.logging_config import get_logger

logger = get_logger("registration")

@dataclass
class RegistrationResult:
    shift_x: float
    shift_y: float
    error: float
    overlap_ratio: float
    quality_flag: bool

class CoRegistrationEngine:
    """Calculates spatial alignment shift using 2D Fourier Phase Correlation."""

    def __init__(self, max_error_pixels: float = 2.5, min_overlap: float = 0.80):
        self.max_error_pixels = max_error_pixels
        self.min_overlap = min_overlap

    def register(self, lr_img: np.ndarray, hr_img_downscaled: np.ndarray) -> RegistrationResult:
        """
        Computes the translation vector between LR image and downscaled HR reference.
        Both images must be 2D single-channel arrays (typically NIR or Green band).
        """
        if lr_img.ndim == 3:
            lr_img = lr_img[3] if lr_img.shape[0] > 3 else lr_img[0]
        if hr_img_downscaled.ndim == 3:
            hr_img_downscaled = hr_img_downscaled[3] if hr_img_downscaled.shape[0] > 3 else hr_img_downscaled[0]

        # Ensure matching dimensions
        h = min(lr_img.shape[0], hr_img_downscaled.shape[0])
        w = min(lr_img.shape[1], hr_img_downscaled.shape[1])
        lr_crop = lr_img[:h, :w].astype(np.float32)
        hr_crop = hr_img_downscaled[:h, :w].astype(np.float32)

        # Normalize windows
        lr_crop = (lr_crop - np.mean(lr_crop)) / (np.std(lr_crop) + 1e-6)
        hr_crop = (hr_crop - np.mean(hr_crop)) / (np.std(hr_crop) + 1e-6)

        # 2D Phase Correlation
        shift, response = cv2.phaseCorrelate(lr_crop, hr_crop)
        shift_x, shift_y = float(shift[0]), float(shift[1])
        error = float(np.sqrt(shift_x**2 + shift_y**2))

        overlap = 1.0 - (error / max(h, w))
        quality = (error <= self.max_error_pixels) and (overlap >= self.min_overlap)

        return RegistrationResult(
            shift_x=shift_x,
            shift_y=shift_y,
            error=error,
            overlap_ratio=overlap,
            quality_flag=quality
        )
