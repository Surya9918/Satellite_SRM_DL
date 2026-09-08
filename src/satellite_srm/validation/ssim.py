"""Structural Similarity Index Measure (SSIM) computed across spatial sliding windows."""
import numpy as np
from scipy.ndimage import uniform_filter

def calculate_ssim(target: np.ndarray, prediction: np.ndarray, win_size: int = 11, data_range: float = 1.0) -> float:
    """Computes mean SSIM across all spectral bands."""
    if target.ndim == 2:
        target = target[np.newaxis, ...]
    if prediction.ndim == 2:
        prediction = prediction[np.newaxis, ...]

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_per_band = []
    for b in range(target.shape[0]):
        x = target[b].astype(np.float64)
        y = prediction[b].astype(np.float64)

        ux = uniform_filter(x, size=win_size)
        uy = uniform_filter(y, size=win_size)

        uxx = uniform_filter(x * x, size=win_size)
        uyy = uniform_filter(y * y, size=win_size)
        uxy = uniform_filter(x * y, size=win_size)

        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy

        ssim_map = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2) + 1e-12)
        ssim_per_band.append(float(np.mean(ssim_map)))

    return float(np.mean(ssim_per_band))
