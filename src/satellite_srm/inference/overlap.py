"""2D Hann and Gaussian blending window generators to eliminate tile boundary artifacts."""
import numpy as np

def generate_blending_window(height: int, width: int, method: str = "weighted_hann") -> np.ndarray:
    """
    Generates a 2D weighting window of shape (height, width) with smooth attenuation near edges.
    """
    if method == "uniform":
        return np.ones((height, width), dtype=np.float32)

    # 1D Hann windows
    hann_h = np.hanning(height)
    hann_w = np.hanning(width)
    # Outer product for 2D Hann window
    window_2d = np.outer(hann_h, hann_w).astype(np.float32)

    # Ensure nonzero floor to prevent divide-by-zero at corners
    window_2d = np.maximum(window_2d, 1e-4)

    return window_2d
