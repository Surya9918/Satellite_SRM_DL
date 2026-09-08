"""Peak Signal-to-Noise Ratio (PSNR) calculation for multi-spectral remote sensing."""
import numpy as np

def calculate_psnr(target: np.ndarray, prediction: np.ndarray, data_range: float = 1.0) -> float:
    """Computes PSNR in decibels across multi-channel arrays."""
    mse = np.mean((target.astype(np.float64) - prediction.astype(np.float64)) ** 2)
    if mse < 1e-12:
        return 100.0
    psnr = 20.0 * np.log10(data_range / np.sqrt(mse))
    return float(psnr)
