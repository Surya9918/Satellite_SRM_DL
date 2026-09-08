"""NDVI quantitative evaluation: MAE, RMSE, Pearson r correlation, and absolute bias."""
import numpy as np
from typing import Dict, Any

def compute_ndvi_array(multispectral: np.ndarray, red_idx: int = 2, nir_idx: int = 3) -> np.ndarray:
    red = multispectral[red_idx].astype(np.float64)
    nir = multispectral[nir_idx].astype(np.float64)
    denom = nir + red
    ndvi = np.zeros_like(nir)
    valid = np.abs(denom) > 1e-6
    ndvi[valid] = (nir[valid] - red[valid]) / denom[valid]
    return np.clip(ndvi, -1.0, 1.0)

def evaluate_ndvi_fidelity(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    """Evaluates vegetative spectral fidelity by comparing ground truth NDVI with SR NDVI."""
    ndvi_gt = compute_ndvi_array(target)
    ndvi_sr = compute_ndvi_array(prediction)

    mae = float(np.mean(np.abs(ndvi_gt - ndvi_sr)))
    rmse = float(np.sqrt(np.mean((ndvi_gt - ndvi_sr) ** 2)))

    # Pearson correlation coefficient
    gt_flat = ndvi_gt.flatten()
    sr_flat = ndvi_sr.flatten()
    r_mat = np.corrcoef(gt_flat, sr_flat)
    pearson_r = float(r_mat[0, 1]) if not np.isnan(r_mat[0, 1]) else 1.0
    pearson_r = float(np.clip(pearson_r, -1.0, 1.0))

    return {
        "ndvi_mae": mae,
        "ndvi_rmse": rmse,
        "ndvi_pearson_r": pearson_r,
        "mean_diff": float(np.mean(ndvi_sr - ndvi_gt))
    }
