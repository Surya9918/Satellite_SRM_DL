"""Spectral Angle Mapper (SAM) measuring spectral vector angle deviation."""
import numpy as np

def calculate_sam(target: np.ndarray, prediction: np.ndarray, in_degrees: bool = True) -> float:
    """
    Computes average spectral angle between ground truth and predicted spectral vectors:
    SAM = arccos( <T, P> / (||T|| * ||P||) )
    """
    # Shapes: (C, H, W)
    t = target.astype(np.float64)
    p = prediction.astype(np.float64)

    dot_product = np.sum(t * p, axis=0)
    norm_t = np.sqrt(np.sum(t ** 2, axis=0))
    norm_p = np.sqrt(np.sum(p ** 2, axis=0))

    denom = norm_t * norm_p
    valid_mask = denom > 1e-8

    cos_theta = np.ones_like(dot_product)
    cos_theta[valid_mask] = np.clip(dot_product[valid_mask] / denom[valid_mask], -1.0, 1.0)
    angles_rad = np.arccos(cos_theta)

    mean_angle = float(np.mean(angles_rad[valid_mask])) if np.any(valid_mask) else 0.0
    if in_degrees:
        return float(np.degrees(mean_angle))
    return mean_angle
