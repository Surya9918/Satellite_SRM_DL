"""Comprehensive evaluation suite executing side-by-side benchmark between Bicubic and SR Model."""
import numpy as np
from typing import Dict, Any
from satellite_srm.validation.psnr import calculate_psnr
from satellite_srm.validation.ssim import calculate_ssim
from satellite_srm.validation.sam import calculate_sam
from satellite_srm.validation.ergas import calculate_ergas
from satellite_srm.validation.ndvi import evaluate_ndvi_fidelity
from satellite_srm.validation.baseline import BicubicBaseline
from satellite_srm.logging_config import get_logger

logger = get_logger("validator")

class SRMValidator:
    """Computes comprehensive quantitative metrics comparing SR model against bicubic baseline."""

    def __init__(self, scale_factor: float = 3.0):
        self.scale_factor = scale_factor
        self.baseline = BicubicBaseline(scale_factor=scale_factor)

    def evaluate_pair(self, lr_img: np.ndarray, hr_target: np.ndarray, sr_prediction: np.ndarray) -> Dict[str, Any]:
        baseline_img = self.baseline.upsample(lr_img)

        # Baseline metrics
        base_psnr = calculate_psnr(hr_target, baseline_img)
        base_ssim = calculate_ssim(hr_target, baseline_img)
        base_sam = calculate_sam(hr_target, baseline_img)
        base_ergas = calculate_ergas(hr_target, baseline_img, scale_ratio=self.scale_factor)
        base_ndvi = evaluate_ndvi_fidelity(hr_target, baseline_img)

        # Model metrics
        model_psnr = calculate_psnr(hr_target, sr_prediction)
        model_ssim = calculate_ssim(hr_target, sr_prediction)
        model_sam = calculate_sam(hr_target, sr_prediction)
        model_ergas = calculate_ergas(hr_target, sr_prediction, scale_ratio=self.scale_factor)
        model_ndvi = evaluate_ndvi_fidelity(hr_target, sr_prediction)

        return {
            "metrics": {
                "PSNR_dB": {"Bicubic": round(base_psnr, 2), "SR_Model": round(model_psnr, 2), "Gain": round(model_psnr - base_psnr, 2)},
                "SSIM": {"Bicubic": round(base_ssim, 4), "SR_Model": round(model_ssim, 4), "Gain": round(model_ssim - base_ssim, 4)},
                "SAM_deg": {"Bicubic": round(base_sam, 2), "SR_Model": round(model_sam, 2), "Gain": round(base_sam - model_sam, 2)},
                "ERGAS": {"Bicubic": round(base_ergas, 2), "SR_Model": round(model_ergas, 2), "Gain": round(base_ergas - model_ergas, 2)},
                "NDVI_Correlation": {"Bicubic": round(base_ndvi["ndvi_pearson_r"], 4), "SR_Model": round(model_ndvi["ndvi_pearson_r"], 4), "Gain": round(model_ndvi["ndvi_pearson_r"] - base_ndvi["ndvi_pearson_r"], 4)},
                "NDVI_MAE": {"Bicubic": round(base_ndvi["ndvi_mae"], 4), "SR_Model": round(model_ndvi["ndvi_mae"], 4), "Gain": round(base_ndvi["ndvi_mae"] - model_ndvi["ndvi_mae"], 4)}
            },
            "scale_factor": self.scale_factor
        }
