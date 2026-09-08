from typing import Tuple
"""Quality control and automated rejection of degraded, cloud-contaminated, or corrupt pairs."""
import numpy as np
from satellite_srm.preprocessing.registration import RegistrationResult

class QualityController:
    """Validates patch pairs against strict scientific quality gates."""

    def __init__(self, max_cloud_fraction: float = 0.10, max_nodata_fraction: float = 0.05):
        self.max_cloud_fraction = max_cloud_fraction
        self.max_nodata_fraction = max_nodata_fraction

    def evaluate_patch(
        self,
        lr_patch: np.ndarray,
        hr_patch: np.ndarray,
        cloud_mask: np.ndarray,
        reg_result: RegistrationResult
    ) -> Tuple[bool, str]:
        """
        Returns (is_valid, reason).
        """
        if not reg_result.quality_flag:
            return False, f"Registration failed: error={reg_result.error:.2f}px"

        cloud_ratio = float(np.mean(cloud_mask))
        if cloud_ratio > self.max_cloud_fraction:
            return False, f"Excessive cloud cover: {cloud_ratio*100:.1f}% > {self.max_cloud_fraction*100:.1f}%"

        nodata_ratio = float(np.mean(lr_patch == 0))
        if nodata_ratio > self.max_nodata_fraction:
            return False, f"Excessive nodata: {nodata_ratio*100:.1f}%"

        if np.isnan(lr_patch).any() or np.isnan(hr_patch).any():
            return False, "NaN values detected in patch pair"

        if np.isinf(lr_patch).any() or np.isinf(hr_patch).any():
            return False, "Inf values detected in patch pair"

        return True, "Passed QC"
