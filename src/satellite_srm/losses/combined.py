"""Combined multi-objective loss: L_total = λ1*L_L1 + λ2*L_perceptual + λ3*L_spectral + λ4*L_ndvi."""
from typing import Dict, Any
from satellite_srm.compat import nn
from satellite_srm.losses.l1 import MaskedL1Loss
from satellite_srm.losses.perceptual import PerceptualVGGLoss
from satellite_srm.losses.ndvi import NDVIConsistencyLoss
from satellite_srm.losses.spectral import SpectralConsistencyLoss

class CombinedSRMLoss(nn.Module):
    """Orchestrates multi-objective training balancing spatial fidelity with spectral integrity."""

    def __init__(self, l1_weight: float = 1.0, perceptual_weight: float = 0.1,
                 spectral_weight: float = 0.25, ndvi_weight: float = 0.50):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.spectral_weight = spectral_weight
        self.ndvi_weight = ndvi_weight

        self.l1_criterion = MaskedL1Loss()
        self.perceptual_criterion = PerceptualVGGLoss()
        self.ndvi_criterion = NDVIConsistencyLoss()
        self.spectral_criterion = SpectralConsistencyLoss()

    def forward(self, pred, target, mask=None) -> Dict[str, Any]:
        l1_val = self.l1_criterion(pred, target, mask=mask)
        perc_val = self.perceptual_criterion(pred, target)
        ndvi_val = self.ndvi_criterion(pred, target)
        spec_val = self.spectral_criterion(pred, target)

        total_loss = (
            self.l1_weight * l1_val +
            self.perceptual_weight * perc_val +
            self.ndvi_weight * ndvi_val +
            self.spectral_weight * spec_val
        )

        return {
            "total_loss": total_loss,
            "l1_loss": l1_val,
            "perceptual_loss": perc_val,
            "ndvi_loss": ndvi_val,
            "spectral_loss": spec_val
        }
