"""Super-resolution loss functions: Masked L1, VGG Perceptual, Spectral NDVI & Ratio Loss."""
from satellite_srm.losses.l1 import MaskedL1Loss
from satellite_srm.losses.perceptual import PerceptualVGGLoss
from satellite_srm.losses.ndvi import NDVIConsistencyLoss
from satellite_srm.losses.spectral import SpectralConsistencyLoss
from satellite_srm.losses.combined import CombinedSRMLoss

__all__ = [
    "MaskedL1Loss",
    "PerceptualVGGLoss",
    "NDVIConsistencyLoss",
    "SpectralConsistencyLoss",
    "CombinedSRMLoss"
]
