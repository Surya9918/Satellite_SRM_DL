"""Differentiable Normalized Difference Vegetation Index (NDVI) consistency loss."""
from satellite_srm.compat import torch, nn

class NDVIConsistencyLoss(nn.Module):
    """
    Computes NDVI = (B08 - B04) / (B08 + B04 + eps) and enforces spectral vegetative consistency:
    L_ndvi = mean(|NDVI_SR - NDVI_HR|)
    Sentinel-2 Band indices: B04=2 (Red), B08=3 (NIR).
    """

    def __init__(self, red_idx: int = 2, nir_idx: int = 3, epsilon: float = 1e-6):
        super().__init__()
        self.red_idx = red_idx
        self.nir_idx = nir_idx
        self.epsilon = epsilon

    def compute_ndvi(self, x):
        red = x[:, self.red_idx:self.red_idx+1, :, :]
        nir = x[:, self.nir_idx:self.nir_idx+1, :, :]
        ndvi = (nir - red) / (nir + red + self.epsilon)
        return ndvi

    def forward(self, pred, target):
        ndvi_pred = self.compute_ndvi(pred)
        ndvi_target = self.compute_ndvi(target)
        loss = (ndvi_pred - ndvi_target).abs().mean()
        return loss
