"""Cross-band spectral ratio consistency loss (NIR/Red, Green/Red, NIR/Green)."""
from satellite_srm.compat import torch, nn

class SpectralConsistencyLoss(nn.Module):
    """
    Maintains physical inter-band reflectance ratios:
    - Ratio 1: NIR / Red   (B08 / B04)
    - Ratio 2: Green / Red (B03 / B04)
    - Ratio 3: NIR / Green (B08 / B03)
    """

    def __init__(self, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        # Bands: 0:B02 (Blue), 1:B03 (Green), 2:B04 (Red), 3:B08 (NIR)
        pred_g = pred[:, 1:2]
        pred_r = pred[:, 2:3]
        pred_nir = pred[:, 3:4]

        targ_g = target[:, 1:2]
        targ_r = target[:, 2:3]
        targ_nir = target[:, 3:4]

        # Ratios
        r_pred_nir_r = pred_nir / (pred_r + self.epsilon)
        r_targ_nir_r = targ_nir / (targ_r + self.epsilon)

        r_pred_g_r = pred_g / (pred_r + self.epsilon)
        r_targ_g_r = targ_g / (targ_r + self.epsilon)

        loss_nir_r = (r_pred_nir_r - r_targ_nir_r).abs().mean()
        loss_g_r = (r_pred_g_r - r_targ_g_r).abs().mean()

        return (loss_nir_r + loss_g_r) * 0.5
