"""Perceptual feature loss for visible RGB channels (B04-B03-B02) using VGG feature extractor."""
from satellite_srm.compat import torch, nn

class PerceptualVGGLoss(nn.Module):
    """
    Extracts deep high-frequency features on natural color channels:
    B04 (Red), B03 (Green), B02 (Blue).
    Does NOT force NIR through RGB networks.
    """

    def __init__(self):
        super().__init__()
        # 3-channel feature extractor
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, pred, target):
        # Extract RGB channels: B04 (index 2), B03 (index 1), B02 (index 0)
        # Reorder to [R, G, B]
        pred_rgb = torch.from_numpy(pred.numpy()[:, [2, 1, 0], :, :]) if hasattr(pred, "numpy") else pred[:, [2, 1, 0], :, :]
        targ_rgb = torch.from_numpy(target.numpy()[:, [2, 1, 0], :, :]) if hasattr(target, "numpy") else target[:, [2, 1, 0], :, :]

        f_pred = self.relu(self.conv2(self.relu(self.conv1(pred_rgb))))
        f_targ = self.relu(self.conv2(self.relu(self.conv1(targ_rgb))))

        loss = (f_pred - f_targ).abs().mean()
        return loss
