"""Masked L1 reconstruction loss excluding cloud, shadow, and nodata pixels."""
import numpy as np
from satellite_srm.compat import torch, nn, F

class MaskedL1Loss(nn.Module):
    """Computes mean absolute error strictly over valid pixels."""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask=None):
        diff = (pred - target).abs()
        if mask is not None:
            # mask: 1 for valid, 0 for cloud/nodata
            valid_diff = diff * mask
            denom = mask.sum() + 1e-6
            return valid_diff.sum() / denom
        return diff.mean()
