"""PyTorch-compatible dataset for multi-spectral LR-HR patch pairs."""
import os
import glob
import numpy as np
from typing import Optional, Callable, Dict, Any
from satellite_srm.compat import torch

class PairedSRDataset:
    """Loads LR and HR satellite patch pairs with on-the-fly normalization and augmentation."""

    def __init__(self, data_dir: str, transform: Optional[Callable] = None):
        self.data_dir = data_dir
        self.transform = transform
        self.lr_files = sorted(glob.glob(os.path.join(data_dir, "lr", "*.npy")))
        self.hr_files = sorted(glob.glob(os.path.join(data_dir, "hr", "*.npy")))
        assert len(self.lr_files) == len(self.hr_files), (
            f"Mismatched pair counts: {len(self.lr_files)} LR vs {len(self.hr_files)} HR in {data_dir}"
        )

    def __len__(self) -> int:
        return len(self.lr_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        lr = np.load(self.lr_files[idx]).astype(np.float32)
        hr = np.load(self.hr_files[idx]).astype(np.float32)

        # Normalize to [0, 1] if values are in raw DN (0-10000)
        if np.max(lr) > 10.0:
            lr = lr / 10000.0
        if np.max(hr) > 10.0:
            hr = hr / 10000.0

        lr = np.clip(lr, 0.0, 1.0)
        hr = np.clip(hr, 0.0, 1.0)

        if self.transform:
            lr, hr = self.transform(lr, hr)

        return {
            "lr": torch.from_numpy(lr),
            "hr": torch.from_numpy(hr),
            "filename": os.path.basename(self.lr_files[idx])
        }
