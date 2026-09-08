"""Mixed precision autocast and gradient scaler wrappers for memory-efficient training."""
from satellite_srm.compat import torch

class MixedPrecisionManager:
    def __init__(self, enabled: bool = True, device: str = "cuda"):
        self.enabled = enabled and (device == "cuda") and hasattr(torch, "cuda") and torch.cuda.is_available()
        if hasattr(torch, "cuda_amp"):
            self.scaler = torch.cuda_amp.GradScaler() if self.enabled else None
        else:
            self.scaler = None

    def autocast(self):
        if hasattr(torch, "autocast"):
            return torch.autocast(device_type="cuda" if self.enabled else "cpu", enabled=self.enabled)
        return torch.DummyAutocast()
