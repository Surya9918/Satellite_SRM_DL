"""Exponential Moving Average (EMA) of model parameters for smoother inference outputs."""
import copy
from satellite_srm.compat import nn

class ExponentialMovingAverage:
    """Maintains shadow parameters moving at exponential decay rate."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.state_dict().items():
            arr = param.detach().cpu().numpy() if hasattr(param, "detach") else (param.numpy() if hasattr(param, "numpy") else param)
            self.shadow[name] = arr.copy()

    def update(self, model: nn.Module):
        for name, param in model.state_dict().items():
            arr = param.detach().cpu().numpy() if hasattr(param, "detach") else (param.numpy() if hasattr(param, "numpy") else param)
            self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * arr

    def apply_shadow(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=False)
