"""Optimizer construction with weight decay exclusion for normalization layers."""
from satellite_srm.compat import optim, nn

def build_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 1e-4):
    """Constructs AdamW optimizer applying weight decay to weights while excluding biases and LayerNorms."""
    decay_params = []
    no_decay_params = []

    for name, param in model.state_dict().items():
        if "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # In compat or native PyTorch
    if hasattr(optim, "AdamW"):
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optim.DummyOptimAdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
