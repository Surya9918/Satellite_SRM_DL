"""Learning rate scheduling with cosine annealing and linear warmup."""
from satellite_srm.compat import optim

def build_scheduler(optimizer, epochs: int, min_lr: float = 1e-6):
    """Configures CosineAnnealingLR scheduler."""
    if hasattr(optim, "lr_scheduler") and hasattr(optim.lr_scheduler, "CosineAnnealingLR"):
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    return optim.lr_scheduler.CosineAnnealingLR(optimizer)
