"""Checkpoint serialization, safe weight loading, and multispectral partial transfer."""
import os
from typing import Dict, Any, Tuple
from satellite_srm.compat import torch, nn
from satellite_srm.logging_config import get_logger

logger = get_logger("checkpoint")

def save_model_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Any = None,
    scheduler: Any = None,
    epoch: int = 0,
    metrics: Dict[str, float] = None,
    config: Dict[str, Any] = None
) -> str:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if hasattr(optimizer, "state_dict") else None,
        "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config or {}
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Saved checkpoint to {filepath} (Epoch {epoch})")
    return filepath

def load_checkpoint_weights(
    model: nn.Module,
    weights_path: str,
    device: str = "cpu"
) -> Tuple[int, Dict[str, float]]:
    """
    Safely loads checkpoint weights with partial transfer support.
    Transfers compatible layers and logs partially adapted multispectral layers.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint file not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model_dict = model.state_dict()
    matched_dict = {}
    transferred_layers = []
    initialized_layers = []

    for k, v in state_dict.items():
        if k in model_dict:
            v_arr = v.numpy() if hasattr(v, "numpy") else v
            target_arr = model_dict[k].numpy() if hasattr(model_dict[k], "numpy") else model_dict[k]
            if v_arr.shape == target_arr.shape:
                matched_dict[k] = v
                transferred_layers.append(k)
            else:
                initialized_layers.append(f"{k} (Shape mismatch: {v_arr.shape} vs {target_arr.shape})")
        else:
            initialized_layers.append(f"{k} (Not in target model)")

    model.load_state_dict(matched_dict, strict=False)
    logger.info(f"Successfully transferred {len(transferred_layers)} layers from {weights_path}.")
    if initialized_layers:
        logger.info(f"{len(initialized_layers)} layers randomly initialized (multispectral adaptation).")

    epoch = checkpoint.get("epoch", 0) if isinstance(checkpoint, dict) else 0
    metrics = checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}
    return epoch, metrics
