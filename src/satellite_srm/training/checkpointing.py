"""Checkpoint management saving best.pt, latest.pt, and periodic epoch weights."""
import os
from typing import Dict, Any
from satellite_srm.models.checkpoint import save_model_checkpoint
from satellite_srm.logging_config import get_logger

logger = get_logger("checkpoint_manager")

class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_metric = -float("inf")

    def save_epoch(self, model, optimizer, scheduler, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        save_model_checkpoint(latest_path, model, optimizer, scheduler, epoch, metrics)

        epoch_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pt")
        save_model_checkpoint(epoch_path, model, optimizer, scheduler, epoch, metrics)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pt")
            save_model_checkpoint(best_path, model, optimizer, scheduler, epoch, metrics)
            logger.info(f"*** New best checkpoint saved to {best_path} (Metric: {metrics}) ***")
