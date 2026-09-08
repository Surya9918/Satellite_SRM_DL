"""Monte Carlo Dropout uncertainty quantification generating pixel-level variance and confidence maps."""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List
from satellite_srm.compat import torch, nn
from satellite_srm.logging_config import get_logger

logger = get_logger("uncertainty")

@dataclass
class UncertaintyResult:
    mean_prediction: np.ndarray       # Shape (C, H, W)
    pixel_variance: np.ndarray        # Shape (1, H, W) or (C, H, W)
    pixel_stddev: np.ndarray          # Shape (1, H, W) or (C, H, W)
    mean_uncertainty: float
    max_uncertainty: float
    percentiles: Dict[str, float]

class MonteCarloUncertaintyEstimator:
    """
    Executes N stochastic forward passes with active Dropout at inference:
    {SR_1, SR_2, ..., SR_N} -> Mean Prediction + Pixel-wise Variance Map.
    """

    def __init__(self, model: nn.Module, mc_passes: int = 8):
        self.model = model
        self.mc_passes = max(2, mc_passes)

    def enable_dropout_at_inference(self):
        """Forces dropout layers to remain active during evaluation."""
        self.model.eval()
        for m in getattr(self.model, "_modules", {}).values():
            if isinstance(m, nn.Dropout):
                m.train(True)

    def predict_with_uncertainty(self, x_input) -> UncertaintyResult:
        self.enable_dropout_at_inference()
        predictions = []

        for p in range(self.mc_passes):
            pred = self.model(x_input)
            arr = pred.detach().cpu().numpy() if hasattr(pred, "detach") else (pred.numpy() if hasattr(pred, "numpy") else pred)
            predictions.append(arr)

        pred_stack = np.stack(predictions, axis=0) # (N, B, C, H, W)
        mean_pred = np.mean(pred_stack, axis=0)[0]  # (C, H, W)
        var_map = np.var(pred_stack, axis=0)[0]    # (C, H, W)
        std_map = np.sqrt(var_map)

        # Spatial scalar uncertainty across spectral channels
        spatial_uncertainty = np.mean(std_map, axis=0, keepdims=True) # (1, H, W)

        p_vals = {
            "p50": float(np.percentile(spatial_uncertainty, 50)),
            "p90": float(np.percentile(spatial_uncertainty, 90)),
            "p95": float(np.percentile(spatial_uncertainty, 95)),
            "p99": float(np.percentile(spatial_uncertainty, 99))
        }

        return UncertaintyResult(
            mean_prediction=mean_pred,
            pixel_variance=np.mean(var_map, axis=0, keepdims=True),
            pixel_stddev=spatial_uncertainty,
            mean_uncertainty=float(np.mean(spatial_uncertainty)),
            max_uncertainty=float(np.max(spatial_uncertainty)),
            percentiles=p_vals
        )
