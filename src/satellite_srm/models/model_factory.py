"""Unified model factory supporting multispectral SwinIR, Diffusion SR, and Bicubic baseline."""
from typing import Dict, Any, List
from satellite_srm.compat import nn, F
from satellite_srm.models.multispectral_swinir import MultispectralSwinIR
from satellite_srm.models.diffusion_sr import LightweightDiffusionSR
from satellite_srm.logging_config import get_logger

logger = get_logger("model_factory")

class BicubicBaselineModel(nn.Module):
    """Analytical baseline performing continuous bicubic interpolation."""
    def __init__(self, scale_factor: float = 3.0):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode="bicubic", align_corners=False)

def create_model(config: Dict[str, Any]) -> nn.Module:
    """Instantiates model architecture according to configuration."""
    backend = config.get("model", {}).get("backend", "swinir").lower()
    m_cfg = config.get("model", {}).get(backend, {})
    scale_factor = float(m_cfg.get("scale", config.get("model", {}).get("scale_factor", 3.0)))

    if backend == "swinir":
        logger.info(f"Initializing Multispectral SwinIR model (Scale: {scale_factor}x, 4 bands)...")
        model = MultispectralSwinIR(
            scale_factor=scale_factor,
            in_channels=m_cfg.get("in_channels", 4),
            out_channels=m_cfg.get("out_channels", 4),
            embed_dim=m_cfg.get("embed_dim", 96),
            depths=m_cfg.get("depths", [4, 4, 4, 4]),
            num_heads=m_cfg.get("num_heads", [6, 6, 6, 6]),
            window_size=m_cfg.get("window_size", 8),
            drop_rate=m_cfg.get("drop_rate", 0.05)
        )
        return model

    elif backend == "diffusion":
        logger.info(f"Initializing Lightweight Conditional Diffusion SR model...")
        model = LightweightDiffusionSR(
            in_channels=m_cfg.get("in_channels", 4),
            out_channels=m_cfg.get("out_channels", 4),
            base_channels=m_cfg.get("base_channels", 64),
            ddim_steps=m_cfg.get("ddim_inference_steps", 8)
        )
        return model

    elif backend in ("bicubic", "baseline_bicubic"):
        logger.info("Initializing Bicubic analytical baseline model...")
        return BicubicBaselineModel(scale_factor=scale_factor)

    else:
        raise ValueError(f"Unsupported model backend: '{backend}'. Choose 'swinir', 'diffusion', or 'bicubic'.")

def list_available_models() -> List[str]:
    return ["swinir", "diffusion", "bicubic"]
