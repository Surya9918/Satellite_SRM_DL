"""
Hardware device management, CUDA detection, memory monitoring, and safe runtime configuration.
"""

from typing import Dict, Any
from satellite_srm.compat import torch
from satellite_srm.logging_config import get_logger

logger = get_logger("device")

def select_device(preferred: str = "auto") -> str:
    """
    Selects the requested hardware device.
    """
    preferred = preferred.lower()
    if preferred == "cpu":
        logger.info("Explicit CPU device selected.")
        return "cpu"

    if preferred == "cuda" and not (hasattr(torch, "cuda") and torch.cuda.is_available()):
        raise RuntimeError("CUDA GPU was requested, but this PyTorch environment cannot access CUDA.")

    if hasattr(torch, "cuda") and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA accelerator available: {device_name}")
        return "cuda"

    logger.info("CUDA not available; using CPU for automatic device selection.")
    return "cpu"

def get_device_info() -> Dict[str, Any]:
    """Returns memory and device specification dictionary."""
    info = {
        "cuda_available": False,
        "device_count": 0,
        "current_device": "cpu",
        "device_name": "CPU",
        "memory_allocated_mb": 0.0,
        "max_memory_allocated_mb": 0.0,
    }
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        info["cuda_available"] = True
        info["device_count"] = torch.cuda.device_count()
        info["current_device"] = "cuda:0"
        info["device_name"] = torch.cuda.get_device_name(0)
        info["memory_allocated_mb"] = torch.cuda.memory_allocated(0) / (1024 * 1024)
        info["max_memory_allocated_mb"] = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
    return info
