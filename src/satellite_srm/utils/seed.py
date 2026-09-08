"""Deterministic pseudo-random seed configuration."""
import os
import random
import numpy as np
from satellite_srm.compat import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if hasattr(torch, "manual_seed"):
        torch.manual_seed(seed)
