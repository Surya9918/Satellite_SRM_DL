"""Multi-spectral band alignment, validation, and naming standardization."""
from typing import List, Dict
import numpy as np

REQUIRED_BANDS = ["B02", "B03", "B04", "B08"]

class BandProcessor:
    """Manages Sentinel-2 10m native bands (Blue, Green, Red, NIR)."""

    @staticmethod
    def validate_band_count(data: np.ndarray, expected: int = 4):
        if data.shape[0] != expected:
            raise ValueError(f"Expected {expected} bands (B02, B03, B04, B08), found {data.shape[0]} bands.")

    @staticmethod
    def get_band_indices(band_names: List[str]) -> Dict[str, int]:
        mapping = {}
        for idx, name in enumerate(band_names):
            clean = name.upper().strip()
            if "B02" in clean or "BLUE" in clean:
                mapping["B02"] = idx
            elif "B03" in clean or "GREEN" in clean:
                mapping["B03"] = idx
            elif "B04" in clean or "RED" in clean:
                mapping["B04"] = idx
            elif "B08" in clean or "NIR" in clean:
                mapping["B08"] = idx
        return mapping
