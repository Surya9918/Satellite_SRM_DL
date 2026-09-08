"""Urban analysis: built-up surface enhancement, road/structure boundary extraction, and RGB composite."""
import numpy as np
import cv2
from typing import Dict, Any

class UrbanAnalyzer:
    """Enhances structural boundaries and urban morphology."""

    def analyze(self, data: np.ndarray) -> Dict[str, Any]:
        # True color RGB composite: Red (B04=2), Green (B03=1), Blue (B02=0)
        rgb = np.stack([data[2], data[1], data[0]], axis=-1)
        rgb_norm = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # High-frequency edge extraction for road networks and building footprints
        gray = cv2.cvtColor(rgb_norm, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 75, 200)

        return {
            "rgb_composite": rgb_norm,
            "structural_edges": edges,
            "edge_density": float(np.mean(edges > 0))
        }
