"""Weighted overlap blending accumulator reconstructing full-scene super-resolved rasters."""
import numpy as np
from satellite_srm.inference.overlap import generate_blending_window

class OverlapBlender:
    """Accumulates tile predictions with 2D weighting and normalizes by overlap coverage."""

    def __init__(self, channels: int, out_height: int, out_width: int, blending_method: str = "weighted_hann"):
        self.channels = channels
        self.out_height = out_height
        self.out_width = out_width
        self.blending_method = blending_method

        self.accumulator = np.zeros((channels, out_height, out_width), dtype=np.float64)
        self.weight_sum = np.zeros((1, out_height, out_width), dtype=np.float64)

    def add_tile(self, row_start: int, col_start: int, tile_data: np.ndarray):
        c, h, w = tile_data.shape
        window = generate_blending_window(h, w, method=self.blending_method)
        weighted_tile = tile_data * window[np.newaxis, ...]

        self.accumulator[:, row_start:row_start+h, col_start:col_start+w] += weighted_tile
        self.weight_sum[:, row_start:row_start+h, col_start:col_start+w] += window[np.newaxis, ...]

    def finalize(self) -> np.ndarray:
        """Normalizes accumulated pixels by total blended weight."""
        weights = np.maximum(self.weight_sum, 1e-8)
        reconstructed = self.accumulator / weights
        return reconstructed.astype(np.float32)
