"""Large-scene tile partitioner supporting configurable overlaps and memory-aware bounds."""
from dataclasses import dataclass
from typing import List

@dataclass
class TileWindow:
    row_off: int
    col_off: int
    height: int
    width: int

class TileExtractor:
    """Splits full-scene raster dimensions into overlapping processing tiles."""

    def __init__(self, tile_size: int = 128, overlap: int = 32):
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap

    def get_tiles(self, total_height: int, total_width: int) -> List[TileWindow]:
        tiles = []
        for r in range(0, total_height, self.stride):
            for c in range(0, total_width, self.stride):
                h = min(self.tile_size, total_height - r)
                w = min(self.tile_size, total_width - c)
                tiles.append(TileWindow(row_off=r, col_off=c, height=h, width=w))
        return tiles
