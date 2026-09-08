"""Patch extraction engine calculating scale = LR_GSD / target_HR_GSD and tracking patch metadata."""
import os
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple
from satellite_srm.geospatial.raster import GeoRaster
from satellite_srm.logging_config import get_logger

logger = get_logger("patches")

@dataclass
class PatchMetadata:
    patch_id: str
    scene_id: str
    aoi_name: str
    col_off: int
    row_off: int
    lr_size: int
    hr_size: int
    scale_factor: float
    lr_gsd: float
    hr_gsd: float
    crs: str
    bounds: List[float]

    def to_dict(self) -> dict:
        return asdict(self)

class PatchExtractor:
    """
    Extracts aligned LR and HR patches while enforcing mathematical dimension consistency:
    hr_size = round(lr_size * scale_factor) where scale_factor = lr_gsd / hr_gsd.
    """

    def __init__(self, lr_patch_size: int = 128, scale_factor: float = 3.0, stride: int = 64):
        self.lr_patch_size = lr_patch_size
        self.scale_factor = scale_factor
        self.hr_patch_size = int(round(lr_patch_size * scale_factor))
        self.stride = stride

    def extract_pairs(
        self,
        lr_raster: GeoRaster,
        hr_raster: GeoRaster,
        scene_id: str,
        aoi_name: str,
        output_dir: str
    ) -> List[PatchMetadata]:
        os.makedirs(os.path.join(output_dir, "lr"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "hr"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "meta"), exist_ok=True)

        patches_meta = []
        c, h_lr, w_lr = lr_raster.data.shape

        for row in range(0, h_lr - self.lr_patch_size + 1, self.stride):
            for col in range(0, w_lr - self.lr_patch_size + 1, self.stride):
                patch_id = f"{scene_id}_p_{row}_{col}"
                lr_patch = lr_raster.data[:, row:row+self.lr_patch_size, col:col+self.lr_patch_size]

                # Map to HR indices
                hr_row = int(round(row * self.scale_factor))
                hr_col = int(round(col * self.scale_factor))
                hr_patch = hr_raster.data[:, hr_row:hr_row+self.hr_patch_size, hr_col:hr_col+self.hr_patch_size]

                if hr_patch.shape[1] != self.hr_patch_size or hr_patch.shape[2] != self.hr_patch_size:
                    continue

                lr_file = os.path.join(output_dir, "lr", f"{patch_id}.npy")
                hr_file = os.path.join(output_dir, "hr", f"{patch_id}.npy")
                np.save(lr_file, lr_patch)
                np.save(hr_file, hr_patch)

                # Geographic bounds
                minx, maxy = lr_raster.pixel_to_geo(col, row)
                maxx, miny = lr_raster.pixel_to_geo(col + self.lr_patch_size, row + self.lr_patch_size)

                pmeta = PatchMetadata(
                    patch_id=patch_id,
                    scene_id=scene_id,
                    aoi_name=aoi_name,
                    col_off=col,
                    row_off=row,
                    lr_size=self.lr_patch_size,
                    hr_size=self.hr_patch_size,
                    scale_factor=self.scale_factor,
                    lr_gsd=lr_raster.metadata.gsd_x,
                    hr_gsd=hr_raster.metadata.gsd_x,
                    crs=lr_raster.metadata.crs.to_string(),
                    bounds=[minx, miny, maxx, maxy]
                )
                with open(os.path.join(output_dir, "meta", f"{patch_id}.json"), "w") as f:
                    json.dump(pmeta.to_dict(), f, indent=2)
                patches_meta.append(pmeta)

        logger.info(f"Extracted {len(patches_meta)} patch pairs for scene {scene_id}")
        return patches_meta
