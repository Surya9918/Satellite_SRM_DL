"""Full-scene super-resolution mapping inference pipeline preserving georeferencing and uncertainty."""
import os
import time
import numpy as np
from typing import Dict, Any, Optional
from satellite_srm.compat import torch, nn
from satellite_srm.geospatial.geotiff import read_geotiff, write_geotiff
from satellite_srm.preprocessing.normalization import RadiometricNormalizer
from satellite_srm.inference.tiled import TileExtractor
from satellite_srm.inference.blending import OverlapBlender
from satellite_srm.inference.uncertainty import MonteCarloUncertaintyEstimator
from satellite_srm.device import select_device
from satellite_srm.logging_config import get_logger

logger = get_logger("inference_pipeline")

class FullSceneSRMPipeline:
    """
    Executes production inference:
    10m Sentinel-2 GeoTIFF -> Preprocessing -> Tiled Batch Inference -> Blending -> GeoTIFF Sub-4m.
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any] = None):
        self.model = model
        self.config = config or {}
        i_cfg = self.config.get("inference", {})
        self.tile_size = i_cfg.get("tile_size", 128)
        self.overlap = i_cfg.get("overlap", 32)
        self.scale_factor = float(i_cfg.get("scale_factor", 3.0))
        self.mc_passes = i_cfg.get("uncertainty", {}).get("mc_passes", 8)
        self.blending_method = i_cfg.get("blending_method", "weighted_hann")
        self.device = torch.device(select_device(str(self.config.get("device", "auto"))))
        self.model = self.model.to(self.device)
        self.normalizer = RadiometricNormalizer()
        self.tile_extractor = TileExtractor(tile_size=self.tile_size, overlap=self.overlap)
        self.uncertainty_estimator = MonteCarloUncertaintyEstimator(self.model, mc_passes=self.mc_passes)

    def run(self, input_geotiff_path: str, output_sr_path: str, output_uncertainty_path: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.time()
        logger.info(f"Starting Full-Scene SRM Inference on: {input_geotiff_path}")
        lr_raster = read_geotiff(input_geotiff_path)

        norm_data = self.normalizer.normalize(lr_raster.data)
        c, h, w = norm_data.shape

        out_h = int(round(h * self.scale_factor))
        out_w = int(round(w * self.scale_factor))

        sr_blender = OverlapBlender(c, out_h, out_w, blending_method=self.blending_method)
        unc_blender = OverlapBlender(1, out_h, out_w, blending_method=self.blending_method)

        tiles = self.tile_extractor.get_tiles(h, w)
        logger.info(f"Decomposed scene into {len(tiles)} overlapping tiles (Tile: {self.tile_size}, Overlap: {self.overlap})")

        for idx, t in enumerate(tiles):
            tile_crop = norm_data[:, t.row_off:t.row_off+t.height, t.col_off:t.col_off+t.width]
            # Zero-pad if edge tile is smaller than standard tile_size
            pad_h = self.tile_size - t.height
            pad_w = self.tile_size - t.width
            if pad_h > 0 or pad_w > 0:
                tile_crop = np.pad(tile_crop, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

            tensor_in = torch.from_numpy(tile_crop[np.newaxis, ...]).to(self.device)
            unc_res = self.uncertainty_estimator.predict_with_uncertainty(tensor_in)

            # Crop back to unpadded target scale
            target_h = int(round(t.height * self.scale_factor))
            target_w = int(round(t.width * self.scale_factor))
            tile_sr = unc_res.mean_prediction[:, :target_h, :target_w]
            tile_unc = unc_res.pixel_stddev[:, :target_h, :target_w]

            target_row = int(round(t.row_off * self.scale_factor))
            target_col = int(round(t.col_off * self.scale_factor))

            sr_blender.add_tile(target_row, target_col, tile_sr)
            unc_blender.add_tile(target_row, target_col, tile_unc)

        sr_image = sr_blender.finalize()
        unc_image = unc_blender.finalize()

        # Update metadata to target sub-4m spatial resolution
        sr_meta = lr_raster.metadata.scale_resolution(self.scale_factor)
        unc_meta = lr_raster.metadata.scale_resolution(self.scale_factor)
        unc_meta.count = 1
        unc_meta.band_names = ["Uncertainty_StdDev"]

        # Write georeferenced outputs
        write_geotiff(output_sr_path, sr_image, sr_meta)
        if output_uncertainty_path is None:
            output_uncertainty_path = output_sr_path.replace(".tif", "_uncertainty.tif")
        write_geotiff(output_uncertainty_path, unc_image, unc_meta)

        elapsed = time.time() - t0
        logger.info(f"Full-Scene inference completed in {elapsed:.2f}s. Sub-4m product: {output_sr_path}")

        return {
            "sr_output_path": output_sr_path,
            "uncertainty_output_path": output_uncertainty_path,
            "scale_factor": self.scale_factor,
            "target_gsd": sr_meta.gsd_x,
            "runtime_seconds": elapsed,
            "tiles_processed": len(tiles),
            "mean_uncertainty": float(np.mean(unc_image))
        }
