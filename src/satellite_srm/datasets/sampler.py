"""Geographic region-based train/val/test data partitioning to eliminate spatial leakage."""
import os
import json
import shutil
from typing import List, Dict
from satellite_srm.logging_config import get_logger

logger = get_logger("sampler")

class GeographicSplitter:
    """Partitions patch pairs into disjoint geographic regions."""

    @staticmethod
    def split(
        all_patches_meta: List[Dict],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        output_base_dir: str = "data/processed"
    ) -> Dict[str, List[str]]:
        # Sort patches by longitude to ensure spatial clustering by region
        sorted_meta = sorted(all_patches_meta, key=lambda m: m["bounds"][0])
        total = len(sorted_meta)
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)

        train_patches = sorted_meta[:n_train]
        val_patches = sorted_meta[n_train:n_train+n_val]
        test_patches = sorted_meta[n_train+n_val:]

        split_dict = {
            "train": [p["patch_id"] for p in train_patches],
            "val": [p["patch_id"] for p in val_patches],
            "test": [p["patch_id"] for p in test_patches]
        }

        # Save manifest
        os.makedirs(os.path.join(output_base_dir, "..", "manifests"), exist_ok=True)
        manifest_path = os.path.join(output_base_dir, "..", "manifests", "spatial_split_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(split_dict, f, indent=2)

        logger.info(f"Geographic split complete: {len(train_patches)} train, {len(val_patches)} val, {len(test_patches)} test.")
        return split_dict
