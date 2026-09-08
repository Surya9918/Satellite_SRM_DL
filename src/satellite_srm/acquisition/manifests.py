"""Acquisition manifest tracking and JSON schema validation."""
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class AcquisitionManifest:
    scene_id: str
    sensor: str
    product: str
    acquisition_date: str
    cloud_cover: float
    bands: List[str]
    crs: str
    source: str
    checksum: str
    file_path: str

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, output_dir: str = "data/manifests") -> str:
        os.makedirs(output_dir, exist_ok=True)
        manifest_path = os.path.join(output_dir, f"{self.scene_id}_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return manifest_path

    @classmethod
    def load(cls, manifest_path: str) -> "AcquisitionManifest":
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
