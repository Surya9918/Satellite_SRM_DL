"""Data acquisition, satellite scene search, download managers, and checksum verification."""
from satellite_srm.acquisition.sentinel2 import Sentinel2Downloader
from satellite_srm.acquisition.worldstrat import WorldStratManager
from satellite_srm.acquisition.copernicus import CopernicusClient
from satellite_srm.acquisition.downloader import ChunkedDownloader
from satellite_srm.acquisition.checksums import verify_checksum, compute_sha256
from satellite_srm.acquisition.manifests import AcquisitionManifest

__all__ = [
    "Sentinel2Downloader",
    "WorldStratManager",
    "CopernicusClient",
    "ChunkedDownloader",
    "verify_checksum",
    "compute_sha256",
    "AcquisitionManifest"
]
