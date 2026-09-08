"""Preprocessing pipeline: cloud/shadow masking, normalization, registration, and patching."""
from satellite_srm.preprocessing.cloud_mask import CloudMasker
from satellite_srm.preprocessing.shadow_mask import ShadowMasker
from satellite_srm.preprocessing.bands import BandProcessor
from satellite_srm.preprocessing.normalization import RadiometricNormalizer
from satellite_srm.preprocessing.registration import CoRegistrationEngine, RegistrationResult
from satellite_srm.preprocessing.quality_control import QualityController
from satellite_srm.preprocessing.patches import PatchExtractor, PatchMetadata

__all__ = [
    "CloudMasker", "ShadowMasker", "BandProcessor", "RadiometricNormalizer",
    "CoRegistrationEngine", "RegistrationResult", "QualityController",
    "PatchExtractor", "PatchMetadata"
]
