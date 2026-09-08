"""Large-scene tiled inference, Hann window blending, and Monte Carlo uncertainty quantification."""
from satellite_srm.inference.tiled import TileExtractor, TileWindow
from satellite_srm.inference.overlap import generate_blending_window
from satellite_srm.inference.blending import OverlapBlender
from satellite_srm.inference.uncertainty import MonteCarloUncertaintyEstimator, UncertaintyResult
from satellite_srm.inference.pipeline import FullSceneSRMPipeline

__all__ = [
    "TileExtractor",
    "TileWindow",
    "generate_blending_window",
    "OverlapBlender",
    "MonteCarloUncertaintyEstimator",
    "UncertaintyResult",
    "FullSceneSRMPipeline"
]
