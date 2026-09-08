# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-09-07
### Added
- Complete production-ready pipeline for Deep Learning Super Resolution Mapping (PS ID: 26142, NTRO).
- Sentinel-2 L2A ingestion (B02, B03, B04, B08) with Copernicus Data Space Ecosystem integration.
- WorldStrat reference proxy acquisition and validation module.
- Preprocessing engine: cloud/shadow masking, radiometric normalization, co-registration with phase correlation, and geographic-leakage-free patch extraction.
- 4-Band Multispectral SwinIR transformer backbone and lightweight conditional Diffusion SR model.
- Multi-component loss: Masked L1 + Perceptual VGG RGB + Spectral Angle/Ratio + Differentiable NDVI consistency.
- Monte Carlo Dropout uncertainty estimation generating pixel-wise variance and standard deviation maps.
- Full-scene tiled inference with overlap and Hann/Gaussian window blending.
- Georeferenced GeoTIFF generation preserving CRS and updating affine transform to sub-4m resolution.
- Validation suite computing PSNR, SSIM, SAM, ERGAS, and NDVI correlation against bicubic baseline.
- Domain application analysis for Agriculture (NDVI/health/edges), Urban (edges/NDBI), and Disaster/Flood (NDWI/water extent).
- Production Streamlit interface and comprehensive CLI suite.
