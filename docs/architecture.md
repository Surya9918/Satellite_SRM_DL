# System Architecture

## Pipeline Overview
The Satellite-SRM system implements an end-to-end remote-sensing pipeline designed for National Technical Research Organisation (NTRO) Problem Statement 26142:

```
[10m Sentinel-2 L2A (B02, B03, B04, B08)]
               │
               ▼
   [Cloud & Shadow Masking (SCL)]
               │
               ▼
  [Radiometric Normalization (1/10000)]
               │
               ▼
 [Sub-pixel Co-Registration (Phase Correl.)]
               │
               ▼
[Patch Extraction (No Geographic Leakage)]
               │
               ▼
[Multispectral SwinIR / Diffusion Backbone]
               │
               ▼
  [Loss: L1 + Perceptual + NDVI + Spectral]
               │
               ▼
   [Monte Carlo Dropout Uncertainty (N=8)]
               │
               ▼
 [Full-Scene Tiled Inference + Hann Blending]
               │
               ▼
  [QGIS-Compatible Sub-4m GeoTIFF Product]
```

## Key Components
1. **Acquisition**: Ingests Sentinel-2 L2A imagery (B02 Blue, B03 Green, B04 Red, B08 NIR) via Copernicus Data Space Ecosystem (CDSE) with SHA-256 manifest tracking.
2. **Preprocessing**: SCL cloud/shadow masking, reflectance normalization, 2D phase-correlation co-registration, and geographic-region-based patch extraction.
3. **Deep Learning Core**: 4-channel Multispectral SwinIR featuring Residual Swin Transformer Blocks (RSTBs) with shifted window multi-head self-attention and cross-band depthwise spectral fusion.
4. **Loss Formulation**: Multi-objective loss balancing structural L1, VGG natural RGB perceptual features, differentiable NDVI consistency, and inter-band spectral ratio constraints.
5. **Uncertainty Quantification**: Monte Carlo Dropout performing N stochastic forward passes at inference to compute pixel-wise variance and standard deviation maps.
6. **Inference & Delivery**: Overlap-aware 2D Hann window blending reconstructing full-scene rasters into georeferenced GeoTIFFs with updated affine transforms.
