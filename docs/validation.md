# Validation Suite & Metrics

## Quantitative Metrics Computed
1. **PSNR (Peak Signal-to-Noise Ratio)**: Evaluates pixel-level numerical fidelity.
2. **SSIM (Structural Similarity Index)**: Measures luminance, contrast, and structural preservation.
3. **SAM (Spectral Angle Mapper)**: Measures spectral vector angular distortion in degrees.
4. **ERGAS**: Standard remote-sensing dimensionless synthesis error.
5. **NDVI Correlation & MAE**: Quantifies biological vegetation index fidelity.

## Benchmark Execution
```bash
python -m satellite_srm validate
```
Outputs reports to `outputs/reports/validation_report.html` and `outputs/reports/validation_report.json`.
