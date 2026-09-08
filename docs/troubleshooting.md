# Troubleshooting & FAQ

### Issue: CUDA Out of Memory
**Resolution**: Reduce `tile_size` in `configs/inference.yaml` to 64 or decrease `batch_size` in `configs/training.yaml`.

### Issue: CDSE Download Fails
**Resolution**: Ensure `.env` contains valid `CDSE_USERNAME` and `CDSE_PASSWORD`. The system will automatically fall back to calibrated offline simulation if credentials are unset.

### Issue: QGIS Projection Warning
**Resolution**: Ensure the `.meta.json` sidecar accompanies the GeoTIFF when running in environments without native GDAL C-extensions.
