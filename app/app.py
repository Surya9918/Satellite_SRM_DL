"""Streamlit Production UI for Satellite-SRM."""
import os
import re
import sys
import numpy as np

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

try:
    import streamlit as st
except ImportError:
    print("Streamlit is not installed. Run: pip install streamlit")
    sys.exit(0)

from satellite_srm.config import load_config
from satellite_srm.geospatial.geotiff import read_geotiff, write_geotiff
from satellite_srm.models.model_factory import create_model
from satellite_srm.inference.pipeline import FullSceneSRMPipeline
from satellite_srm.device import get_device_info
from components.sidebar import render_sidebar
from components.viewer import render_comparison
from components.analytics import render_analytics
from utils.export import get_file_download_bytes

st.set_page_config(
    page_title="Satellite-SRM | NTRO PS 26142",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Deep Learning Super Resolution Mapping (SRM)")
st.caption("Target Problem Statement 26142 — National Technical Research Organisation (NTRO) | Space Technology")

params = render_sidebar(st)
device_info = get_device_info()
if device_info["cuda_available"]:
    st.sidebar.success(f"CUDA available: {device_info['device_name']}")
else:
    st.sidebar.warning("CUDA unavailable; inference will use CPU.")

# File Selection / Upload
default_scene = "data/raw/sentinel2/S2A_MSIL2A_20260515_agriculture.tif"
uploaded_files = st.file_uploader(
    "Upload Sentinel-2 GeoTIFF (one 4-band file or four band files)",
    type=["tif", "tiff"],
    accept_multiple_files=True
)

target_file = None
if uploaded_files:
    os.makedirs("data/interim", exist_ok=True)
    uploaded_rasters = []
    for uploaded_file in uploaded_files:
        uploaded_path = os.path.join("data/interim", uploaded_file.name)
        with open(uploaded_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        uploaded_rasters.append((uploaded_file.name, read_geotiff(uploaded_path)))

    if len(uploaded_rasters) == 1:
        target_file = os.path.join("data/interim", uploaded_rasters[0][0])
        st.success(f"Uploaded: {uploaded_rasters[0][0]}")
    elif len(uploaded_rasters) == 4:
        band_order = ["B02", "B03", "B04", "B08"]
        band_files = {}
        for filename, raster in uploaded_rasters:
            matches = re.findall(r"B(?:02|03|04|08)", filename.upper())
            if raster.metadata.count != 1 or not matches:
                band_files = {}
                break
            band_files[matches[0]] = (filename, raster)

        if set(band_files) != set(band_order):
            st.error("When uploading four files, name them with band codes B02, B03, B04, and B08.")
            st.stop()

        first_raster = band_files[band_order[0]][1]
        for band_name in band_order[1:]:
            raster = band_files[band_name][1]
            if (raster.data.shape[1:] != first_raster.data.shape[1:] or
                    raster.metadata.transform != first_raster.metadata.transform or
                    raster.metadata.crs != first_raster.metadata.crs):
                st.error("The four band files must have matching dimensions, CRS, and geospatial transform.")
                st.stop()

        combined_path = os.path.join("data/interim", "combined_sentinel2_4band.tif")
        combined_data = np.concatenate([band_files[band_name][1].data for band_name in band_order], axis=0)
        first_raster.metadata.band_names = band_order
        first_raster.metadata.count = 4
        write_geotiff(combined_path, combined_data, first_raster.metadata)
        target_file = combined_path
        st.success("Uploaded and combined B02, B03, B04, and B08 into a 4-band scene.")
    else:
        st.error("Upload either one 4-band TIFF or exactly four single-band TIFFs.")
        st.stop()
elif os.path.exists(default_scene):
    target_file = default_scene
    st.info(f"Using default sample scene: {default_scene}")

if target_file and os.path.exists(target_file):
    lr_raster = read_geotiff(target_file)
    meta = lr_raster.metadata

    if meta.count != 4:
        st.error(
            f"This pipeline requires a 4-band Sentinel-2 GeoTIFF (B02, B03, B04, B08), "
            f"but this file contains {meta.count} band{'s' if meta.count != 1 else ''}."
        )
        st.info("Upload the original 4-band multispectral TIFF instead of a single-band or RGB export.")
        st.stop()

    # Display Metadata
    st.markdown("### Scene Metadata & Spatial Properties")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Dimensions", f"{meta.width} × {meta.height}")
    mcol2.metric("Band Count", f"{meta.count} ({', '.join(meta.band_names)})")
    mcol3.metric("Input GSD", f"{meta.gsd_x:.1f} m")
    mcol4.metric("CRS", meta.crs.to_string() if hasattr(meta.crs, "to_string") else str(meta.crs))

    if st.button("🚀 Run Super-Resolution Pipeline", type="primary"):
        with st.spinner("Processing full-scene tiled inference with overlap blending & MC uncertainty..."):
            cfg = load_config()
            cfg["inference"]["tile_size"] = params["tile_size"]
            cfg["inference"]["overlap"] = params["overlap"]
            cfg["inference"]["scale_factor"] = params["scale_factor"]
            cfg["inference"]["uncertainty"]["mc_passes"] = params["mc_passes"]
            cfg["device"] = {
                "CUDA GPU": "cuda",
                "Auto Detect": "auto",
                "CPU": "cpu"
            }[params["device"]]

            model = create_model(cfg)
            pipeline = FullSceneSRMPipeline(model, config=cfg)
            st.info(f"Inference device: `{pipeline.device}`")

            sr_out = "outputs/sr/SR_product.tif"
            unc_out = "outputs/uncertainty/uncertainty_map.tif"
            res = pipeline.run(target_file, sr_out, unc_out)

            st.success(f"Inference completed in {res['runtime_seconds']:.2f} seconds! Reconstructed GSD: {res['target_gsd']:.2f}m")

            sr_raster = read_geotiff(sr_out)
            unc_raster = read_geotiff(unc_out)

            # Render visual comparison
            render_comparison(st, lr_raster.data, sr_raster.data, unc_raster.data)

            # Render application analytics
            render_analytics(st, sr_raster.data)

            # Download buttons
            st.markdown("### Export Geospatial Deliverables (QGIS Compatible)")
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                st.download_button(
                    "💾 Download Sub-4m GeoTIFF (SR_product.tif)",
                    data=get_file_download_bytes(sr_out),
                    file_name="SR_product.tif",
                    mime="image/tiff"
                )
            with dcol2:
                st.download_button(
                    "💾 Download Uncertainty GeoTIFF (uncertainty_map.tif)",
                    data=get_file_download_bytes(unc_out),
                    file_name="uncertainty_map.tif",
                    mime="image/tiff"
                )
