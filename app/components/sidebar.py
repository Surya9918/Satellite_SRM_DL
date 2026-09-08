"""Streamlit sidebar component configuring model, device, and tiling parameters."""
from typing import Dict, Any

def render_sidebar(st) -> Dict[str, Any]:
    st.sidebar.title("🛰️ Satellite-SRM Config")
    st.sidebar.markdown("**Problem Statement:** 26142 (NTRO)\n**Target:** Sentinel-2 (10m) &rarr; Sub-4m")

    model_type = st.sidebar.selectbox("Model Architecture", ["Multispectral SwinIR", "Lightweight Diffusion SR", "Bicubic Baseline"])
    device = st.sidebar.selectbox(
        "Device Execution Mode",
        ["CUDA GPU", "Auto Detect", "CPU"],
        index=0
    )
    scale_factor = st.sidebar.slider("Resolution Scale Factor", 2.0, 4.0, 3.0, 0.5)

    st.sidebar.subheader("Tiling & Uncertainty")
    tile_size = st.sidebar.select_slider("Inference Tile Size", options=[64, 128, 256], value=128)
    overlap = st.sidebar.slider("Tile Overlap (px)", 0, 64, 16)
    mc_passes = st.sidebar.slider("MC Uncertainty Passes", 2, 16, 2)
    blending = st.sidebar.selectbox("Blending Window", ["weighted_hann", "gaussian", "uniform"])

    st.sidebar.markdown("---")
    st.sidebar.info("Reconstructs sub-4 m spatial representation. Does not claim physically measured in-situ truth.")

    return {
        "model_type": model_type,
        "device": device,
        "scale_factor": scale_factor,
        "tile_size": tile_size,
        "overlap": overlap,
        "mc_passes": mc_passes,
        "blending": blending
    }
