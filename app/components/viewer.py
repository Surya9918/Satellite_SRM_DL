"""Synchronized visualizer for original LR 10m, reconstructed SR <4m, and uncertainty maps."""
import numpy as np
import cv2

def render_comparison(st, lr_data: np.ndarray, sr_data: np.ndarray, unc_data: np.ndarray):
    st.subheader("Visual Inspection: Original 10m vs. Reconstructed Sub-4m vs. Uncertainty")
    col1, col2, col3 = st.columns(3)

    # RGB Composite: Red (idx 2), Green (idx 1), Blue (idx 0)
    lr_rgb = np.stack([lr_data[2], lr_data[1], lr_data[0]], axis=-1)
    lr_rgb = cv2.normalize(lr_rgb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    sr_rgb = np.stack([sr_data[2], sr_data[1], sr_data[0]], axis=-1)
    sr_rgb = cv2.normalize(sr_rgb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    unc_norm = cv2.normalize(unc_data[0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    unc_colored = cv2.applyColorMap(unc_norm, cv2.COLORMAP_MAGMA)
    unc_rgb = cv2.cvtColor(unc_colored, cv2.COLOR_BGR2RGB)

    with col1:
        st.markdown("**Original Sentinel-2 (10 m)**")
        st.image(lr_rgb, use_container_width=True)

    with col2:
        st.markdown("**Super-Resolved Product (<4 m)**")
        st.image(sr_rgb, use_container_width=True)

    with col3:
        st.markdown("**Uncertainty Map (StdDev)**")
        st.image(unc_rgb, use_container_width=True)
