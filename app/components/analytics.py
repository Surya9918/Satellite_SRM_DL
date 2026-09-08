"""Domain analytics visualizer for Agriculture (NDVI), Urban (Edges), and Disaster (NDWI)."""
import numpy as np
import cv2
from satellite_srm.applications.agriculture import AgricultureAnalyzer
from satellite_srm.applications.urban import UrbanAnalyzer
from satellite_srm.applications.disaster import DisasterAnalyzer

def render_analytics(st, sr_data: np.ndarray):
    st.subheader("Domain-Specific Application Analytics")
    tab1, tab2, tab3 = st.tabs(["🌱 Agriculture (NDVI)", "🏙️ Urban & Roads", "🌊 Flood & Disaster (NDWI)"])

    with tab1:
        agri = AgricultureAnalyzer().analyze(sr_data)
        st.metric("Mean NDVI", f"{agri['mean_ndvi']:.3f}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**NDVI Vegetation Canopy Map**")
            ndvi_norm = cv2.normalize(agri["ndvi_map"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            st.image(cv2.applyColorMap(ndvi_norm, cv2.COLORMAP_SUMMER), use_container_width=True)
        with col2:
            st.markdown("**Field Boundary Edge Extraction**")
            st.image(agri["field_boundary_edges"], use_container_width=True)

    with tab2:
        urban = UrbanAnalyzer().analyze(sr_data)
        st.metric("Structural Edge Density", f"{urban['edge_density']:.4f}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**RGB Structural View**")
            st.image(urban["rgb_composite"], use_container_width=True)
        with col2:
            st.markdown("**Extracted Road & Building Edges**")
            st.image(urban["structural_edges"], use_container_width=True)

    with tab3:
        flood = DisasterAnalyzer().analyze(sr_data)
        st.metric("Water Inundation Area", f"{flood['inundation_percentage']:.2f}%")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Normalized Difference Water Index (NDWI)**")
            ndwi_norm = cv2.normalize(flood["ndwi_map"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            st.image(cv2.applyColorMap(ndwi_norm, cv2.COLORMAP_WINTER), use_container_width=True)
        with col2:
            st.markdown("**Delineated Water Boundary Mask**")
            st.image(flood["water_boundaries"] * 255, use_container_width=True)
