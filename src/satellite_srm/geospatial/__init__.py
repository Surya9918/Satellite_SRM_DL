"""Geospatial data structures, metadata tracking, and GeoTIFF I/O."""
from satellite_srm.geospatial.metadata import GeoMetadata
from satellite_srm.geospatial.crs import CRSManager
from satellite_srm.geospatial.raster import GeoRaster
from satellite_srm.geospatial.geotiff import read_geotiff, write_geotiff

__all__ = ["GeoMetadata", "CRSManager", "GeoRaster", "read_geotiff", "write_geotiff"]
