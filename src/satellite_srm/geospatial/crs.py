"""Coordinate Reference System management and geographic projection utilities."""
from typing import Tuple, Optional
from satellite_srm.compat import CRS

class CRSManager:
    """Handles parsing, validation, and conversion of Coordinate Reference Systems."""

    @staticmethod
    def parse_crs(crs_input: any) -> CRS:
        if isinstance(crs_input, CRS):
            return crs_input
        if isinstance(crs_input, int):
            return CRS.from_epsg(crs_input)
        if isinstance(crs_input, str):
            if crs_input.isdigit():
                return CRS.from_epsg(int(crs_input))
            return CRS.from_string(crs_input)
        return CRS.from_epsg(4326)

    @staticmethod
    def get_epsg_code(crs: CRS) -> int:
        code = crs.to_epsg()
        return code if code is not None else 4326

    @staticmethod
    def is_projected(crs: CRS) -> bool:
        epsg = CRSManager.get_epsg_code(crs)
        # EPSG 4326 is Geographic (degrees); UTM zones are typically 326xx or 327xx (meters)
        return epsg != 4326
