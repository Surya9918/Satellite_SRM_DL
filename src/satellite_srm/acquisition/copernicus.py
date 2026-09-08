"""Copernicus Data Space Ecosystem (CDSE) client authentication and catalog query."""
import os
import requests
from typing import Optional, Dict, Any, List
from satellite_srm.logging_config import get_logger

logger = get_logger("copernicus")

class CopernicusClient:
    """Authenticates with CDSE Keycloak and queries the OData product catalogue."""

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None,
                 client_id: Optional[str] = None, client_secret: Optional[str] = None):
        self.username = username or os.environ.get("CDSE_USERNAME")
        self.password = password or os.environ.get("CDSE_PASSWORD")
        self.client_id = client_id or os.environ.get("CDSE_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("CDSE_CLIENT_SECRET")
        self.token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        self.odata_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        self._access_token: Optional[str] = None

    def authenticate(self) -> bool:
        if not (self.username and self.password) and not (self.client_id and self.client_secret):
            logger.info("CDSE credentials not configured in environment (CDSE_USERNAME/CDSE_PASSWORD).")
            return False

        payload = {"grant_type": "password", "client_id": "cdse-public"}
        if self.username and self.password:
            payload.update({"username": self.username, "password": self.password})
        elif self.client_id and self.client_secret:
            payload.update({
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret
            })

        try:
            resp = requests.post(self.token_url, data=payload, timeout=15)
            if resp.status_code == 200:
                self._access_token = resp.json().get("access_token")
                logger.info("Authenticated successfully with Copernicus Data Space Ecosystem.")
                return True
            else:
                logger.warning(f"Authentication failed: HTTP {resp.status_code} - {resp.text}")
                return False
        except Exception as e:
            logger.warning(f"Error connecting to CDSE token endpoint: {e}")
            return False

    def search_scenes(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        max_cloud_cover: float = 15.0
    ) -> List[Dict[str, Any]]:
        """
        Searches Sentinel-2 L2A scenes by AOI bbox [min_lon, min_lat, max_lon, max_lat] and date range.
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        poly_str = f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
        filter_query = (
            f"Collection/Name eq 'SENTINEL-2' and "
            f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;{poly_str}') and "
            f"ContentDate/Start gt {start_date}T00:00:00.000Z and "
            f"ContentDate/Start lt {end_date}T23:59:59.999Z and "
            f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {max_cloud_cover})"
        )
        url = f"{self.odata_url}?$filter={filter_query}&$top=10&$orderby=ContentDate/Start desc"
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code == 200:
                return resp.json().get("value", [])
            logger.warning(f"CDSE search returned HTTP {resp.status_code}")
            return []
        except Exception as e:
            logger.warning(f"CDSE search request error: {e}")
            return []
