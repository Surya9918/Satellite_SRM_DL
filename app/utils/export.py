"""Export helpers for downloading GeoTIFFs and analytics reports from Streamlit."""
import os
import io
import json

def get_file_download_bytes(filepath: str) -> bytes:
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return f.read()
    return b""
