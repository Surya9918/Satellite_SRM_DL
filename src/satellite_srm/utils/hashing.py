"""File hashing and checksum helpers."""
from satellite_srm.acquisition.checksums import compute_sha256, verify_checksum

def hash_file(filepath: str) -> str:
    return compute_sha256(filepath)
