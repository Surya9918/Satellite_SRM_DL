"""Cryptographic verification and file integrity hashing."""
import hashlib
import os

def compute_sha256(filepath: str, block_size: int = 65536) -> str:
    """Computes SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def verify_checksum(filepath: str, expected_sha256: str) -> bool:
    """Verifies that a file matches its expected SHA-256 hash."""
    if not os.path.exists(filepath):
        return False
    actual_hash = compute_sha256(filepath)
    return actual_hash.lower() == expected_sha256.lower()
