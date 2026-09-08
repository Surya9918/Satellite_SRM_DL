"""Resilient chunked HTTP download engine with exponential backoff and resume support."""
import os
import time
import requests
from typing import Optional, Callable
from satellite_srm.logging_config import get_logger

logger = get_logger("downloader")

class ChunkedDownloader:
    """Downloads large satellite assets with range requests, retries, and rate limiting."""

    def __init__(self, max_retries: int = 5, backoff_factor: float = 1.5, chunk_size: int = 1048576):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.chunk_size = chunk_size

    def download_file(
        self,
        url: str,
        dest_path: str,
        headers: Optional[dict] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None
    ) -> str:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        temp_path = dest_path + ".part"
        resume_byte_pos = 0

        if os.path.exists(temp_path):
            resume_byte_pos = os.path.getsize(temp_path)

        req_headers = (headers or {}).copy()
        if resume_byte_pos > 0:
            req_headers["Range"] = f"bytes={resume_byte_pos}-"

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(url, headers=req_headers, stream=True, timeout=30)
                if response.status_code in [200, 206]:
                    total_size = int(response.headers.get("content-length", 0)) + resume_byte_pos
                    mode = "ab" if resume_byte_pos > 0 else "wb"
                    downloaded = resume_byte_pos

                    with open(temp_path, mode) as f:
                        for chunk in response.iter_content(chunk_size=self.chunk_size):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if progress_cb and total_size > 0:
                                    progress_cb(downloaded, total_size)

                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    os.rename(temp_path, dest_path)
                    logger.info(f"Successfully downloaded: {dest_path}")
                    return dest_path
                else:
                    logger.warning(f"Download HTTP {response.status_code} on attempt {attempt}/{self.max_retries}")
            except (requests.RequestException, IOError) as err:
                logger.warning(f"Download attempt {attempt} failed: {err}")

            sleep_time = self.backoff_factor ** attempt
            time.sleep(sleep_time)

        raise RuntimeError(f"Failed to download from {url} after {self.max_retries} attempts.")
