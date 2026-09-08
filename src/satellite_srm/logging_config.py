"""
Structured logging configuration supporting formatted console output and file rotation.
"""

import os
import logging
import sys
from typing import Optional

def configure_logging(log_level: str = "INFO", log_file: Optional[str] = "logs/satellite_srm.log") -> logging.Logger:
    """Configures application-wide structured logging."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger("satellite_srm")
    logger.setLevel(numeric_level)
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Retrieves a named child logger under satellite_srm namespace."""
    if name:
        return logging.getLogger(f"satellite_srm.{name}")
    return logging.getLogger("satellite_srm")
