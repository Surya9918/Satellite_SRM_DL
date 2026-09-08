#!/usr/bin/env python3
"""Executes full-scene tiled inference on a GeoTIFF."""
import sys
from satellite_srm.cli import main

if __name__ == "__main__":
    sys.argv = ["satellite-srm", "infer"] + sys.argv[1:]
    main()
