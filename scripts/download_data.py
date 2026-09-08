#!/usr/bin/env python3
"""Acquires Sentinel-2 and WorldStrat reference scenes."""
import sys
from satellite_srm.cli import main

if __name__ == "__main__":
    sys.argv = ["satellite-srm", "download-data"] + sys.argv[1:]
    main()
