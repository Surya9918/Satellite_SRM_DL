#!/usr/bin/env python3
"""Preprocesses scenes and extracts aligned LR-HR patch pairs."""
import sys
from satellite_srm.cli import main

if __name__ == "__main__":
    sys.argv = ["satellite-srm", "prepare-data"] + sys.argv[1:]
    main()
