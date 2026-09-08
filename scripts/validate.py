#!/usr/bin/env python3
"""Runs validation benchmark against bicubic baseline."""
import sys
from satellite_srm.cli import main

if __name__ == "__main__":
    sys.argv = ["satellite-srm", "validate"] + sys.argv[1:]
    main()
