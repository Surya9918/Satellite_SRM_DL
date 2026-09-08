#!/usr/bin/env python3
"""Executes model training loop."""
import sys
from satellite_srm.cli import main

if __name__ == "__main__":
    sys.argv = ["satellite-srm", "train"] + sys.argv[1:]
    main()
