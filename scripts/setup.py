#!/usr/bin/env python3
"""One-command system setup and environment sanity check."""
import sys
from satellite_srm.cli import main

if __name__ == "__main__":
    sys.argv = ["satellite-srm", "setup"]
    main()
