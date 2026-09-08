#!/usr/bin/env python3
"""Downloads or verifies pretrained deep learning weights."""
import sys
from satellite_srm.cli import main

if __name__ == "__main__":
    sys.argv = ["satellite-srm", "download-models"] + sys.argv[1:]
    main()
