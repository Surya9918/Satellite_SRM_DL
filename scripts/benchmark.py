#!/usr/bin/env python3
"""Runs inference speed and throughput benchmarks."""
import sys
from satellite_srm.cli import main

if __name__ == "__main__":
    sys.argv = ["satellite-srm", "benchmark"] + sys.argv[1:]
    main()
