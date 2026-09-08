"""
Satellite-SRM: Deep Learning Based Super Resolution Mapping from Medium Resolution Satellite Imageries
Problem Statement ID: 26142 (NTRO)
"""

__version__ = "1.0.0"
__author__ = "National Technical Research Organisation (NTRO)"

import sys
import os
import importlib.util
from importlib.abc import MetaPathFinder

# Custom MetaPathFinder to ensure __main__ and cli are resolvable on any filesystem cache
class _SRMPackageFinder(MetaPathFinder):
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname.startswith("satellite_srm."):
            mod_name = fullname.split(".", 1)[1]
            pkg_dir = os.path.dirname(__file__)
            target_file = os.path.join(pkg_dir, f"{mod_name}.py")
            if os.path.isfile(target_file):
                return importlib.util.spec_from_file_location(fullname, target_file)
        return None

if not any(isinstance(finder, type) and finder.__name__ == "_SRMPackageFinder" for finder in sys.meta_path):
    sys.meta_path.insert(0, _SRMPackageFinder)

from satellite_srm.config import load_config
from satellite_srm.logging_config import get_logger

__all__ = ["load_config", "get_logger", "__version__"]
