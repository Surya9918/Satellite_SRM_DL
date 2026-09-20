"""
Hierarchical configuration manager loading and validating YAML settings with environment variable overrides.
"""

import os
import yaml
from typing import Any, Dict, Optional
import dotenv

dotenv.load_dotenv()


class ConfigDict(dict):
    """Dictionary subclass enabling attribute-style dot access."""
    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict) and not isinstance(val, ConfigDict):
                val = ConfigDict(val)
                self[key] = val
            return val
        except KeyError:
            raise AttributeError(f"Configuration key '{key}' not found.")

    def __setattr__(self, key: str, value: Any):
        self[key] = value

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges override dictionary into base dictionary."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged

def load_config(config_path: Optional[str] = None, base_dir: str = "configs") -> ConfigDict:
    """
    Loads default configuration and layers specific configuration overrides.
    """
    default_path = os.path.join(base_dir, "default.yaml")
    cfg: Dict[str, Any] = {}

    if os.path.exists(default_path):
        with open(default_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # Load all modular configs from configs directory if present
    for mod in ["data", "preprocessing", "model", "training", "inference", "validation", "application"]:
        mod_path = os.path.join(base_dir, f"{mod}.yaml")
        if os.path.exists(mod_path):
            with open(mod_path, "r", encoding="utf-8") as f:
                mod_data = yaml.safe_load(f) or {}
                cfg[mod] = mod_data

    # Load explicit config override file
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            override_data = yaml.safe_load(f) or {}
            cfg = deep_merge(cfg, override_data)

    # Environment variable overrides
    if "DEVICE" in os.environ:
        cfg["device"] = os.environ["DEVICE"]
    if "LOG_LEVEL" in os.environ:
        cfg["log_level"] = os.environ["LOG_LEVEL"]

    return ConfigDict(cfg)
