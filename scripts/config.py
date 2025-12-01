import json
import os
from copy import deepcopy

# Default values used if config.json is missing or partial
#_DEFAULT_CONFIG = {
#    "features": {
#        "long_window": 20,
#        "short_window": 5,
#        "momentum_alpha": 1.2  # can be >1 as long as <2
#    }
#}

# Default config path: project_root/config.json
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
#DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.json")


def _deep_update(dst: dict, src: dict) -> dict:
    """Recursively update dst with src."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(path: str | None = None) -> dict:
    """
    Load config from JSON.

    If 'path' is None:
      - try DEFAULT_CONFIG_PATH
      - if it does not exist, return default config
    """
    default_path = os.path.join(_PROJECT_ROOT, "configs", "config.json")
    target_path = path or default_path

    if not os.path.isfile(target_path):
        raise FileNotFoundError(
            f"Config file not found. Looked for '{target_path}'"
        )

    with open(target_path, "r") as f:
        try:
            cfg = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in config file '{target_path}': {e}"
            ) from e

    return cfg
