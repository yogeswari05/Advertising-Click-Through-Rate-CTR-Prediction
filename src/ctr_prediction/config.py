"""Configuration loader and schemas for CTR prediction."""

from pathlib import Path
from typing import Any, Dict, List
import yaml


def get_project_root() -> Path:
    """Return the root directory of the project."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load YAML configuration file.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to configuration file. If None, uses default.yaml.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = get_project_root() / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
