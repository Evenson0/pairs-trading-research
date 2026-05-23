"""Configuration utilities for the pairs trading research framework."""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the configuration file is empty.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Configuration file is empty: {path}")

    return config


def get_project_root() -> Path:
    """Return the project root directory.

    This assumes the package is located under:

        project_root/src/pairs_trading_research/
    """
    return Path(__file__).resolve().parents[2]


def get_config_path(config_name: str) -> Path:
    """Return the path to a configuration file in the config directory.

    Parameters
    ----------
    config_name:
        Name of the configuration file. The `.yaml` extension is optional.

    Returns
    -------
    Path
        Full path to the requested configuration file.
    """
    if not config_name.endswith(".yaml"):
        config_name = f"{config_name}.yaml"

    return get_project_root() / "config" / config_name


def load_named_config(config_name: str) -> dict[str, Any]:
    """Load a configuration file from the project config directory.

    Examples
    --------
    >>> config = load_named_config("tsx60")
    >>> config["universe"]["name"]
    'S&P/TSX 60'
    """
    return load_config(get_config_path(config_name))
