# src/fencast/utils/paths.py

import yaml
from pathlib import Path

# Define the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Define other key directories relative to the project root
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

def load_config(config_name: str) -> dict:
    """
    Loads a YAML configuration file from the 'configs' directory.

    Args:
        config_name (str): The name of the config file (without the .yaml extension).

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    config_path = CONFIG_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)