import os
import yaml
from typing import Any, Optional
from pathlib import Path

class Config:
    def __init__(self, config_file: str = 'config.yaml'):
        """Initialize configuration by loading from YAML file."""
        try:
            # If config_file is a relative path or just a filename, look for it in the same directory as this file
            if not os.path.isabs(config_file):
                current_dir = Path(__file__).parent
                config_file = os.path.join(current_dir, config_file)
            
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {config_file} not found")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a configuration value by section and key."""
        return self.config.get(section, {}).get(key, default)

# Initialize global config instance with a relative path
# This will be resolved to the same directory as this file
config = Config(config_file='config.yaml')