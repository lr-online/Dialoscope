import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

CONFIG_FILE_NAME = "config.yaml"

def find_config_file() -> Optional[Path]:
    """Search for the config file upwards from the current directory."""
    current_dir = Path.cwd()
    while True:
        config_path = current_dir / CONFIG_FILE_NAME
        if config_path.is_file():
            return config_path
        if current_dir.parent == current_dir:
            # Reached root directory
            return None
        current_dir = current_dir.parent

def load_config() -> Dict[str, Any]:
    """Load configuration from the YAML file."""
    config_path = find_config_file()
    if not config_path:
        # Fallback to checking relative to this script's location if not found upwards
        script_dir = Path(__file__).parent.parent # Assumes config.py is in dialoscope/
        config_path = script_dir / CONFIG_FILE_NAME
        if not config_path.is_file():
             raise FileNotFoundError(
                f"Configuration file '{CONFIG_FILE_NAME}' not found in the current directory, parent directories, or project root."
            )

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not config:
            return {}
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        print(f"Error loading configuration: {e}")
        raise

CONFIG = load_config()

def get_provider_config(provider_name: Optional[str] = None) -> Dict[str, Any]:
    """Get configuration for a specific provider or the default provider."""
    if not CONFIG:
        raise ValueError("Configuration not loaded or empty.")

    providers = CONFIG.get('providers', {})
    if not provider_name:
        provider_name = CONFIG.get('default_provider')
        if not provider_name:
            raise ValueError("No default provider specified in configuration.")

    provider_config = providers.get(provider_name)
    if not provider_config:
        raise ValueError(f"Configuration for provider '{provider_name}' not found.")
    
    # Inject environment variables if needed (e.g., OpenAI API key)
    if provider_name == 'openai' and 'api_key' not in provider_config:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
             provider_config['api_key'] = api_key
        # If still no key, the client initialization should handle the error
        
    return provider_config 