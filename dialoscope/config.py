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

def get_provider_details(provider_name: str) -> Dict[str, Any]:
    """Get connection details for a specific provider."""
    if not CONFIG:
        raise ValueError("Configuration not loaded or empty.")

    providers = CONFIG.get('providers', {})
    if not providers:
        raise ValueError("No 'providers' section found in configuration.")

    provider_details = providers.get(provider_name)
    if not provider_details:
        raise ValueError(f"Configuration details for provider '{provider_name}' not found.")

    # Inject environment variables if needed (e.g., OpenAI API key)
    if provider_name == 'openai' and 'api_key' not in provider_details:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
             provider_details['api_key'] = api_key
        # If still no key, the client initialization should handle the error

    return provider_details

def get_agent_config(agent_role: str) -> Dict[str, Any]:
    """
    Get the configuration for a specific agent role directly from the 'agents' section.

    Args:
        agent_role: The role of the agent (e.g., 'proponent', 'opponent', 'judge').

    Returns:
        A dictionary containing the configuration for the agent,
        including provider details merged in.

    Raises:
        ValueError: If the agent configuration or necessary keys (provider, model) are missing.
    """
    if not CONFIG:
        raise ValueError("Configuration not loaded or empty.")

    agents_config = CONFIG.get('agents', {})
    if not agents_config:
         raise ValueError("No 'agents' section found in configuration.")

    agent_config = agents_config.get(agent_role)
    if not agent_config:
        raise ValueError(f"Configuration for agent role '{agent_role}' not found.")

    # Ensure mandatory fields are present
    provider_name = agent_config.get('provider')
    model_name = agent_config.get('model')

    if not provider_name:
        raise ValueError(f"Agent '{agent_role}' configuration is missing the 'provider' key.")
    if not model_name:
        raise ValueError(f"Agent '{agent_role}' configuration is missing the 'model' key.")

    # Fetch provider connection details
    provider_details = get_provider_details(provider_name)

    # Merge agent-specific config with provider details
    # Agent config takes precedence if keys overlap (though unlikely for connection details)
    final_config = provider_details.copy()
    final_config.update(agent_config)

    return final_config