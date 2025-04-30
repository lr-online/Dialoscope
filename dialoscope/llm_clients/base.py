from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LLMClient(ABC):
    """Abstract base class for Large Language Model clients."""

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """Initialize the client with provider-specific configuration."""
        pass

    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response based on the conversation history."""
        pass 