from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator

class LLMClient(ABC):
    """Abstract base class for Large Language Model clients."""

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """Initialize the client with provider-specific configuration."""
        pass

    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a complete response based on the conversation history."""
        pass

    @abstractmethod
    def generate_response_stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        """Generate a response stream based on the conversation history."""
        # This method should yield chunks of the response content as strings.
        # Example usage: for chunk in client.generate_response_stream(messages): print(chunk, end='')
        pass 