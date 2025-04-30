from openai import OpenAI
from typing import List, Dict, Any, Iterator

from .base import LLMClient

class OpenAIClient(LLMClient):
    """Client for interacting with OpenAI compatible APIs."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the OpenAI client using configuration."""
        self.model = config.get("model", "gpt-4o") # Default to gpt-4o if not specified
        # api_key will be picked up from environment variable OPENAI_API_KEY by default
        # or can be passed directly in config
        # base_url can also be passed in config for OpenAI-like APIs
        self.client = OpenAI(
            api_key=config.get("api_key"), 
            base_url=config.get("base_url")
        )

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a complete response using the OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            # print(f"\nDEBUG OpenAI Response: {response}\n") # Keep for potential debugging
            content = response.choices[0].message.content
            return content if content else ""
        except Exception as e:
            # TODO: Add more robust error handling
            print(f"Error calling OpenAI API (non-stream): {e}")
            # In case of API error, return a predefined message or raise exception
            return "[Error generating response from OpenAI]"

    def generate_response_stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        """Generate a response stream using the OpenAI API."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content
        except Exception as e:
            print(f"Error calling OpenAI API (stream): {e}")
            # In case of stream error, yield an error message chunk
            yield "[Error generating streaming response from OpenAI]" 