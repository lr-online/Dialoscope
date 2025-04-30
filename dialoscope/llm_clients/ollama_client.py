import ollama
from typing import List, Dict, Any, Iterator

from .base import LLMClient

class OllamaClient(LLMClient):
    """Client for interacting with Ollama."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the Ollama client using configuration."""
        self.model = config.get("model", "llama3") # Default to llama3 if not specified
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.timeout = config.get("timeout") # Optional timeout

        # The ollama library uses OLLAMA_HOST environment variable or the host parameter.
        # We create a client instance to manage connection settings.
        client_options = {'host': self.base_url}
        if self.timeout is not None:
             client_options['timeout'] = self.timeout
        self.client = ollama.Client(**client_options)
        
        # Optional: Verify connection or model availability here?
        # try:
        #     self.client.list()
        # except Exception as e:
        #     print(f"Warning: Could not connect to Ollama at {self.base_url}: {e}")
        
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a complete response using the Ollama API."""
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
            )
            # print(f"\nDEBUG Ollama Response: {response}\n") # Keep for potential debugging
            content = response['message']['content']
            return content if content else ""
        except Exception as e:
            # TODO: Add more robust error handling
            print(f"Error calling Ollama API (non-stream): {e}")
            # In case of API error, return a predefined message or raise exception
            return f"[Error generating response from Ollama ({self.model})]" 

    def generate_response_stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        """Generate a response stream using the Ollama API."""
        try:
            stream = self.client.chat(
                model=self.model,
                messages=messages,
                stream=True
            )
            for chunk in stream:
                # Ensure the chunk and message content exist before accessing
                if chunk and 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    if content:
                        yield content
                # Optional: Handle other parts of the chunk if needed, e.g., final response stats
                # if chunk.get('done'):
                #     final_data = chunk # Contains total_duration, prompt_eval_count, etc.
                #     # print(f"\nStream finished. Stats: {final_data}")

        except Exception as e:
            print(f'Error calling Ollama API (stream): {e}')
            yield f'[Error generating streaming response from Ollama ({self.model})]' 