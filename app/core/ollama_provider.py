# app/core/ollama_provider.py
import logging
import requests
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaProvider:
    """Provider for local LLM inference using Ollama."""
    
    def __init__(self, model_name: str = "deepseek-r1:14b", base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama provider.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for the Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """
        Generate text using the local Ollama LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Generated text
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            logger.info(f"Sending request to Ollama: {self.model_name}")
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            return f"Error generating text: {str(e)}"
        
    def is_available(self) -> bool:
        """
        Check if Ollama is available.
        
        Returns:
            True if Ollama is available, False otherwise
        """
        try:
            # Try a simple request to the Ollama API
            response = requests.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False