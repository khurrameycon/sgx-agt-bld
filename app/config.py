# Update this in app/config.py

import os
from pydantic import BaseModel
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class HuggingFaceModel(BaseModel):
    """Configuration for a Hugging Face model."""
    id: str
    name: str
    max_length: int
    description: str

class OllamaModel(BaseModel):
    """Configuration for an Ollama model."""
    id: str
    name: str
    max_length: int
    description: str

class Config(BaseModel):
    """Application configuration."""
    # API Settings
    huggingface_token: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")
    
    # Ollama settings
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    use_ollama_for_agents: bool = os.getenv("USE_OLLAMA_FOR_AGENTS", "true").lower() == "true"
    
    # Available Hugging Face models
    available_models: List[HuggingFaceModel] = [
        # General purpose models
        HuggingFaceModel(
            id="mistralai/Mistral-7B-Instruct-v0.2",
            name="Mistral 7B Instruct v0.2",
            max_length=4096,
            description="Powerful 7B parameter instruct model"
        ),
        HuggingFaceModel(
            id="meta-llama/Llama-2-7b-chat-hf",
            name="Llama 2 7B Chat",
            max_length=4096,
            description="Meta's 7B parameter chat model"
        ),
        HuggingFaceModel(
            id="tiiuae/falcon-7b-instruct",
            name="Falcon 7B Instruct",
            max_length=2048,
            description="TII's 7B parameter instruction-tuned model"
        ),
        HuggingFaceModel(
            id="HuggingFaceH4/zephyr-7b-beta",
            name="Zephyr 7B Beta",
            max_length=2048,
            description="Alignment-focused 7B parameter model"
        ),
        HuggingFaceModel(
            id="bigscience/bloom-7b1",
            name="BLOOM 7B",
            max_length=2048,
            description="Multilingual 7B parameter model"
        ),
    ]
    
    # Available Ollama models
    available_ollama_models: List[OllamaModel] = [
        OllamaModel(
            id="deepseek-r1:14b",
            name="DeepSeek R1 14B",
            max_length=4096,
            description="DeepSeek R1 14B model for general tasks"
        ),
        OllamaModel(
            id="llama2:7b",
            name="Llama 2 7B",
            max_length=4096,
            description="Meta's Llama 2 7B model"
        ),
        OllamaModel(
            id="mistral:7b",
            name="Mistral 7B",
            max_length=4096,
            description="Mistral 7B model"
        ),
        OllamaModel(
            id="codellama:7b",
            name="CodeLlama 7B",
            max_length=4096,
            description="Specialized for code generation"
        ),
        OllamaModel(
            id="gemma:7b",
            name="Gemma 7B",
            max_length=4096,
            description="Google's Gemma 7B model"
        ),
    ]
    
    # Default models
    default_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"
    default_ollama_model_id: str = "deepseek-r1:14b"
    
    # CrewAI settings
    max_agents: int = 5
    default_timeout_minutes: int = 10
    
    # Tool configuration
    tool_configs: Dict = {
        "web_search": {
            "enabled": True,
        },
        "document_analysis": {
            "enabled": True,
        },
        "data_processing": {
            "enabled": True,
        },
        "text_generation": {
            "enabled": True,
        },
        "content_formatting": {
            "enabled": True,
        },
        "text_processing": {
            "enabled": True,
        }
    }

# Create global config instance
config = Config()

def get_config() -> Config:
    """Get the application configuration."""
    return config