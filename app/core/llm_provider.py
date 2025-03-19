# app/core/llm_provider.py
import json
import requests
import logging
from typing import Dict, Any, Optional, List, Union
from app.config import get_config, HuggingFaceModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider:
    """Provider for LLM inference using Hugging Face."""
    
    def __init__(self, model_id: Optional[str] = None):
        """
        Initialize the LLM provider.
        
        Args:
            model_id: ID of the model to use. If None, the default model will be used.
        """
        self.config = get_config()
        self.model_id = model_id or self.config.default_model_id
        self.token = self.config.huggingface_token
        self._model = self._get_model_config(self.model_id)
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
    
    def _get_model_config(self, model_id: str) -> HuggingFaceModel:
        """
        Get the configuration for a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Model configuration
            
        Raises:
            ValueError: If the model ID is not in the available models
        """
        for model in self.config.available_models:
            if model.id == model_id:
                return model
        
        # If the model is not found, use the default model
        logger.warning(f"Model {model_id} not found in available models. Using default model.")
        for model in self.config.available_models:
            if model.id == self.config.default_model_id:
                return model
        
        # If the default model is not found either, use the first available model
        logger.warning(f"Default model {self.config.default_model_id} not found in available models. Using first available model.")
        return self.config.available_models[0]
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """
        Get the list of available models.
        
        Returns:
            List of available models
        """
        return [
            {
                "id": model.id,
                "name": model.name,
                "description": model.description
            }
            for model in self.config.available_models
        ]
    
    async def analyze_task(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze a task using the LLM.
        
        Args:
            prompt: Task prompt
            
        Returns:
            Analysis of the task
        """
        # Prepare the prompt template for task analysis
        analysis_prompt = self._prepare_analysis_prompt(prompt)
        
        # Get the response from the LLM
        response = await self._generate_text(analysis_prompt)
        
        # Extract the JSON analysis from the response
        analysis = self._extract_json_from_response(response)
        
        return analysis
    
    def _prepare_analysis_prompt(self, prompt: str) -> str:
        """
        Prepare the prompt for task analysis.
        
        Args:
            prompt: Task prompt
            
        Returns:
            Formatted prompt for the LLM
        """
        return f"""
        You are an AI assistant tasked with analyzing user requests to determine the optimal agent configuration for task execution.
        
        Analyze the following user task and determine the optimal agent configuration:
        
        USER TASK: {prompt}
        
        Your analysis should include:
        1. Task complexity (1-10 scale)
        2. Estimated completion time
        3. Necessary agent roles, with specific goals and backstories
        4. Required tools for each agent (select from: web_search, document_analysis, data_processing, text_generation, content_formatting)
        5. Optimal workflow type (sequential, parallel, or hierarchical)
        6. Dependencies between agents
        
        Provide your response in valid JSON format with the following structure:
        {{
            "task_id": "unique_id",
            "complexity": 5,
            "estimated_time": "10-15 minutes",
            "required_agents": [
                {{
                    "role": "Agent Role",
                    "goal": "Agent Goal",
                    "backstory": "Agent Backstory",
                    "tools": ["tool1", "tool2"],
                    "allow_delegation": true
                }}
            ],
            "workflow_type": "sequential", 
            "dependencies": {{
                "Agent Role": []
            }}
        }}
        
        Return ONLY the JSON with no additional text or explanation.
        """
    
    async def _generate_text(self, prompt: str) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": min(1024, self._model.max_length),
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        try:
            logger.info(f"Sending request to Hugging Face: {self.model_id}")
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Handle different response formats from Hugging Face
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    return result[0]["generated_text"]
                else:
                    return str(result[0])
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"]
            else:
                return str(result)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Hugging Face API: {e}")
            # Return a fallback response for development purposes
            return self._fallback_analysis_response()
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return self._fallback_analysis_response()
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """
        Extract JSON from the LLM response.
        
        Args:
            response: LLM response
            
        Returns:
            Extracted JSON
        """
        try:
            # Look for JSON content in the response
            # This handles cases where the model might output additional text before or after the JSON
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx+1]
                return json.loads(json_str)
            else:
                logger.warning("Could not find JSON in the response. Using fallback response.")
                return json.loads(self._fallback_analysis_response())
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from the response. Using fallback response.")
            return json.loads(self._fallback_analysis_response())
    
    def _fallback_analysis_response(self) -> str:
        """
        Provide a fallback response for task analysis in case of API failure.
        
        Returns:
            Fallback analysis as a JSON string
        """
        import uuid
        
        fallback = {
            "task_id": f"task_{uuid.uuid4().hex[:8]}",
            "complexity": 5,
            "estimated_time": "10-15 minutes",
            "required_agents": [
                {
                    "role": "Task Coordinator",
                    "goal": "Coordinate and execute the requested task efficiently",
                    "backstory": "An experienced coordinator who excels at breaking down complex tasks",
                    "tools": ["web_search", "text_generation"],
                    "allow_delegation": True
                },
                {
                    "role": "Content Specialist",
                    "goal": "Generate high-quality content specific to the task",
                    "backstory": "A creative specialist with expertise in producing engaging content",
                    "tools": ["text_generation", "content_formatting"],
                    "allow_delegation": False
                }
            ],
            "workflow_type": "sequential",
            "dependencies": {
                "Task Coordinator": [],
                "Content Specialist": ["Task Coordinator"]
            }
        }
        
        return json.dumps(fallback)