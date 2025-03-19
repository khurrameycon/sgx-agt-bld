# app/core/task_analyzer.py
import uuid
import logging
from typing import Dict, Any, List
from app.models import TaskAnalysis, AgentSpec, WorkflowType
from app.core.llm_provider import LLMProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskAnalyzer:
    """Analyzes user prompts to determine agent requirements."""
    
    def __init__(self, llm_provider: LLMProvider):
        """
        Initialize the task analyzer.
        
        Args:
            llm_provider: Provider for LLM inference
        """
        self.llm_provider = llm_provider
    
    async def analyze(self, prompt: str) -> TaskAnalysis:
        """
        Analyze the user prompt and determine the optimal agent configuration.
        
        Args:
            prompt: The user's task description
            
        Returns:
            TaskAnalysis object with agent specifications
        """
        try:
            # Get analysis from the LLM
            analysis_result = await self.llm_provider.analyze_task(prompt)
            logger.info(f"Received analysis from LLM: {analysis_result}")
            
            # Process the analysis result
            # Make sure workflow_type is one of the allowed values
            if "workflow_type" in analysis_result:
                if analysis_result["workflow_type"].lower() not in ["sequential", "parallel", "hierarchical"]:
                    analysis_result["workflow_type"] = "sequential"
                analysis_result["workflow_type"] = analysis_result["workflow_type"].lower()
            else:
                analysis_result["workflow_type"] = "sequential"
            
            # Ensure task_id is present
            if "task_id" not in analysis_result:
                analysis_result["task_id"] = f"task_{uuid.uuid4().hex[:8]}"
            
            # Validate required_agents
            if "required_agents" not in analysis_result or not analysis_result["required_agents"]:
                analysis_result["required_agents"] = self._generate_default_agents()
            
            # Validate dependencies
            if "dependencies" not in analysis_result:
                analysis_result["dependencies"] = self._generate_default_dependencies(analysis_result["required_agents"])
            
            # Convert to TaskAnalysis object
            return TaskAnalysis(**analysis_result)
            
        except Exception as e:
            logger.error(f"Error analyzing task: {e}")
            # Return a default analysis in case of error
            return self._generate_default_analysis()
    
    def _generate_default_analysis(self) -> TaskAnalysis:
        """
        Generate a default analysis for general tasks.
        
        Returns:
            Default TaskAnalysis object
        """
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        # Create default agent specs
        required_agents = self._generate_default_agents()
        
        # Create default dependencies
        dependencies = self._generate_default_dependencies(required_agents)
        
        return TaskAnalysis(
            task_id=task_id,
            complexity=4,
            estimated_time="5-10 minutes",
            required_agents=required_agents,
            workflow_type=WorkflowType.SEQUENTIAL,
            dependencies=dependencies
        )
    
    def _generate_default_agents(self) -> List[AgentSpec]:
        """
        Generate default agent specifications.
        
        Returns:
            List of default AgentSpec objects
        """
        return [
            AgentSpec(
                role="Task Manager",
                goal="Manage and execute the requested task efficiently",
                backstory="A versatile agent capable of handling diverse requests",
                tools=["web_search", "text_processing"],
                allow_delegation=True
            ),
            AgentSpec(
                role="Content Creator",
                goal="Generate high-quality content based on requirements",
                backstory="A creative agent with strong writing and synthesis abilities",
                tools=["text_generation", "content_formatting"],
                allow_delegation=False
            )
        ]
    
    def _generate_default_dependencies(self, agents: List[AgentSpec]) -> Dict[str, List[str]]:
        """
        Generate default dependencies for agents.
        
        Args:
            agents: List of agent specifications
            
        Returns:
            Dictionary of dependencies
        """
        dependencies = {}
        
        # Add each agent to the dependencies dict
        for i, agent in enumerate(agents):
            if i == 0:
                # First agent has no dependencies
                dependencies[agent.role] = []
            else:
                # Subsequent agents depend on the previous agent
                dependencies[agent.role] = [agents[i-1].role]
        
        return dependencies