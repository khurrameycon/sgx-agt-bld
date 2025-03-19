
# app/core/agent_builder.py
import logging
from typing import List, Dict, Any, Callable
from crewai import Agent, LLM
from langchain.tools import BaseTool
from app.models import TaskAnalysis, AgentSpec
from app.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentBuilder:
    """Builds agents based on task analysis."""
    
    def __init__(self):
        """Initialize the agent builder."""
        # Get configuration
        self.config = get_config()
        
        # Tool registry - mapping of tool names to factory functions
        self.tool_registry = {
            "web_search": self._create_web_search_tool,
            "document_analysis": self._create_document_analysis_tool,
            "data_processing": self._create_data_processing_tool,
            "text_generation": self._create_text_generation_tool,
            "content_formatting": self._create_content_formatting_tool,
            "text_processing": self._create_text_processing_tool,
        }
    
    def build_agents(self, task_analysis: TaskAnalysis, max_agents: int = 5) -> List[Agent]:
        """
        Build CrewAI agents based on the task analysis.
        
        Args:
            task_analysis: The analysis of the user task
            max_agents: Maximum number of agents to create
            
        Returns:
            List of configured CrewAI Agent objects
        """
        logger.info(f"Building agents for task {task_analysis.task_id}")
        
        # Limit the number of agents according to the max parameter
        agent_specs = task_analysis.required_agents[:max_agents]
        
        # Configure LLM if using Ollama for agents
        llm = None
        if self.config.use_ollama_for_agents:
            try:
                model_name = f"ollama/{self.config.default_ollama_model_id}"
                llm = LLM(
                    model=model_name,
                    base_url=self.config.ollama_base_url,
                    api_type="ollama"
                )
                logger.info(f"Using Ollama LLM for agents: {model_name}")
            except Exception as e:
                logger.error(f"Error creating Ollama LLM: {e}")
        
        # Create CrewAI agents from the specifications
        agents = []
        for spec in agent_specs:
            # Get the required tools for this agent
            agent_tools = self._get_tools_for_agent(spec.tools)
            
            # Create agent parameters
            agent_params = {
                "role": spec.role,
                "goal": spec.goal,
                "backstory": spec.backstory,
                "verbose": True,
                "allow_delegation": spec.allow_delegation,
                "tools": agent_tools
            }
            
            # Add LLM if configured
            if llm:
                agent_params["llm"] = llm
            
            # Create the CrewAI agent
            agent = Agent(**agent_params)
            
            logger.info(f"Created agent: {spec.role}")
            agents.append(agent)
            
        return agents
    
    def _get_tools_for_agent(self, tool_names: List[str]) -> List[BaseTool]:
        """
        Get the actual tool implementations for the specified tool names.
        
        Args:
            tool_names: List of tool identifiers
            
        Returns:
            List of actual tool implementations
        """
        tools = []
        for name in tool_names:
            if name in self.tool_registry:
                try:
                    # Call the factory function to create the tool
                    tool = self.tool_registry[name]()
                    if tool:
                        tools.append(tool)
                except Exception as e:
                    logger.error(f"Error creating {name} tool: {e}")
        
        return tools
    
    # Tool factory methods using standard LangChain tools
    
    def _create_web_search_tool(self) -> BaseTool:
        """Create a web search tool."""
        class SimpleWebSearchTool(BaseTool):
            name = "web_search"
            description = "Search the web for information"
            
            def _run(self, query: str) -> str:
                return f"Simulated web search results for: {query}"
                
            def _arun(self, query: str) -> str:
                # Async version just calls the sync version for simplicity
                return self._run(query)
                
        return SimpleWebSearchTool()
    
    def _create_document_analysis_tool(self) -> BaseTool:
        """Create a document analysis tool."""
        class DocumentAnalysisTool(BaseTool):
            name = "document_analysis"
            description = "Analyze document content"
            
            def _run(self, document: str) -> str:
                result = {
                    "summary": f"Simulated analysis of document ({len(document)} chars)",
                    "key_topics": ["topic1", "topic2", "topic3"],
                    "sentiment": "neutral"
                }
                return str(result)
                
            def _arun(self, document: str) -> str:
                return self._run(document)
                
        return DocumentAnalysisTool()
    
    def _create_data_processing_tool(self) -> BaseTool:
        """Create a data processing tool."""
        class DataProcessingTool(BaseTool):
            name = "data_processing"
            description = "Process and analyze data"
            
            def _run(self, data: str) -> str:
                result = {
                    "processed": True,
                    "summary": f"Simulated processing of data"
                }
                return str(result)
                
            def _arun(self, data: str) -> str:
                return self._run(data)
                
        return DataProcessingTool()
    
    def _create_text_generation_tool(self) -> BaseTool:
        """Create a text generation tool."""
        class TextGenerationTool(BaseTool):
            name = "text_generation"
            description = "Generate text based on prompts"
            
            def _run(self, prompt: str) -> str:
                return f"Simulated text generation for prompt: {prompt[:50]}..."
                
            def _arun(self, prompt: str) -> str:
                return self._run(prompt)
                
        return TextGenerationTool()
    
    def _create_content_formatting_tool(self) -> BaseTool:
        """Create a content formatting tool."""
        class ContentFormattingTool(BaseTool):
            name = "content_formatting"
            description = "Format content in various styles"
            
            def _run(self, content: str, format_type: str = "default") -> str:
                return f"Simulated {format_type} formatting of content ({len(content)} chars)"
                
            def _arun(self, content: str, format_type: str = "default") -> str:
                return self._run(content, format_type)
                
        return ContentFormattingTool()
    
    def _create_text_processing_tool(self) -> BaseTool:
        """Create a text processing tool."""
        class TextProcessingTool(BaseTool):
            name = "text_processing"
            description = "Process and analyze text"
            
            def _run(self, text: str) -> str:
                result = {
                    "word_count": len(text.split()),
                    "char_count": len(text),
                    "summary": f"Simulated processing of text ({len(text)} chars)"
                }
                return str(result)
                
            def _arun(self, text: str) -> str:
                return self._run(text)
                
        return TextProcessingTool()