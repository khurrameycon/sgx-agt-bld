# app/core/agent_repository.py
import os
import yaml
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from crewai import Agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentRepository:
    """Repository for saving and loading agents."""
    
    def __init__(self, repository_path: str = None):
        """
        Initialize the agent repository.
        
        Args:
            repository_path: Path to the repository directory
        """
        if repository_path is None:
            # Use default path in user's home directory
            home_dir = Path.home()
            repository_path = os.path.join(home_dir, ".dynamic_agent_platform", "agents")
        
        self.repository_path = repository_path
        
        # Create the repository directory if it doesn't exist
        os.makedirs(self.repository_path, exist_ok=True)
        
        # Create an agents directory for YAML files
        self.agents_path = os.path.join(self.repository_path, "agents")
        os.makedirs(self.agents_path, exist_ok=True)
        
        # Create a crews directory for saved crew configurations
        self.crews_path = os.path.join(self.repository_path, "crews")
        os.makedirs(self.crews_path, exist_ok=True)
        
        # Create a results directory for markdown outputs
        self.results_path = os.path.join(self.repository_path, "results")
        os.makedirs(self.results_path, exist_ok=True)
        
        logger.info(f"Agent repository initialized at {self.repository_path}")
    
    def save_agent(self, agent: Agent, collection: str = "default") -> str:
        """
        Save an agent to the repository.
        
        Args:
            agent: The agent to save
            collection: Collection name for grouping agents
            
        Returns:
            Path to the saved agent file
        """
        # Create collection directory if it doesn't exist
        collection_path = os.path.join(self.agents_path, collection)
        os.makedirs(collection_path, exist_ok=True)
        
        # Generate a filename based on role and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_role = agent.role.replace(" ", "_").lower()
        filename = f"{safe_role}_{timestamp}.yaml"
        file_path = os.path.join(collection_path, filename)
        
        # Extract agent properties
        agent_data = {
            "role": agent.role,
            "goal": agent.goal,
            "backstory": agent.backstory,
            "allow_delegation": agent.allow_delegation,
            "verbose": agent.verbose,
            "tools": [tool.name for tool in agent.tools] if agent.tools else [],
            "created_at": timestamp
        }
        
        # Save to YAML file
        with open(file_path, 'w') as file:
            yaml.dump(agent_data, file, default_flow_style=False)
        
        logger.info(f"Agent {agent.role} saved to {file_path}")
        return file_path
    
    def save_crew(self, crew_name: str, agent_roles: List[str], workflow_type: str, task_template: str) -> str:
        """
        Save a crew configuration.
        
        Args:
            crew_name: Name of the crew
            agent_roles: List of agent roles in the crew
            workflow_type: Type of workflow (sequential, parallel, hierarchical)
            task_template: Template for tasks to be performed by this crew
            
        Returns:
            Path to the saved crew file
        """
        # Generate a filename based on crew name and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = crew_name.replace(" ", "_").lower()
        filename = f"{safe_name}_{timestamp}.yaml"
        file_path = os.path.join(self.crews_path, filename)
        
        # Create crew data
        crew_data = {
            "name": crew_name,
            "agent_roles": agent_roles,
            "workflow_type": workflow_type,
            "task_template": task_template,
            "created_at": timestamp
        }
        
        # Save to YAML file
        with open(file_path, 'w') as file:
            yaml.dump(crew_data, file, default_flow_style=False)
        
        logger.info(f"Crew {crew_name} saved to {file_path}")
        return file_path
    
    def save_result(self, result: str, task_id: str, format: str = "markdown") -> str:
        """
        Save a task result to a file.
        
        Args:
            result: The result content
            task_id: ID of the task
            format: Output format (markdown, txt, etc.)
            
        Returns:
            Path to the saved result file
        """
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{task_id}_{timestamp}.md"
        file_path = os.path.join(self.results_path, filename)
        
        # Save the result
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(result)
        
        logger.info(f"Result for task {task_id} saved to {file_path}")
        return file_path
    
    def list_agents(self, collection: str = "default") -> List[Dict[str, Any]]:
        """
        List all agents in a collection.
        
        Args:
            collection: Collection name
            
        Returns:
            List of agent data dictionaries
        """
        collection_path = os.path.join(self.agents_path, collection)
        if not os.path.exists(collection_path):
            return []
        
        agents = []
        for filename in os.listdir(collection_path):
            if filename.endswith(".yaml"):
                file_path = os.path.join(collection_path, filename)
                try:
                    with open(file_path, 'r') as file:
                        agent_data = yaml.safe_load(file)
                        agent_data["file_path"] = file_path
                        agent_data["filename"] = filename
                        agents.append(agent_data)
                except Exception as e:
                    logger.error(f"Error loading agent file {file_path}: {e}")
        
        # Sort by creation time (newest first)
        agents.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return agents
    
    def list_crews(self) -> List[Dict[str, Any]]:
        """
        List all saved crews.
        
        Returns:
            List of crew data dictionaries
        """
        crews = []
        for filename in os.listdir(self.crews_path):
            if filename.endswith(".yaml"):
                file_path = os.path.join(self.crews_path, filename)
                try:
                    with open(file_path, 'r') as file:
                        crew_data = yaml.safe_load(file)
                        crew_data["file_path"] = file_path
                        crew_data["filename"] = filename
                        crews.append(crew_data)
                except Exception as e:
                    logger.error(f"Error loading crew file {file_path}: {e}")
        
        # Sort by creation time (newest first)
        crews.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return crews
    
    def list_results(self) -> List[Dict[str, Any]]:
        """
        List all saved results.
        
        Returns:
            List of result file information
        """
        results = []
        for filename in os.listdir(self.results_path):
            if filename.endswith(".md"):
                file_path = os.path.join(self.results_path, filename)
                file_stats = os.stat(file_path)
                
                # Extract task ID from filename
                task_id = filename.split('_')[0] if '_' in filename else "unknown"
                
                results.append({
                    "task_id": task_id,
                    "filename": filename,
                    "file_path": file_path,
                    "size": file_stats.st_size,
                    "created_at": datetime.fromtimestamp(file_stats.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
                    "modified_at": datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })
        
        # Sort by modification time (newest first)
        results.sort(key=lambda x: x.get("modified_at", ""), reverse=True)
        return results
    
    def get_agent(self, file_path: str) -> Dict[str, Any]:
        """
        Get an agent by file path.
        
        Args:
            file_path: Path to the agent file
            
        Returns:
            Agent data dictionary
        """
        try:
            with open(file_path, 'r') as file:
                agent_data = yaml.safe_load(file)
                agent_data["file_path"] = file_path
                return agent_data
        except Exception as e:
            logger.error(f"Error loading agent file {file_path}: {e}")
            return None
    
    def get_crew(self, file_path: str) -> Dict[str, Any]:
        """
        Get a crew by file path.
        
        Args:
            file_path: Path to the crew file
            
        Returns:
            Crew data dictionary
        """
        try:
            with open(file_path, 'r') as file:
                crew_data = yaml.safe_load(file)
                crew_data["file_path"] = file_path
                return crew_data
        except Exception as e:
            logger.error(f"Error loading crew file {file_path}: {e}")
            return None
    
    def get_result_content(self, file_path: str) -> str:
        """
        Get the content of a result file.
        
        Args:
            file_path: Path to the result file
            
        Returns:
            Content of the result file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading result file {file_path}: {e}")
            return f"Error: Could not read file - {str(e)}"