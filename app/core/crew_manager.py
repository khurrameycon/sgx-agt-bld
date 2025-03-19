# app/core/crew_manager.py - Updated version with markdown output support
import time
import threading
import logging
from typing import List, Dict, Any, Optional
from crewai import Agent, Task, Crew, Process, LLM
from app.models import TaskAnalysis, TaskStatus
from app.config import get_config
from app.core.ollama_provider import OllamaProvider
from app.core.agent_repository import AgentRepository

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrewManager:
    """Manages CrewAI crews and task execution."""
    
    def __init__(self):
        """Initialize the crew manager."""
        # Dictionary to store task information
        self.tasks = {}
        # Get configuration
        self.config = get_config()
        # Initialize agent repository
        self.agent_repository = AgentRepository()
    
    def create_crew(self, agents: List[Agent], task_analysis: TaskAnalysis, save_agents: bool = True) -> Crew:
        """
        Create a CrewAI crew with the specified agents and configuration.
        
        Args:
            agents: List of configured CrewAI agents
            task_analysis: The analysis of the user task
            save_agents: Whether to save the agents to the repository
            
        Returns:
            Configured CrewAI Crew object
        """
        logger.info(f"Creating crew for task {task_analysis.task_id}")
        
        # Save agents if requested
        if save_agents:
            collection_name = task_analysis.task_id
            for agent in agents:
                self.agent_repository.save_agent(agent, collection=collection_name)
            
            # Also save the crew configuration
            agent_roles = [agent.role for agent in agents]
            self.agent_repository.save_crew(
                crew_name=f"Crew for {task_analysis.task_id}",
                agent_roles=agent_roles,
                workflow_type=task_analysis.workflow_type,
                task_template=f"Task for {task_analysis.task_id}"
            )
        
        # Create tasks for each agent based on their roles and the dependencies
        tasks = self._create_tasks(agents, task_analysis)
        
        # Determine the optimal process based on the workflow type
        process = self._get_process_for_workflow(task_analysis.workflow_type)
        
        # Configure the crew with the LLM if using Ollama
        try:
            if self.config.use_ollama_for_agents:
                # Create LLM instance for agents
                model_name = f"ollama/{self.config.default_ollama_model_id}"
                llm = LLM(
                    model=model_name,
                    base_url=self.config.ollama_base_url,
                    api_type="ollama"
                )
                
                # Create the crew with the Ollama LLM
                crew = Crew(
                    agents=agents,
                    tasks=tasks,
                    process=process,
                    verbose=True,
                    llm=llm
                )
                logger.info(f"Created crew with Ollama LLM: {model_name}")
            else:
                # Create the crew without specifying the LLM (will use OpenAI default)
                crew = Crew(
                    agents=agents,
                    tasks=tasks,
                    process=process,
                    verbose=True
                )
                logger.info("Created crew with default LLM (requires OpenAI API key)")
                
            return crew
            
        except Exception as e:
            logger.error(f"Error creating crew: {e}")
            # Create the crew without specifying the LLM as a fallback
            crew = Crew(
                agents=agents,
                tasks=tasks,
                process=process,
                verbose=True
            )
            logger.warning("Created crew with default LLM due to error (requires OpenAI API key)")
            return crew
    
    def start_task(self, task_id: str, crew: Crew, timeout_minutes: int = 10):
        """
        Start the crew working on the task in a background thread.
        
        Args:
            task_id: The ID of the task
            crew: The configured CrewAI crew
            timeout_minutes: Maximum execution time in minutes
        """
        logger.info(f"Starting task {task_id}")
        
        # Set up task tracking
        self.tasks[task_id] = {
            "start_time": time.time(),
            "timeout": timeout_minutes * 60,  # Convert to seconds
            "crew": crew,
            "progress": 0.6,  # We're already at 60% progress (analysis and agent creation)
        }
        
        # Start the crew in a background thread
        thread = threading.Thread(target=self._run_crew, args=(task_id, crew))
        thread.daemon = True
        thread.start()
    
    def get_elapsed_time(self, task_id: str) -> str:
        """
        Get the elapsed time for a task.
        
        Args:
            task_id: The ID of the task
            
        Returns:
            Formatted elapsed time string
        """
        if task_id not in self.tasks or "start_time" not in self.tasks[task_id]:
            return "0 seconds"
        
        elapsed_seconds = time.time() - self.tasks[task_id]["start_time"]
        
        # Format the elapsed time
        if elapsed_seconds < 60:
            return f"{elapsed_seconds:.1f} seconds"
        elif elapsed_seconds < 3600:
            minutes = elapsed_seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = elapsed_seconds / 3600
            return f"{hours:.1f} hours"
    
    def _run_crew(self, task_id: str, crew: Crew):
        """
        Run the crew and update task status.
        
        Args:
            task_id: The ID of the task
            crew: The CrewAI crew to run
        """
        from app import main  # Import here to avoid circular imports
        
        try:
            logger.info(f"Running crew for task {task_id}")
            
            # Update progress
            self.tasks[task_id]["progress"] = 0.7
            main.active_tasks[task_id]["progress"] = 0.7
            
            try:
                # Run the crew
                result = crew.kickoff()
                
                # Update progress and status
                self.tasks[task_id]["progress"] = 1.0
                main.active_tasks[task_id]["progress"] = 1.0
                main.active_tasks[task_id]["status"] = TaskStatus.COMPLETED
                main.active_tasks[task_id]["result"] = result
                
                # Save the result as a markdown file
                if result:
                    # Format the result as markdown if it's not already
                    markdown_result = self._format_result_as_markdown(result, task_id)
                    
                    # Save to the repository
                    result_path = self.agent_repository.save_result(markdown_result, task_id)
                    
                    # Add the file path to the task info
                    main.active_tasks[task_id]["result_file"] = result_path
                
                logger.info(f"Task {task_id} completed successfully")
                
            except Exception as e:
                logger.error(f"Error running crew for task {task_id}: {e}")
                
                # Update status to error
                main.active_tasks[task_id]["status"] = TaskStatus.FAILED
                main.active_tasks[task_id]["error"] = str(e)
            
        except Exception as e:
            logger.error(f"Unexpected error in task {task_id}: {e}")
            
            # Update status to error
            main.active_tasks[task_id]["status"] = TaskStatus.FAILED
            main.active_tasks[task_id]["error"] = f"Unexpected error: {str(e)}"
    
    def _create_tasks(self, agents: List[Agent], task_analysis: TaskAnalysis) -> List[Task]:
        """
        Create tasks for each agent based on the task analysis.
        
        Args:
            agents: List of configured CrewAI agents
            task_analysis: The analysis of the user task
            
        Returns:
            List of configured CrewAI Task objects
        """
        # Create a mapping of agent roles to agent objects
        agent_map = {agent.role: agent for agent in agents}
        tasks = []
        
        # Create a task for each agent
        for agent in agents:
            # Get the agent's dependencies
            dependent_roles = task_analysis.dependencies.get(agent.role, [])
            dependent_agents = [agent_map[role] for role in dependent_roles if role in agent_map]
            
            # Create the task description
            task_description = f"""
            Complete your role as {agent.role} according to your goal: {agent.goal}.
            
            You are working on the following task:
            {task_analysis.task_id}
            
            Use your skills and tools to accomplish your goal. Be thorough and detailed in your work.
            """
            
            # Create the task
            task = Task(
                description=task_description,
                agent=agent,
                expected_output=f"Complete analysis and results from {agent.role}",
                async_execution=task_analysis.workflow_type == "parallel"
            )
            
            tasks.append(task)
        
        return tasks
    
    def _get_process_for_workflow(self, workflow_type: str) -> Process:
        """
        Determine the appropriate CrewAI process based on the workflow type.
        
        Args:
            workflow_type: The type of workflow ("sequential", "parallel", or "hierarchical")
            
        Returns:
            CrewAI Process enum value
        """
        if workflow_type == "sequential":
            return Process.sequential
        elif workflow_type == "parallel":
            return Process.sequential  # CrewAI handles parallel tasks within sequential process
        elif workflow_type == "hierarchical":
            return Process.hierarchical
        else:
            # Default to sequential
            return Process.sequential
    
    def _format_result_as_markdown(self, result: Any, task_id: str) -> str:
        """
        Format the result as a markdown document.
        
        Args:
            result: The result from the crew
            task_id: The ID of the task
            
        Returns:
            Formatted markdown string
        """
        # If the result is already a string, check if it has markdown formatting
        if isinstance(result, str):
            # Check if it already has markdown formatting
            if result.startswith('#') or '**' in result or '*' in result:
                # Add a header if not already present
                if not result.startswith('# '):
                    result = f"# Task Result: {task_id}\n\n{result}"
                return result
            else:
                # Add markdown formatting
                return f"# Task Result: {task_id}\n\n{result}"
        
        # For dictionary results (from CrewAI tasks_output)
        elif isinstance(result, dict) and "tasks_output" in result:
            markdown = f"# Task Result: {task_id}\n\n"
            
            # Add task outputs
            for i, task_output in enumerate(result["tasks_output"]):
                agent_name = task_output.get("agent", f"Agent {i+1}")
                markdown += f"## {agent_name}\n\n"
                
                # Add task description
                if "description" in task_output:
                    markdown += f"**Task:** {task_output['description'].strip()}\n\n"
                
                # Add raw output
                if "raw" in task_output:
                    markdown += f"{task_output['raw']}\n\n"
                    
            return markdown
        
        # For other types of results
        else:
            return f"# Task Result: {task_id}\n\n```\n{str(result)}\n```"