# app/models.py
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from enum import Enum

class TaskStatus(str, Enum):
    """Status of a task."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    BUILDING = "building_agents"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class AgentSpec(BaseModel):
    """Specification for an agent."""
    role: str
    goal: str
    backstory: str
    tools: List[str] = []
    allow_delegation: bool = True
    
    model_config = {
        'protected_namespaces': ()
    }

class WorkflowType(str, Enum):
    """Type of agent workflow."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"

class TaskAnalysis(BaseModel):
    """Analysis of a user task."""
    task_id: str
    complexity: int = Field(..., ge=1, le=10)
    estimated_time: str
    required_agents: List[AgentSpec]
    workflow_type: WorkflowType
    dependencies: Dict[str, List[str]] = {}
    
    model_config = {
        'protected_namespaces': ()
    }

class TaskRequest(BaseModel):
    """Request to create a new task."""
    prompt: str
    model_id: Optional[str] = None
    ollama_model_id: Optional[str] = None
    max_agents: Optional[int] = None
    timeout_minutes: Optional[int] = None
    
    model_config = {
        'protected_namespaces': ()
    }

class TaskResponse(BaseModel):
    """Response to a task creation request."""
    task_id: str
    agents_created: int
    status: TaskStatus
    estimated_completion_time: str
    
    model_config = {
        'protected_namespaces': ()
    }

class TaskStatusResponse(BaseModel):
    """Response to a task status request."""
    task_id: str
    status: TaskStatus
    elapsed_time: str
    progress: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    
    model_config = {
        'protected_namespaces': ()
    }

class ModelListResponse(BaseModel):
    """Response to a model list request."""
    models: List[Dict[str, str]]
    default_model_id: str
    
    model_config = {
        'protected_namespaces': ()
    }