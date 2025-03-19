# Update these imports in app/main.py
import uuid
import logging
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from fastapi import FastAPI, Request, Depends, HTTPException, Form, Query
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any, Optional
import uvicorn

from app.models import TaskRequest, TaskResponse, TaskStatusResponse, ModelListResponse, TaskStatus
from app.config import get_config
from app.core.llm_provider import LLMProvider
from app.core.ollama_provider import OllamaProvider
from app.core.task_analyzer import TaskAnalyzer
from app.core.agent_builder import AgentBuilder
from app.core.crew_manager import CrewManager
from app.core.agent_repository import AgentRepository
from app.api.routes import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Dynamic Agent Platform",
    description="A platform for building and running dynamic agent systems using CrewAI, Hugging Face, and Ollama",
    version="0.1.0",
)

# Include API routes
app.include_router(api_router, prefix="/api")

# Define templates directory
templates_path = Path(__file__).parent / "templates"
templates_path.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(templates_path))

# Define static files directory
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Store active tasks
active_tasks = {}

# Get configuration
config = get_config()

# Create agent repository
agent_repository = AgentRepository()

# Dependencies
def get_llm_provider(model_id: str = None):
    """Get the LLM provider."""
    return LLMProvider(model_id)

def get_ollama_provider(model_id: str = None):
    """Get the Ollama provider."""
    model_id = model_id or config.default_ollama_model_id
    return OllamaProvider(model_name=model_id, base_url=config.ollama_base_url)

def get_task_analyzer(llm_provider: LLMProvider = Depends(get_llm_provider)):
    """Get the task analyzer."""
    return TaskAnalyzer(llm_provider)

def get_agent_builder():
    """Get the agent builder."""
    return AgentBuilder()

def get_crew_manager():
    """Get the crew manager."""
    return CrewManager()

def get_agent_repository():
    """Get the agent repository."""
    return agent_repository

# Add these routes to app/main.py

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the index page."""
    llm_provider = LLMProvider()
    models = llm_provider.get_available_models()
    
    # Check Ollama availability
    ollama_available = False
    try:
        ollama_provider = OllamaProvider()
        ollama_available = ollama_provider.is_available()
    except:
        ollama_available = False
    
    # Get counts of saved agents and results
    saved_agents_count = len(agent_repository.list_agents())
    saved_crews_count = len(agent_repository.list_crews())
    saved_results_count = len(agent_repository.list_results())
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "models": models,
            "default_model_id": config.default_model_id,
            "ollama_models": config.available_ollama_models,
            "default_ollama_model_id": config.default_ollama_model_id,
            "max_agents": config.max_agents,
            "default_timeout": config.default_timeout_minutes,
            "ollama_available": ollama_available,
            "saved_agents_count": saved_agents_count,
            "saved_crews_count": saved_crews_count,
            "saved_results_count": saved_results_count
        },
    )

@app.get("/results_list", response_class=HTMLResponse)
async def results_list(request: Request):
    """Render the results list page."""
    results = agent_repository.list_results()
    
    return templates.TemplateResponse(
        "results_list.html",
        {
            "request": request,
            "results": results
        },
    )

@app.get("/results", response_class=HTMLResponse)
async def results(request: Request, task_id: str):
    """Render the results page."""
    if task_id not in active_tasks:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "error": "Task not found. It may have expired or been deleted.",
            },
        )
    
    task_info = active_tasks[task_id]
    
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "task_id": task_id,
            "task_info": task_info,
        },
    )

@app.get("/error", response_class=HTMLResponse)
async def error(request: Request, message: str = "An error occurred"):
    """Render the error page."""
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "error": message,
        },
    )

# Update this method in app/main.py

@app.post("/submit_task")
async def submit_task(
    request: Request,
    prompt: str = Form(...),
    model_id: str = Form(...),
    ollama_model: str = Form(...),
    max_agents: int = Form(5),
    timeout_minutes: int = Form(10),
):
    """Submit a task and redirect to the results page."""
    # Create a task request
    task_request = TaskRequest(
        prompt=prompt,
        model_id=model_id,
        ollama_model_id=ollama_model,
        max_agents=max_agents,
        timeout_minutes=timeout_minutes,
    )
    
    # Create a task
    try:
        # Create providers
        llm_provider = LLMProvider(model_id)
        
        # Create analyzer, builder, and manager
        task_analyzer = TaskAnalyzer(llm_provider)
        agent_builder = AgentBuilder()
        crew_manager = CrewManager()
        
        # Create the task
        task_response = await create_task(
            task_request=task_request,
            task_analyzer=task_analyzer,
            agent_builder=agent_builder,
            crew_manager=crew_manager
        )
        
        return RedirectResponse(
            url=f"/results?task_id={task_response.task_id}",
            status_code=303,
        )
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        return RedirectResponse(
            url=f"/error?message={str(e)}",
            status_code=303,
        )

@app.post("/api/tasks", response_model=TaskResponse)
async def create_task(
    task_request: TaskRequest,
    task_analyzer: TaskAnalyzer = Depends(get_task_analyzer),
    agent_builder: AgentBuilder = Depends(get_agent_builder),
    crew_manager: CrewManager = Depends(get_crew_manager),
):
    """Create a new task."""
    # Use provided values or defaults from config
    model_id = task_request.model_id or config.default_model_id
    ollama_model_id = task_request.ollama_model_id or config.default_ollama_model_id
    max_agents = task_request.max_agents or config.max_agents
    timeout_minutes = task_request.timeout_minutes or config.default_timeout_minutes
    
    # Generate a task ID
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    
    # Initialize task status
    active_tasks[task_id] = {
        "status": TaskStatus.PENDING,
        "prompt": task_request.prompt,
        "model_id": model_id,
        "ollama_model_id": ollama_model_id,
        "max_agents": max_agents,
        "timeout_minutes": timeout_minutes,
        "analysis": None,
        "agents": [],
        "result": None,
        "error": None,
        "progress": 0.0,
    }
    
    try:
        # Update status to analyzing
        active_tasks[task_id]["status"] = TaskStatus.ANALYZING
        
        # Analyze the task
        task_analysis = await task_analyzer.analyze(task_request.prompt)
        
        active_tasks[task_id]["analysis"] = task_analysis
        active_tasks[task_id]["progress"] = 0.3
        
        # Update status to building agents
        active_tasks[task_id]["status"] = TaskStatus.BUILDING
        
        # Set the ollama model to use for agents (this will update the config for this run)
        config.default_ollama_model_id = ollama_model_id
        
        # Build the agents
        agents = agent_builder.build_agents(task_analysis, max_agents)
        active_tasks[task_id]["agents"] = [agent.role for agent in agents]
        active_tasks[task_id]["progress"] = 0.6
        
        # Update status to running
        active_tasks[task_id]["status"] = TaskStatus.RUNNING
        
        # Create and start the crew
        crew = crew_manager.create_crew(agents, task_analysis, save_agents=True)
        
        # Start the task in a background thread
        crew_manager.start_task(task_id, crew, timeout_minutes)
        
        return TaskResponse(
            task_id=task_id,
            agents_created=len(agents),
            status=TaskStatus.RUNNING,
            estimated_completion_time=task_analysis.estimated_time,
        )
        
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        active_tasks[task_id]["status"] = TaskStatus.FAILED
        active_tasks[task_id]["error"] = str(e)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error creating task: {str(e)}",
        )

@app.get("/api/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get the status of a task."""
    if task_id not in active_tasks:
        raise HTTPException(
            status_code=404,
            detail="Task not found",
        )
    
    task_info = active_tasks[task_id]
    
    # Get elapsed time
    crew_manager = CrewManager()
    elapsed_time = crew_manager.get_elapsed_time(task_id) if task_id in crew_manager.tasks else "0 seconds"
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task_info["status"],
        elapsed_time=elapsed_time,
        progress=task_info["progress"],
        result=task_info["result"],
        error=task_info["error"],
    )

# Add this to your app/main.py file
@app.get("/agents", response_class=HTMLResponse)
async def agents_page(request: Request):
    """Render the agents page."""
    # Get the agent repository
    agent_repo = get_agent_repository()
    
    # Get all saved crews
    crews = agent_repo.list_crews()
    
    # Get agent collections (group by task_id)
    collections = []
    default_agents = agent_repo.list_agents(collection="default")
    
    # Add default collection if it has agents
    if default_agents:
        collections.append({
            "name": "default",
            "agent_count": len(default_agents),
            "agent_types": list(set(agent["role"] for agent in default_agents)),
            "created_at": default_agents[0].get("created_at", "") if default_agents else ""
        })
    
    # Get non-default collections
    collection_dirs = [d for d in os.listdir(agent_repo.agents_path) 
                      if os.path.isdir(os.path.join(agent_repo.agents_path, d)) and d != "default"]
    
    for collection_name in collection_dirs:
        collection_agents = agent_repo.list_agents(collection=collection_name)
        if collection_agents:
            collections.append({
                "name": collection_name,
                "agent_count": len(collection_agents),
                "agent_types": list(set(agent["role"] for agent in collection_agents)),
                "created_at": collection_agents[0].get("created_at", "") if collection_agents else ""
            })
    
    # Sort collections by name
    collections.sort(key=lambda x: x["name"])
    
    # Determine whether to show all agents
    show_all_agents = len(collections) <= 1
    
    # Get all agents if show_all_agents is True
    agents = []
    if show_all_agents:
        for collection_name in collection_dirs:
            collection_agents = agent_repo.list_agents(collection=collection_name)
            for agent in collection_agents:
                agent["collection"] = collection_name
                agents.append(agent)
        
        # Sort agents by role
        agents.sort(key=lambda x: x["role"])
    
    return templates.TemplateResponse(
        "saved_agents.html",
        {
            "request": request,
            "crews": crews,
            "collections": collections,
            "agents": agents,
            "show_all_agents": show_all_agents
        }
    )


@app.get("/customize_agents", response_class=HTMLResponse)
async def customize_agents_page(
    request: Request,
    task_id: Optional[str] = None,
    agent: Optional[str] = None
):
    """Render the customize agents page."""
    # Get the agent repository
    agent_repo = get_agent_repository()
    
    # Get all available agents
    all_agents = []
    selected_agents = []
    
    # Get all collections
    collection_dirs = [d for d in os.listdir(agent_repo.agents_path) 
                      if os.path.isdir(os.path.join(agent_repo.agents_path, d))]
    
    for collection in collection_dirs:
        agents = agent_repo.list_agents(collection=collection)
        for a in agents:
            a["collection"] = collection
            all_agents.append(a)
    
    # If task_id is provided, get the agents for that task
    if task_id and task_id in active_tasks and "agents" in active_tasks[task_id]:
        agent_roles = active_tasks[task_id]["agents"]
        for a in all_agents:
            if a["role"] in agent_roles:
                a["selected"] = True
                selected_agents.append(a)
    
    # If agent is provided, mark it as selected
    if agent:
        for a in all_agents:
            if a.get("file_path") == agent:
                a["selected"] = True
                if a not in selected_agents:
                    selected_agents.append(a)
    
    # Group agents by role type (can be a simplified approach for now)
    role_groups = {"All Agents": all_agents}
    
    return templates.TemplateResponse(
        "customize_agents.html",
        {
            "request": request,
            "all_agents": all_agents,
            "selected_agents": selected_agents,
            "role_groups": role_groups,
            "config": get_config()
        }
    )


# Update this in your app/main.py file
@app.get("/results", response_class=HTMLResponse)
async def results(request: Request, task_id: Optional[str] = None):
    """Render the results page."""
    if task_id is None:
        # If no task_id is provided, show the results list
        return await results_list(request)
    
    # Otherwise, show the specific task results
    if task_id not in active_tasks:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "error": "Task not found. It may have expired or been deleted.",
            },
        )
    
    task_info = active_tasks[task_id]
    
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "task_id": task_id,
            "task_info": task_info,
        },
    )


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="localhost", port=8000, reload=True)