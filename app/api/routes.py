# app/api/routes.py
import os
from fastapi import APIRouter, Request, Depends, HTTPException, Form, Query
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from typing import List, Dict, Any, Optional
from app.models import (
    TaskRequest, TaskResponse, TaskStatusResponse, 
    AgentSpec, WorkflowType, TaskAnalysis
)
from app.config import get_config
from app.core.llm_provider import LLMProvider
from app.core.task_analyzer import TaskAnalyzer
from app.core.agent_builder import AgentBuilder
from app.core.crew_manager import CrewManager
from app.core.agent_repository import AgentRepository

router = APIRouter()

# Dependencies
def get_agent_repository():
    """Get the agent repository."""
    return AgentRepository()

# Routes for the web interface

@router.get("/agents")
async def list_agents(request: Request, agent_repo: AgentRepository = Depends(get_agent_repository)):
    """
    Render the saved agents page.
    
    Lists all saved agents and crews for reuse.
    """
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
        for collection_name in [d for d in os.listdir(agent_repo.agents_path) 
                           if os.path.isdir(os.path.join(agent_repo.agents_path, d))]:
            collection_agents = agent_repo.list_agents(collection=collection_name)
            for agent in collection_agents:
                agent["collection"] = collection_name
                agents.append(agent)
        
        # Sort agents by role
        agents.sort(key=lambda x: x["role"])
    
    return {
        "request": request,
        "crews": crews,
        "collections": collections,
        "agents": agents,
        "show_all_agents": show_all_agents
    }

@router.get("/collection")
async def view_collection(
    request: Request, 
    name: str = Query(..., description="Collection name"),
    agent_repo: AgentRepository = Depends(get_agent_repository)
):
    """
    Render the agent collection page.
    
    Shows all agents in a specific collection.
    """
    # Get agents in the collection
    agents = agent_repo.list_agents(collection=name)
    
    # Add collection to each agent
    for agent in agents:
        agent["collection"] = name
    
    return {
        "request": request,
        "collection_name": name,
        "agents": agents
    }

@router.get("/results")
async def list_results(request: Request, agent_repo: AgentRepository = Depends(get_agent_repository)):
    """
    Render the results page.
    
    Lists all saved task results.
    """
    # Get all saved results
    results = agent_repo.list_results()
    
    return {
        "request": request,
        "results": results
    }

@router.get("/view_result")
async def view_result(
    request: Request, 
    file: str = Query(..., description="Result file path"),
    agent_repo: AgentRepository = Depends(get_agent_repository)
):
    """
    Render a specific result file.
    
    Displays the content of a markdown result file.
    """
    # Get the result content
    content = agent_repo.get_result_content(file)
    
    # Get the file info
    file_stats = os.stat(file)
    filename = os.path.basename(file)
    task_id = filename.split('_')[0] if '_' in filename else "unknown"
    
    return {
        "request": request,
        "content": content,
        "filename": filename,
        "task_id": task_id,
        "size": file_stats.st_size,
        "created_at": os.path.getctime(file),
        "file_path": file
    }

@router.get("/download")
async def download_file(
    file: str = Query(..., description="File path to download"),
    agent_repo: AgentRepository = Depends(get_agent_repository)
):
    """
    Download a file.
    
    Returns the file as an attachment for downloading.
    """
    if not os.path.exists(file):
        raise HTTPException(status_code=404, detail="File not found")
    
    filename = os.path.basename(file)
    return FileResponse(
        path=file,
        filename=filename,
        media_type="application/octet-stream"
    )

@router.get("/reuse_crew")
async def reuse_crew(
    request: Request,
    crew_id: Optional[str] = None,
    task_id: Optional[str] = None,
    agent_repo: AgentRepository = Depends(get_agent_repository)
):
    """
    Render the reuse crew page.
    
    Allows reusing a saved crew for a new task.
    """
    # Get the crew
    crew = None
    
    if crew_id:
        # Get crew by file path
        crew_path = os.path.join(agent_repo.crews_path, crew_id)
        crew = agent_repo.get_crew(crew_path)
    elif task_id:
        # Find crew associated with task
        crews = agent_repo.list_crews()
        for c in crews:
            if task_id in c.get("name", ""):
                crew = c
                break
    
    if not crew:
        return RedirectResponse(url="/agents", status_code=303)
    
    # Get the agents for this crew
    agent_data = []
    for role in crew.get("agent_roles", []):
        # Find agent in all collections
        collection_dirs = [d for d in os.listdir(agent_repo.agents_path) 
                          if os.path.isdir(os.path.join(agent_repo.agents_path, d))]
        
        for collection in collection_dirs:
            agents = agent_repo.list_agents(collection=collection)
            for agent in agents:
                if agent.get("role") == role:
                    agent["collection"] = collection
                    agent_data.append(agent)
                    break
        
    return {
        "request": request,
        "crew": crew,
        "agents": agent_data,
        "config": get_config()
    }

@router.post("/reuse_crew")
async def submit_reuse_crew(
    request: Request,
    prompt: str = Form(...),
    ollama_model: str = Form(...),
    workflow_type: str = Form(...),
    agent_files: List[str] = Form(...),
    timeout_minutes: int = Form(10),
    agent_repo: AgentRepository = Depends(get_agent_repository)
):
    """
    Submit a new task using a saved crew.
    
    Creates a new task with the specified agents and crew configuration.
    """
    # Get the agents
    agents = []
    for file_path in agent_files:
        agent_data = agent_repo.get_agent(file_path)
        if agent_data:
            agents.append(agent_data)
    
    if not agents:
        raise HTTPException(status_code=400, detail="No valid agents selected")
    
    # Create Agent specifications
    agent_specs = []
    for agent in agents:
        agent_spec = AgentSpec(
            role=agent["role"],
            goal=agent["goal"],
            backstory=agent["backstory"],
            tools=agent.get("tools", []),
            allow_delegation=agent.get("allow_delegation", True)
        )
        agent_specs.append(agent_spec)
    
    # Create a minimal task analysis
    import uuid
    task_id = f"reuse_task_{uuid.uuid4().hex[:8]}"
    
    task_analysis = TaskAnalysis(
        task_id=task_id,
        complexity=5,
        estimated_time="10-20 minutes",
        required_agents=agent_specs,
        workflow_type=WorkflowType(workflow_type),
        dependencies={}  # Empty dependencies, will use sequential execution
    )
    
    # Create a task request
    task_request = TaskRequest(
        prompt=prompt,
        ollama_model_id=ollama_model,
        max_agents=len(agents),
        timeout_minutes=timeout_minutes
    )
    
    # Create the task (similar to main.py create_task)
    from app.main import create_task, active_tasks
    
    # Initialize active_tasks
    active_tasks[task_id] = {
        "status": "pending",
        "prompt": prompt,
        "ollama_model_id": ollama_model,
        "max_agents": len(agents),
        "timeout_minutes": timeout_minutes,
        "analysis": task_analysis,
        "agents": [agent["role"] for agent in agents],
        "result": None,
        "error": None,
        "progress": 0.3,  # Start at 30% since we're skipping analysis
    }
    
    # Build agents and run the crew
    try:
        # Update status to building agents
        active_tasks[task_id]["status"] = "building_agents"
        
        # Build the agents
        agent_builder = AgentBuilder()
        built_agents = agent_builder.build_agents(task_analysis, len(agents))
        active_tasks[task_id]["progress"] = 0.6
        
        # Update status to running
        active_tasks[task_id]["status"] = "running"
        
        # Create and start the crew
        crew_manager = CrewManager()
        crew = crew_manager.create_crew(built_agents, task_analysis)
        
        # Start the task in a background thread
        crew_manager.start_task(task_id, crew, timeout_minutes)
        
        # Return a redirect response to the results page
        return RedirectResponse(url=f"/results?task_id={task_id}", status_code=303)
        
    except Exception as e:
        # Handle any errors
        active_tasks[task_id]["status"] = "failed"
        active_tasks[task_id]["error"] = str(e)
        return RedirectResponse(url=f"/error?message={str(e)}", status_code=303)

@router.get("/agent_details")
async def agent_details(
    request: Request,
    file: str = Query(..., description="Agent file path"),
    agent_repo: AgentRepository = Depends(get_agent_repository)
):
    """
    Render the agent details page.
    
    Shows detailed information about a specific agent.
    """
    # Get the agent
    agent = agent_repo.get_agent(file)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return {
        "request": request,
        "agent": agent
    }

@router.post("/reuse_agent")
async def reuse_agent(
    request: Request,
    file: str = Query(..., description="Agent file path"),
    agent_repo: AgentRepository = Depends(get_agent_repository)
):
    """
    Reuse a specific agent.
    
    Initiates the process of creating a custom crew with this agent.
    """
    # Get the agent
    agent = agent_repo.get_agent(file)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Redirect to customize agents page with this agent pre-selected
    return RedirectResponse(url=f"/customize_agents?agent={file}", status_code=303)

@router.get("/customize_agents")
async def customize_agents(
    request: Request,
    task_id: Optional[str] = None,
    agent: Optional[str] = None,
    agent_repo: AgentRepository = Depends(get_agent_repository)
):
    """
    Render the customize agents page.
    
    Allows selecting and customizing agents for a new crew.
    """
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
    if task_id:
        from app.main import active_tasks
        if task_id in active_tasks and "agents" in active_tasks[task_id]:
            agent_roles = active_tasks[task_id]["agents"]
            for a in all_agents:
                if a["role"] in agent_roles:
                    a["selected"] = True
                    selected_agents.append(a)
    
    # If agent is provided, mark it as selected
    if agent:
        for a in all_agents:
            if a["file_path"] == agent:
                a["selected"] = True
                if a not in selected_agents:
                    selected_agents.append(a)
    
    # Group agents by role type
    role_groups = {}
    for a in all_agents:
        role = a["role"].lower()
        
        # Try to categorize by common patterns
        if any(keyword in role for keyword in ["search", "retriev", "find"]):
            category = "Information Retrieval"
        elif any(keyword in role for keyword in ["analy", "process", "assess"]):
            category = "Analysis & Processing"
        elif any(keyword in role for keyword in ["writ", "generat", "creat", "content"]):
            category = "Content Generation"
        elif any(keyword in role for keyword in ["format", "edit", "refin"]):
            category = "Formatting & Editing"
        else:
            category = "Other Agents"
        
        if category not in role_groups:
            role_groups[category] = []
        
        role_groups[category].append(a)
    
    return {
        "request": request,
        "all_agents": all_agents,
        "selected_agents": selected_agents,
        "role_groups": role_groups,
        "config": get_config()
    }

# API endpoints for programmatic access

@router.get("/api/agents", response_model=List[Dict[str, Any]])
async def api_list_agents(
    collection: Optional[str] = None,
    agent_repo: AgentRepository = Depends(get_agent_repository)
):
    """
    List available agents.
    
    Args:
        collection: Optional collection name to filter by
        
    Returns:
        List of agent data
    """
    if collection:
        return agent_repo.list_agents(collection=collection)
    
    # Get all agents from all collections
    all_agents = []
    collection_dirs = [d for d in os.listdir(agent_repo.agents_path) 
                      if os.path.isdir(os.path.join(agent_repo.agents_path, d))]
    
    for coll in collection_dirs:
        agents = agent_repo.list_agents(collection=coll)
        for agent in agents:
            agent["collection"] = coll
            all_agents.append(agent)
    
    return all_agents

@router.get("/api/crews", response_model=List[Dict[str, Any]])
async def api_list_crews(
    agent_repo: AgentRepository = Depends(get_agent_repository)
):
    """
    List available crews.
    
    Returns:
        List of crew data
    """
    return agent_repo.list_crews()

@router.get("/api/results", response_model=List[Dict[str, Any]])
async def api_list_results(
    agent_repo: AgentRepository = Depends(get_agent_repository)
):
    """
    List available results.
    
    Returns:
        List of result data
    """
    return agent_repo.list_results()

@router.get("/api/result_content")
async def api_result_content(
    file_path: str,
    agent_repo: AgentRepository = Depends(get_agent_repository)
):
    """
    Get the content of a result file.
    
    Args:
        file_path: Path to the result file
        
    Returns:
        Content of the result file
    """
    content = agent_repo.get_result_content(file_path)
    return {"content": content}