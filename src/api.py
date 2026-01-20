"""
REST API - FastAPI endpoints for orchestrator control.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from .agent_registry import SubAgentRegistry, SubAgentSpec
from .orchestrator import DeepAgentOrchestrator

logger = logging.getLogger(__name__)

orchestrator: DeepAgentOrchestrator | None = None
registry: SubAgentRegistry | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global orchestrator, registry

    logger.info("Starting DeepAgent Orchestrator API")

    registry = SubAgentRegistry()

    yield

    logger.info("Shutting down DeepAgent Orchestrator API")
    if registry and registry._watcher:
        registry.disable_hot_reload()


app = FastAPI(
    title="DeepAgent Orchestrator API",
    description="REST API for managing and running the DeepAgent orchestrator",
    version="0.1.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    """Request model for running a task."""

    task: str = Field(..., description="The task to execute", min_length=1)
    model: str = Field(
        default="anthropic:claude-sonnet-4-20250514", description="LLM model"
    )
    system_prompt: str | None = Field(
        default=None, description="Custom system prompt"
    )
    subagents: list[dict[str, Any]] = Field(
        default_factory=list, description="Sub-agents"
    )
    thread_id: str | None = Field(default=None, description="Thread ID for state")
    stream: bool = Field(default=True, description="Enable streaming")
    interrupt_on: dict | None = Field(default=None, description="HITL configuration")

    @validator("subagents", each_item=True)
    def validate_subagent(cls, v):
        """Validate subagent has required fields."""
        if not isinstance(v, dict):
            raise ValueError(f"Subagent must be a dictionary, got {type(v)}")
        if "name" not in v:
            raise ValueError("Subagent must have a 'name' field")
        if "system_prompt" not in v:
            raise ValueError("Subagent must have a 'system_prompt' field")
        return v


class SubAgentCreateRequest(BaseModel):
    """Request model for creating a sub-agent."""

    name: str = Field(..., description="Agent name", min_length=1)
    description: str = Field(..., description="Agent description", min_length=1)
    system_prompt: str = Field(..., description="System prompt")
    tools: list[str] = Field(default_factory=list, description="Tool names")
    model: str | None = Field(default=None, description="Model override")


class SubAgentResponse(BaseModel):
    """Response model for sub-agent."""

    name: str
    description: str
    system_prompt: str
    tools: list[str]
    model: str | None


class RunResponse(BaseModel):
    """Response model for task execution."""

    thread_id: str
    status: str
    stream_url: str | None = None
    result: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str
    model: str | None = None
    subagent_count: int = 0


@app.get("/", summary="API Info")
async def root():
    """Get API information."""
    return {
        "name": "DeepAgent Orchestrator API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check():
    """Check API health status."""
    global orchestrator

    model = None
    subagent_count = 0

    if orchestrator:
        model = orchestrator.config.model_name
        subagent_count = len(orchestrator.list_subagents())

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model=model,
        subagent_count=subagent_count,
    )


@app.post("/run", response_model=RunResponse, summary="Run Task")
async def run_task(request: RunRequest):
    """
    Run a task through the orchestrator.

    This endpoint initiates task execution and returns immediately with a thread ID.
    If streaming is enabled, connect to the WebSocket endpoint to receive output.
    """
    global orchestrator

    thread_id = request.thread_id or str(uuid.uuid4())

    try:
        orchestrator = DeepAgentOrchestrator(
            model_name=request.model,
            system_prompt=request.system_prompt,
            subagents=request.subagents,
            interrupt_on=request.interrupt_on,
        )

        if request.stream:
            return RunResponse(
                thread_id=thread_id,
                status="started",
                stream_url=f"/ws/run/{thread_id}",
            )
        else:
            result = await orchestrator.run(request.task, thread_id)
            output = result.get("messages", [{}])[-1].get("content", "")
            return RunResponse(
                thread_id=thread_id,
                status="completed",
                result=output,
            )

    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        return RunResponse(
            thread_id=thread_id,
            status="failed",
            error=str(e),
        )


@app.websocket("/ws/run/{thread_id}")
async def websocket_run(websocket: WebSocket, thread_id: str):
    """WebSocket endpoint for streaming execution."""
    global orchestrator

    await websocket.accept()
    logger.info(f"WebSocket connected for thread: {thread_id}")

    try:
        async for chunk in orchestrator.stream("", thread_id):
            await websocket.send_json(chunk)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()


@app.get("/subagents", response_model=list[SubAgentResponse], summary="List Sub-agents")
async def list_subagents():
    """List all registered sub-agents."""
    if orchestrator:
        return orchestrator.list_subagents()
    return []


@app.post("/subagents", response_model=dict, summary="Create Sub-agent")
async def create_subagent(request: SubAgentCreateRequest):
    """Register a new sub-agent."""
    global orchestrator

    try:
        spec = SubAgentSpec(
            name=request.name,
            description=request.description,
            system_prompt=request.system_prompt,
            tools=request.tools,
            model=request.model,
        )

        if orchestrator:
            orchestrator.add_subagent(
                name=spec.name,
                description=spec.description,
                system_prompt=spec.system_prompt,
                tools=spec.tools,
                model=spec.model,
            )

        if registry:
            registry.register(spec)

        return {
            "status": "created",
            "name": spec.name,
            "message": f"Sub-agent '{spec.name}' created successfully",
        }

    except Exception as e:
        logger.error(f"Failed to create sub-agent: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/subagents/{name}", response_model=dict, summary="Delete Sub-agent")
async def delete_subagent(name: str):
    """Delete a sub-agent by name."""
    global orchestrator

    if orchestrator:
        removed = orchestrator.remove_subagent(name)
        if removed:
            return {
                "status": "deleted",
                "name": name,
                "message": f"Sub-agent '{name}' deleted successfully",
            }
        raise HTTPException(status_code=404, detail=f"Sub-agent '{name}' not found")

    raise HTTPException(status_code=404, detail="No orchestrator available")


@app.get("/subagents/{name}", response_model=SubAgentResponse, summary="Get Sub-agent")
async def get_subagent(name: str):
    """Get a sub-agent by name."""
    if orchestrator:
        spec = orchestrator.list_subagents()
        for s in spec:
            if s.get("name") == name:
                return s
        raise HTTPException(status_code=404, detail=f"Sub-agent '{name}' not found")
    raise HTTPException(status_code=404, detail="No orchestrator available")


@app.get("/state/{thread_id}", response_model=dict, summary="Get Thread State")
async def get_thread_state(thread_id: str):
    """Get the state of a thread."""
    if orchestrator:
        state = orchestrator.get_state(thread_id)
        if state:
            return state
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found")
    raise HTTPException(status_code=404, detail="No orchestrator available")


@app.delete("/state/{thread_id}", response_model=dict, summary="Clear Thread State")
async def clear_thread_state(thread_id: str):
    """Clear the state of a thread."""
    if orchestrator:
        orchestrator.clear_state(thread_id)
        return {
            "status": "cleared",
            "thread_id": thread_id,
            "message": f"Thread '{thread_id}' state cleared",
        }
    raise HTTPException(status_code=404, detail="No orchestrator available")


@app.post("/config/hitl", response_model=dict, summary="Configure HITL")
async def configure_hitl(
    tool_name: str, allowed_decisions: Annotated[list[str], Query(...)] = None
):
    """Configure human-in-the-loop for a tool."""
    global orchestrator

    if allowed_decisions is None:
        allowed_decisions = ["approve", "edit", "reject"]

    if orchestrator:
        orchestrator.configure_hitl(tool_name, allowed_decisions)
        return {
            "status": "configured",
            "tool": tool_name,
            "allowed_decisions": allowed_decisions,
        }
    raise HTTPException(status_code=404, detail="No orchestrator available")


@app.get("/models", response_model=list[dict], summary="List Available Models")
async def list_models():
    """List available models."""
    return [
        {"id": "anthropic:claude-sonnet-4-20250514", "provider": "anthropic"},
        {"id": "anthropic:claude-3-5-sonnet-latest", "provider": "anthropic"},
        {"id": "anthropic:claude-3-haiku-20250514", "provider": "anthropic"},
        {"id": "openai:gpt-4o", "provider": "openai"},
        {"id": "openai:gpt-4o-mini", "provider": "openai"},
        {"id": "openai:o1", "provider": "openai"},
        {"id": "openai:o1-mini", "provider": "openai"},
        {"id": "ollama:llama3.2", "provider": "ollama"},
        {"id": "ollama:llama3.1", "provider": "ollama"},
        {"id": "ollama:mistral", "provider": "ollama"},
    ]


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return {"error": str(exc)}
