# DeepAgent Orchestrator - Implementation Guide

## Project Overview

A LangChain-based multi-agent orchestrator using DeepAgent as the core engine, supporting parallel execution, hot-reloadable sub-agents, CLI and REST API interfaces.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Core Modules](#core-modules)
3. [Configuration](#configuration)
4. [Sub-agent Definitions](#sub-agent-definitions)
5. [API & CLI Usage](#api--cli-usage)
6. [Execution Modes](#execution-modes)
7. [Dependencies](#dependencies)

---

## Project Structure

```
agent-orchestrator/
├── src/
│   ├── __init__.py
│   ├── orchestrator.py       # DeepAgent orchestrator wrapper
│   ├── agent_registry.py     # Sub-agent YAML registry with hot-reload
│   ├── cli.py                # Click CLI interface
│   ├── api.py                # FastAPI REST endpoints
│   └── tools/
│       ├── __init__.py
│       ├── registry.py       # Custom tool registry
│       ├── search.py         # Web search tools
│       └── custom_tools.py   # User-defined tools
├── agents/                   # Sub-agent YAML definitions
│   ├── research_agent.yaml
│   ├── coding_agent.yaml
│   ├── planning_agent.yaml
│   └── critic_agent.yaml
├── config/
│   └── settings.yaml
├── tests/
├── pyproject.toml
└── README.md
```

---

## Core Modules

### 1. `src/orchestrator.py`

```python
"""
DeepAgent Orchestrator - Core orchestrator using LangChain DeepAgent.

Features:
- Built-in planning (write_todos/read_todos)
- Built-in file operations (ls, read_file, write_file, edit_file, glob, grep)
- Built-in sub-agent delegation (task tool)
- Built-in human-in-the-loop approval (interrupt_on)
- Built-in context summarization
"""

from deepagents import create_deep_agent, CompiledSubAgent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore
from langchain.chat_models import init_chat_model
from typing import Optional, AsyncGenerator
import uuid

class DeepAgentOrchestrator:
    """
    Multi-agent orchestrator using LangChain DeepAgent.
    """
    
    def __init__(
        self,
        model_name: str = "anthropic:claude-sonnet-4-20250514",
        system_prompt: Optional[str] = None,
        backend: Optional[CompositeBackend] = None,
        subagents: list = None,
        interrupt_on: Optional[dict] = None,
        custom_tools: list = None,
        middleware: list = None,
    ):
        self.model_name = model_name
        self.model = init_chat_model(model_name)
        self.system_prompt = system_prompt
        self.backend = backend or self._default_backend()
        self.subagents = subagents or []
        self.interrupt_on = interrupt_on
        self.custom_tools = custom_tools or []
        self.custom_middleware = middleware or []
        self.agent = self._create_agent()
    
    def _default_backend(self) -> CompositeBackend:
        return CompositeBackend(
            default=StateBackend(),
            routes={"/memories/": StoreBackend(store=InMemoryStore())},
        )
    
    def _create_agent(self):
        return create_deep_agent(
            model=self.model,
            system_prompt=self.system_prompt,
            tools=self.custom_tools,
            subagents=self.subagents,
            backend=self.backend,
            interrupt_on=self.interrupt_on,
            middleware=self.custom_middleware,
        )
    
    async def run(self, task: str, thread_id: str = None) -> dict:
        """Execute a task through the orchestrator."""
        inputs = {"messages": [{"role": "user", "content": task}]}
        config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
        return await self.agent.ainvoke(inputs, config=config)
    
    async def stream(self, task: str, thread_id: str = None) -> AsyncGenerator[dict, None]:
        """Stream execution for real-time output."""
        inputs = {"messages": [{"role": "user", "content": task}]}
        config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
        async for chunk in self.agent.astream(inputs, config=config):
            yield chunk
    
    def add_subagent(self, name: str, description: str, system_prompt: str, 
                     tools: list = None, model: str = None):
        """Add a new sub-agent dynamically."""
        self.subagents.append({
            "name": name,
            "description": description,
            "system_prompt": system_prompt,
            "tools": tools or [],
            "model": model,
        })
```

### 2. `src/agent_registry.py`

```python
"""
Sub-agent Registry - Manage sub-agents with hot-reload capability.

Features:
- Load YAML definitions from directory
- Hot-reload on file changes
- Thread-safe registry access
"""

import yaml
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Optional
import threading
import logging

logger = logging.getLogger(__name__)

class SubAgentFileHandler(FileSystemEventHandler):
    """Handle file system events for agent YAML files."""
    
    def __init__(self, registry: 'SubAgentRegistry'):
        self.registry = registry
        self.debounce_timer = None
    
    def on_modified(self, event):
        if event.src_path.endswith(('.yaml', '.yml')):
            self._debounced_reload(event.src_path)
    
    def on_created(self, event):
        if event.src_path.endswith(('.yaml', '.yml')):
            self._debounced_reload(event.src_path)
    
    def _debounced_reload(self, path: str):
        if self.debounce_timer:
            self.debounce_timer.cancel()
        self.debounce_timer = threading.Timer(0.5, self._do_reload, [path])
        self.debounce_timer.start()
    
    def _do_reload(self, path: str):
        try:
            self.registry.load_file(path)
            logger.info(f"Hot-reloaded sub-agent from {path}")
        except Exception as e:
            logger.error(f"Failed to reload {path}: {e}")


class SubAgentRegistry:
    """Registry for managing sub-agents."""
    
    def __init__(self, orchestrator: 'DeepAgentOrchestrator'):
        self.orchestrator = orchestrator
        self._subagents: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._watcher: Optional[Observer] = None
    
    def load_directory(self, path: str):
        """Load all YAML files from directory."""
        for yaml_file in Path(path).glob("*.yaml"):
            self.load_file(yaml_file)
    
    def load_file(self, path: str):
        """Load a single YAML file."""
        with open(path) as f:
            spec = yaml.safe_load(f)
        name = spec["name"]
        with self._lock:
            self._subagents[name] = spec
        self.orchestrator.add_subagent(**spec)
        logger.info(f"Loaded sub-agent: {name}")
    
    def save_spec(self, spec: dict, path: str):
        """Save sub-agent spec to YAML file."""
        with open(path, "w") as f:
            yaml.dump(spec, f)
    
    def list_subagents(self) -> list[dict]:
        """List all registered sub-agents."""
        with self._lock:
            return list(self._subagents.values())
    
    def enable_hot_reload(self, paths: list[str]):
        """Enable file watching for hot-reload."""
        self._watcher = Observer()
        handler = SubAgentFileHandler(self)
        for path in paths:
            self._watcher.schedule(handler, path, recursive=True)
        self._watcher.start()
    
    def disable_hot_reload(self):
        """Disable file watching."""
        if self._watcher:
            self._watcher.stop()
            self._watcher.join()
```

### 3. `src/cli.py`

```python
"""
CLI Interface - Click-based command-line interface.
"""

import click
import asyncio
import yaml
from pathlib import Path
from .orchestrator import DeepAgentOrchestrator
from .agent_registry import SubAgentRegistry

@click.group()
def cli():
    """DeepAgent Orchestrator CLI"""
    pass

@cli.command()
@click.argument("task", nargs=-1, type=str)
@click.option("--model", "-m", default="anthropic:claude-sonnet-4-20250514")
@click.option("--agents-dir", "-d", help="Directory containing agent YAML files")
@click.option("--system-prompt", "-s", help="Custom system prompt")
@click.option("--stream/--no-stream", default=True)
@click.option("--thread-id", "-t")
def run(task, model, agents_dir, system_prompt, stream, thread_id):
    """Run a task through the DeepAgent orchestrator."""
    task_str = " ".join(task)
    
    subagents = []
    if agents_dir:
        agent_dir = Path(agents_dir)
        for yaml_file in agent_dir.glob("*.yaml"):
            with open(yaml_file) as f:
                subagents.append(yaml.safe_load(f))
    
    orchestrator = DeepAgentOrchestrator(
        model_name=model,
        system_prompt=system_prompt,
        subagents=subagents,
    )
    
    if stream:
        async def stream_output():
            async for chunk in orchestrator.stream(task_str, thread_id):
                if "messages" in chunk:
                    print(chunk["messages"][-1].content, end="", flush=True)
        asyncio.run(stream_output())
    else:
        result = asyncio.run(orchestrator.run(task_str, thread_id))
        print(result["messages"][-1].content)

@cli.command()
@click.argument("name")
@click.argument("description")
@click.argument("prompt")
@click.option("--output", "-o", help="Output YAML file path")
def create_agent(name, description, prompt, output):
    """Create a new sub-agent YAML definition."""
    spec = {
        "name": name,
        "description": description,
        "system_prompt": prompt,
        "tools": [],
    }
    if output:
        with open(output, "w") as f:
            yaml.dump(spec, f)
        click.echo(f"Saved to {output}")
    else:
        click.echo(yaml.dump(spec))

@cli.command()
@click.argument("agents_dir")
def list_agents(agents_dir):
    """List all sub-agents in a directory."""
    agent_dir = Path(agents_dir)
    for yaml_file in agent_dir.glob("*.yaml"):
        with open(yaml_file) as f:
            spec = yaml.safe_load(f)
        click.echo(f"  - {spec['name']}: {spec['description']}")

@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000)
def api(host, port):
    """Start the REST API server."""
    import uvicorn
    from .api import app
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    cli()
```

### 4. `src/api.py`

```python
"""
REST API - FastAPI endpoints for orchestrator control.
"""

from fastapi import FastAPI, WebSocket, HTTPException
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
import asyncio
import uuid

app = FastAPI(title="DeepAgent Orchestrator API")

orchestrator = None

class RunRequest(BaseModel):
    task: str
    model: str = "anthropic:claude-sonnet-4-20250514"
    system_prompt: Optional[str] = None
    subagents: list[dict] = []
    thread_id: Optional[str] = None
    stream: bool = True

class SubAgentCreateRequest(BaseModel):
    name: str
    description: str
    system_prompt: str
    tools: list[str] = []
    model: Optional[str] = None

@app.post("/run")
async def run_task(request: RunRequest):
    """Run a task through the orchestrator."""
    global orchestrator
    
    orchestrator = DeepAgentOrchestrator(
        model_name=request.model,
        system_prompt=request.system_prompt,
        subagents=request.subagents,
    )
    
    thread_id = request.thread_id or str(uuid.uuid4())
    
    if request.stream:
        return {"thread_id": thread_id, "stream_url": f"/ws/run/{thread_id}"}
    else:
        result = await orchestrator.run(request.task, thread_id)
        return {"thread_id": thread_id, "result": result["messages"][-1].content}

@app.websocket("/ws/run/{thread_id}")
async def websocket_run(websocket: WebSocket, thread_id: str):
    """WebSocket for streaming execution."""
    await websocket.accept()
    try:
        async for chunk in orchestrator.stream(request.task, thread_id):
            await websocket.send_json(chunk)
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()

@app.post("/subagents")
async def create_subagent(request: SubAgentCreateRequest):
    """Register a new sub-agent."""
    if orchestrator:
        orchestrator.add_subagent(**request.model_dump())
    return {"status": "created", "name": request.name}

@app.get("/subagents")
async def list_subagents():
    """List all registered sub-agents."""
    return {"subagents": orchestrator.subagents if orchestrator else []}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

---

## Configuration

### `config/settings.yaml`

```yaml
# DeepAgent Orchestrator Configuration

# LLM Model
model:
  provider: "anthropic"
  name: "claude-sonnet-4-20250514"
  base_url: "http://localhost:11434"
  api_key: "${ANTHROPIC_API_KEY}"
  temperature: 0.1
  max_tokens: 4096

# Backend Configuration
backend:
  type: "composite"
  config:
    root_dir: "/tmp/agent-workspace"
    store_namespace: "memories"

# Sub-agents
subagents:
  directory: "agents/"
  auto_load: true
  hot_reload: true

# Human-in-the-Loop
hitl:
  enabled: true
  tools:
    "run_command":
      allowed_decisions: ["approve", "edit", "reject"]
    "delete_file":
      allowed_decisions: ["approve", "reject"]

# API Server
api:
  host: "0.0.0.0"
  port: 8000
  reload: false
  cors_origins:
    - "*"

# Logging
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

---

## Sub-agent Definitions

### Example: `agents/research_agent.yaml`

```yaml
name: research_agent
description: "Deep research specialist for in-depth questions"
system_prompt: |
  You are an expert researcher. Your approach:
  1. Break down the research question into key topics
  2. Use web search to find diverse, authoritative sources
  3. Read and extract relevant information from sources
  4. Cross-reference findings across multiple sources
  5. Synthesize into a comprehensive, well-cited report
  
  Always cite your sources with links when possible.
tools:
  - web_search
  - tavily_search
  - read_file
model: "openai:gpt-4o"
```

### Example: `agents/coding_agent.yaml`

```yaml
name: coding_agent
description: "Code generation and review specialist"
system_prompt: |
  You are an expert programmer. Your approach:
  1. Understand the requirements thoroughly
  2. Plan the implementation steps using write_todos
  3. Write clean, well-documented code
  4. Test your implementation
  5. Review and refactor as needed
  
  Follow best practices: DRY, SOLID, clean code principles.
tools:
  - read_file
  - write_file
  - edit_file
  - glob
  - execute
```

---

## API & CLI Usage

### CLI Commands

```bash
# Run a task
orchestrate run "Research AI agents and write a report" --agents-dir agents/

# Create a new sub-agent
orchestrate create_agent research_agent "Research specialist" "You are an expert researcher..."

# Start API server
orchestrate api --port 8000

# List all agents
orchestrate list-agents agents/
```

### REST API Endpoints

```bash
# Run task
POST /run
{
  "task": "Research AI agents",
  "model": "anthropic:claude-sonnet-4-20250514",
  "subagents": [...],
  "stream": true
}

# List sub-agents
GET /subagents

# Create sub-agent
POST /subagents
{
  "name": "research_agent",
  "description": "Research specialist",
  "system_prompt": "You are an expert researcher..."
}

# Health check
GET /health
```

### WebSocket Streaming

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/run/thread-id');

// Receive chunks
ws.onmessage = (event) => {
  const chunk = JSON.parse(event.data);
  console.log(chunk);
};
```

---

## Execution Modes

### Parallel Sub-agent Execution

DeepAgent's `task` tool supports parallel execution:

```python
system_prompt = """
You can delegate to sub-agents using the task() tool.

For PARALLEL execution:
- task(name="research_agent", task="Research AI trends")
- task(name="coding_agent", task="Build a prototype")

Sub-agents execute in parallel and return results.
"""
```

### Sequential Execution with Planning

```python
system_prompt = """
Use write_todos() to plan your approach:
1. First, research the topic
2. Then, analyze findings
3. Finally, write the report

Use read_todos() to track progress.
"""
```

---

## Dependencies

### `pyproject.toml`

```toml
[project]
name = "agent-orchestrator"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "deepagents>=0.3.0",
    "langchain>=0.2.0",
    "langgraph>=0.2.0",
    "langchain-openai>=0.1.0",
    "langchain-anthropic>=0.1.0",
    "click>=8.0",
    "fastapi>=0.100",
    "uvicorn>=0.23",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "watchdog>=3.0",
    "httpx>=0.25",
    "python-dotenv>=1.0",
]

[project.scripts]
orchestrate = "src.cli:main"

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "black>=23.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
]
```

---

## Built-in DeepAgent Tools

| Tool | Description |
|------|-------------|
| `write_todos` | Create and manage task lists |
| `read_todos` | Read todo list state |
| `ls` | List directory contents |
| `read_file` | Read file with optional pagination |
| `write_file` | Create or overwrite files |
| `edit_file` | Perform string replacements |
| `glob` | Find files matching pattern |
| `grep` | Search text patterns in files |
| `execute` | Run shell commands (sandboxed) |
| `task` | Delegate to sub-agents |

---

## Built-in Middleware

| Middleware | Purpose |
|------------|---------|
| `TodoListMiddleware` | Task planning and tracking |
| `FilesystemMiddleware` | File operations and context offloading |
| `SubAgentMiddleware` | Sub-agent delegation |
| `SummarizationMiddleware` | Auto-summarize long contexts |
| `AnthropicPromptCachingMiddleware` | Prompt caching (Anthropic) |
| `PatchToolCallsMiddleware` | Fix dangling tool calls |
| `HumanInTheLoopMiddleware` | Human approval for sensitive tools |

---

## Quick Start

```bash
# Install dependencies
pip install deepagents click fastapi uvicorn pyyaml watchdog httpx

# Set API key
export ANTHROPIC_API_KEY="your-api-key"

# Run a task
python -m src.cli run "Research AI agents and write a report" --agents-dir agents/

# Start API
python -m src.cli api --port 8000
```
