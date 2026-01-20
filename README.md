# DeepAgent Orchestrator

A LangChain-based multi-agent orchestrator using DeepAgent as the core engine, supporting parallel execution, hot-reloadable sub-agents, CLI and REST API interfaces.

## Features

- **DeepAgent Core**: Built on LangChain's DeepAgent for battle-tested patterns
- **Parallel Execution**: Multiple sub-agents can work concurrently
- **Hot Reload**: Sub-agents defined in YAML reload automatically on changes
- **CLI Interface**: Click-based command-line tool
- **REST API**: FastAPI endpoints for external integrations
- **Planning Tools**: Built-in todo list management
- **File Operations**: Built-in filesystem tools
- **Human-in-the-Loop**: Approval workflows for sensitive operations
- **Memory Management**: Persistent memory across conversations

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd agent-orchestrator

# Install dependencies
pip install -e ".[dev]"

# Set environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

The project uses centralized configuration through `Settings` in `src/config.py`:

```bash
# .env file - environment variables
DEFAULT_MODEL=anthropic:claude-sonnet-4-20250514
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=http://localhost:11434/v1  # Optional: for local LLMs
TAVILY_API_KEY=your-tavily-key  # Optional: for web search
```

Or use `config/settings.yaml` for file-based configuration.

### Basic Usage

```bash
# Run a task through CLI
orchestrate run "Research AI agents and write a report" --agents-dir agents/

# Run with verbose logging
orchestrate run "Your task" --agents-dir agents/ -v

# Start the REST API
orchestrate api --port 8000
```

### Python API

```python
from src.orchestrator import DeepAgentOrchestrator, OrchestratorConfig
from src.config import load_settings, apply_env_overrides

# Method 1: Direct configuration
orchestrator = DeepAgentOrchestrator(
    model_name="anthropic:claude-sonnet-4-20250514",
    system_prompt="You are a helpful assistant.",
)

# Method 2: Using centralized Settings
settings = load_settings()
settings = apply_env_overrides(settings)
config = OrchestratorConfig.from_settings(settings)
orchestrator = DeepAgentOrchestrator(config=config)

result = asyncio.run(orchestrator.run("Research AI agents"))
print(result["messages"][-1].content)
```

## Documentation

See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for detailed documentation.

For agent developers, see [AGENTS.md](AGENTS.md) for coding conventions and guidelines.

## Project Structure

```
agent-orchestrator/
├── src/
│   ├── __init__.py
│   ├── orchestrator.py       # DeepAgent orchestrator wrapper
│   ├── agent_registry.py     # Sub-agent registry with hot-reload
│   ├── cli.py                # Click CLI interface
│   ├── api.py                # FastAPI REST endpoints
│   ├── config.py             # Centralized configuration
│   └── tools/
│       ├── __init__.py
│       ├── registry.py       # Custom tool registry
│       └── search.py         # Web search tools
├── agents/                   # Sub-agent YAML definitions
├── tests/
├── config/
│   └── settings.yaml         # Configuration file
├── pyproject.toml
├── README.md
└── AGENTS.md                 # Guidelines for agent developers
```

## License

MIT
