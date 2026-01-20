# AGENTS.md

This file provides guidelines and instructions for agentic coding agents operating in this repository.

## Build, Lint, and Test Commands

### Installation
```bash
pip install -e ".[dev]"  # Install with dev dependencies
```

### Running the Project
```bash
orchestrate run "Your task here" --agents-dir agents/  # Run CLI task
orchestrate api --port 8000  # Start REST API server
```

### Linting and Type Checking
```bash
black src/ tests/  # Format code
ruff check src/ tests/  # Run linter
ruff check --fix src/ tests/  # Auto-fix linter issues
mypy src/ tests/  # Type checking
```

### Testing
```bash
pytest  # Run all tests
pytest tests/test_orchestrator.py  # Run tests in specific file
pytest tests/test_orchestrator.py::TestOrchestratorConfig::test_defaults  # Run single test
pytest -k "test_defaults"  # Run tests matching pattern
pytest --cov=src --cov-report=term-missing  # Run with coverage
```

## Code Style Guidelines

### Imports
- Group imports: stdlib, third-party, local
- Sort alphabetically within groups
- Use relative imports for internal modules (`from .orchestrator import...`)
- Use absolute imports for external packages

```python
# Good
import asyncio
import logging
from pathlib import Path

from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from .orchestrator import DeepAgentOrchestrator

# Bad
from .orchestrator import *
import sys, os
```

### Formatting
- 4 spaces for indentation
- Maximum line length: 100 characters
- Use double quotes for strings
- One blank line between top-level declarations
- No redundant parentheses in if statements

### Types
- Use type hints for all function signatures
- Use `Optional[X]` instead of `X | None`
- Use `List[X]`, `Dict[K, V]` from typing (not built-in generics)
- Use dataclasses for simple data structures
- Use Pydantic BaseModel for complex validation

```python
# Good
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class Config:
    model_name: str = "anthropic:claude-sonnet-4-20250514"
    subagents: List[Dict[str, Any]] = field(default_factory=list)

async def run(task: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
    pass
```

### Naming Conventions
- **Files**: snake_case (e.g., `agent_registry.py`)
- **Classes**: PascalCase (e.g., `DeepAgentOrchestrator`)
- **Functions/variables**: snake_case (e.g., `get_state`, `orchestrator_config`)
- **Constants**: SCREAMING_SNAKE_CASE (e.g., `DEFAULT_MODEL`)
- **Booleans**: Prefix with `is_`, `has_`, `should_` (e.g., `is_enabled`)
- **Private methods**: Leading underscore (e.g., `_create_agent`)

### Error Handling
- Use try/except with specific exception types
- Log errors with context using the logging module
- Re-raise exceptions after logging unless handling specifically
- Use custom exception classes for domain errors

```python
# Good
try:
    result = await self.agent.ainvoke(inputs, config=config)
    logger.info(f"Task completed for thread: {thread_id}")
    return result
except Exception as e:
    logger.error(f"Task execution failed: {e}")
    raise
```

### Async Patterns
- Use `async def` for I/O-bound operations
- Use `asyncio.run()` for entry points
- Prefer async generators for streaming
- Avoid blocking calls in async functions

### File Organization
```
src/
├── __init__.py          # Package root
├── orchestrator.py      # Core orchestrator logic
├── agent_registry.py    # Sub-agent management
├── cli.py               # Click CLI interface
├── api.py               # FastAPI endpoints
├── config.py            # Configuration loading
└── tools/
    ├── __init__.py
    ├── registry.py      # Tool registry
    └── custom_tools.py  # Custom tool implementations
```

### Documentation
- Module-level docstrings for all public modules
- Class docstrings describing attributes and usage
- Function docstrings with Args and Returns sections
- Use Google-style docstrings

```python
class DeepAgentOrchestrator:
    """
    Multi-agent orchestrator using LangChain DeepAgent.

    Attributes:
        config: OrchestratorConfig instance
        agent: Compiled DeepAgent graph
    """

    async def run(self, task: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a task through the orchestrator.

        Args:
            task: The task to execute
            thread_id: Optional thread ID for state persistence

        Returns:
            Dict containing the execution result with messages
        """
```

## Testing Guidelines
- Use pytest with `pytest-asyncio`
- Test files: `test_*.py` in `tests/` directory
- Test classes: `Test*` naming
- Test methods: `test_*` naming
- Use `pytest.mark.asyncio` for async tests
- Follow AAA pattern: Arrange, Act, Assert
- Mock external dependencies in unit tests

## Git Conventions
- Use conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`
- Write descriptive commit messages (50 chars max for title)
- Create feature branches from main: `feat/add-agent-yaml-support`
- Keep PRs small and focused on single concerns

## Security Practices
- Never commit secrets, keys, or credentials
- Validate and sanitize all user inputs
- Use parameterized queries for database operations
- Implement proper authentication and authorization checks
