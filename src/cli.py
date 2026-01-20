"""
CLI Interface - Click-based command-line interface.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .agent_registry import SubAgentRegistry, SubAgentSpec
from .orchestrator import DeepAgentOrchestrator

# Load .env file from current directory or parent directories
load_dotenv()

console = Console()
logger = logging.getLogger(__name__)


# Registry of tool functions
_TOOL_REGISTRY = {}


def _register_tool(name: str, func, description: str = ""):
    """Register a tool function."""
    from langchain.tools import tool as langchain_tool

    # Wrap the function with @tool decorator
    wrapped_func = langchain_tool(func)
    wrapped_func.name = name
    wrapped_func.description = (
        description or func.__doc__.strip() if func.__doc__ else f"Tool: {name}"
    )

    _TOOL_REGISTRY[name] = wrapped_func


def _get_tool(tool_name: str):
    """Get a tool by name."""
    return _TOOL_REGISTRY.get(tool_name)


def _convert_tools_to_langchain_list(tool_names: list):
    """Convert tool names to LangChain BaseTool objects."""
    result = []
    for name in tool_names:
        tool = _get_tool(name)
        if tool:
            result.append(tool)
        else:
            logger.warning(f"Tool '{name}' not found, skipping")
    return result


# Initialize tool registry with custom search tools
def _init_tool_registry():
    """Initialize the tool registry with available tools."""
    from .tools.analysis import get_analysis_tools
    from .tools.dependency_tools import get_dependency_tools
    from .tools.pattern_tools import get_pattern_tools
    from .tools.registry import get_default_tool_registry
    from .tools.search import get_search_tools

    registry = get_default_tool_registry()
    registry = get_search_tools(registry)

    # Register all analysis tools
    analysis_registry = get_analysis_tools()
    for tool_def in analysis_registry.list():
        _register_tool(
            name=tool_def.name,
            func=tool_def.func,
            description=tool_def.description,
        )

    # Register all pattern tools
    pattern_registry = get_pattern_tools()
    for tool_def in pattern_registry.list():
        _register_tool(
            name=tool_def.name,
            func=tool_def.func,
            description=tool_def.description,
        )

    # Register all dependency tools
    dependency_registry = get_dependency_tools()
    for tool_def in dependency_registry.list():
        _register_tool(
            name=tool_def.name,
            func=tool_def.func,
            description=tool_def.description,
        )

    logger.debug(f"Registered {_len_tool_registry()} tools")


def _len_tool_registry():
    """Get the number of registered tools."""
    return len(_TOOL_REGISTRY)


# Initialize on module load
_init_tool_registry()


@click.group()
def cli():
    """DeepAgent Orchestrator CLI"""
    pass


@cli.command()
@click.argument("task", nargs=-1, type=str)
@click.option(
    "--model",
    "-m",
    default=None,
    help="LLM model to use (reads DEFAULT_MODEL from .env if not specified)",
)
@click.option("--agents-dir", "-d", help="Directory containing agent YAML files")
@click.option("--system-prompt", "-s", help="Custom system prompt")
@click.option("--stream/--no-stream", default=True, help="Stream output")
@click.option("--thread-id", "-t", help="Thread ID for state persistence")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--openai-api-key", help="API key for OpenAI-compatible API")
@click.option("--openai-base-url", help="Base URL for OpenAI-compatible API")
def run(
    task,
    model,
    agents_dir,
    system_prompt,
    stream,
    thread_id,
    verbose,
    openai_api_key,
    openai_base_url,
):
    """Run a task through the DeepAgent orchestrator."""
    # Configure logging
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        logger.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        logger.setLevel(logging.INFO)

    if not task:
        click.echo("Error: No task provided", err=True)
        sys.exit(1)

    task_str = " ".join(task)
    click.echo(f"Task: {task_str}")
    click.echo(f"Model: {model}")

    subagents = []
    if agents_dir:
        agent_dir = Path(agents_dir)
        if not agent_dir.exists():
            click.echo(f"Error: Agent directory not found: {agents_dir}", err=True)
            sys.exit(1)

        for yaml_file in agent_dir.glob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    subagent_data = yaml.safe_load(f)

                    # Remove model field - use default model from orchestrator
                    if subagent_data and "model" in subagent_data:
                        click.echo(f"Note: Removing model from {yaml_file.name} (using default)")
                        del subagent_data["model"]
                    # Convert tools to LangChain BaseTool objects if present
                    # Note: deepagents provides built-in tools automatically (read_file, write_file, etc.)
                    # We only need to provide custom tools that are NOT built into deepagents
                    if subagent_data and "tools" in subagent_data:
                        tool_names = subagent_data["tools"]
                        # Get only custom tools (not built-in deepagents tools)
                        builtin_tools = {
                            "read_file",
                            "write_file",
                            "edit_file",
                            "ls",
                            "glob",
                            "grep",
                            "write_todos",
                            "read_todos",
                            "execute",
                            "task",
                        }
                        custom_tool_names = [t for t in tool_names if t not in builtin_tools]

                        if custom_tool_names:
                            tools = _convert_tools_to_langchain_list(custom_tool_names)
                            subagent_data["tools"] = tools
                            click.echo(
                                f"Loaded agent: {yaml_file.name} with custom tools: {custom_tool_names}"
                            )
                        else:
                            # Use only built-in tools - remove tools field
                            del subagent_data["tools"]
                            click.echo(f"Loaded agent: {yaml_file.name} (using built-in tools)")
                    else:
                        click.echo(f"Loaded agent: {yaml_file.name}")

                    subagents.append(subagent_data)
            except Exception as e:
                click.echo(f"Error loading {yaml_file.name}: {e}", err=True)

    try:
        orchestrator = DeepAgentOrchestrator(
            model_name=model,
            system_prompt=system_prompt,
            subagents=subagents,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            enable_memory=False,  # Disable memory to avoid store errors
        )
    except Exception as e:
        click.echo(f"Error initializing orchestrator: {e}", err=True)
        sys.exit(1)

    click.echo("\nExecuting...")

    if stream:
        import time

        # Animation frames for progress indicator
        animation_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠸"]
        last_update = [time.time()]
        activity_log = []

        def log_activity(message):
            """Log agent activity."""
            timestamp = time.strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}"
            activity_log.append(log_entry)
            if verbose:
                print(f"\n{log_entry}", flush=True)

        async def stream_output():
            output = []
            frame_idx = [0]

            try:
                print("\nExecuting... (press Ctrl+C to cancel)", flush=True)
                print("-" * 50, flush=True)

                async for chunk in orchestrator.stream(task_str, thread_id):
                    # Update progress indicator every second
                    current_time = time.time()
                    if current_time - last_update[0] >= 1.0:
                        frame = animation_frames[frame_idx[0] % len(animation_frames)]
                        print(f"\r{frame} Agent is working...", end="", flush=True)
                        frame_idx[0] += 1
                        last_update[0] = current_time

                    content = None

                    if "messages" in chunk:
                        content = chunk["messages"][-1].content
                    elif "model" in chunk:
                        model_chunk = chunk["model"]
                        if isinstance(model_chunk, dict) and "messages" in model_chunk:
                            content = model_chunk["messages"][-1].content
                        elif isinstance(model_chunk, str):
                            content = model_chunk
                    else:
                        for key, value in chunk.items():
                            if isinstance(value, str) and value.strip():
                                content = value
                                break
                            elif isinstance(value, dict) and "content" in value:
                                content = value["content"]
                                break

                    if content:
                        # Log tool usage or significant events
                        if "tool" in str(chunk).lower() or "search" in str(content).lower():
                            log_activity(f"Tool output received: {str(content)[:100]}...")
                        output.append(content)
                        print(content, end="", flush=True)

                # Clear progress indicator and show completion
                print(f"\r{' ' * 30}\r", flush=True)
                print("-" * 50, flush=True)
                print(f"✓ Execution completed ({len(output)} output chunks)", flush=True)

                if activity_log and verbose:
                    print("\nActivity log:")
                    for entry in activity_log[-10:]:  # Show last 10 entries
                        print(f"  {entry}")

            except asyncio.CancelledError:
                print("\n\n⚠ Execution cancelled by user")
                sys.exit(130)
            except Exception as e:
                print(f"\r{' ' * 30}\r", flush=True)
                click.echo(f"\nError during execution: {e}", err=True)
                if activity_log:
                    print("\nActivity log:")
                    for entry in activity_log:
                        print(f"  {entry}")
                sys.exit(1)
            return "".join(output)

        result = asyncio.run(stream_output())
        print()
    else:
        try:
            result = asyncio.run(orchestrator.run(task_str, thread_id))
            output = result["messages"][-1].content
            click.echo("\nResult:")
            click.echo(output)
        except Exception as e:
            click.echo(f"Error during execution: {e}", err=True)
            sys.exit(1)


@cli.command()
@click.argument("name")
@click.argument("description")
@click.argument("prompt")
@click.option("--output", "-o", help="Output YAML file path")
@click.option("--tools", "-T", multiple=True, help="Tool names for the agent")
@click.option("--model", "-m", help="Model for the agent")
def create_agent(name, description, prompt, output, tools, model):
    """Create a new sub-agent YAML definition."""
    spec = SubAgentSpec(
        name=name,
        description=description,
        system_prompt=prompt,
        tools=list(tools) if tools else [],
        model=model,
    )

    if output:
        registry = SubAgentRegistry()
        registry.save_spec(spec, output)
        click.echo(f"Saved to {output}")
    else:
        yaml_output = yaml.dump(spec.to_dict(), default_flow_style=False, sort_keys=False)
        click.echo(yaml_output)


@cli.command()
@click.argument("agents_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--format",
    "-f",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format",
)
def list_agents(agents_dir, format):
    """List all sub-agents in a directory."""
    agent_dir = Path(agents_dir)
    agents = []

    for yaml_file in agent_dir.glob("*.yaml"):
        try:
            with open(yaml_file) as f:
                spec = yaml.safe_load(f)
                agents.append(
                    {
                        "name": spec.get("name", yaml_file.stem),
                        "description": spec.get("description", ""),
                        "tools": len(spec.get("tools", [])),
                        "model": spec.get("model", "default"),
                    }
                )
        except Exception as e:
            click.echo(f"Error loading {yaml_file.name}: {e}", err=True)

    if format == "json":
        click.echo(click.style(json.dumps(agents, indent=2), fg="green"))
    elif format == "yaml":
        click.echo(click.style(yaml.dump(agents, default_flow_style=False), fg="green"))
    else:
        table = Table(title="Sub-Agents")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="magenta")
        table.add_column("Tools", style="yellow")
        table.add_column("Model", style="green")

        for agent in agents:
            table.add_row(
                agent["name"],
                (
                    agent["description"][:50] + "..."
                    if len(agent["description"]) > 50
                    else agent["description"]
                ),
                str(agent["tools"]),
                agent["model"],
            )

        console.print(table)


@cli.command()
@click.argument("name")
@click.argument("description")
@click.argument("prompt")
@click.option("--dir", "-d", default="agents", help="Directory to save the agent YAML")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode with prompts")
def add_agent(name, description, prompt, dir, interactive):
    """Add a new agent to the registry."""
    import logging

    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", default=8000, help="Port to listen on")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def api(host, port, reload):
    """Start the REST API server."""
    import uvicorn

    from .api import app

    click.echo(f"Starting API server on {host}:{port}")

    uvicorn.run(app, host=host, port=port, reload=reload, log_level="info")


@cli.command()
@click.argument("task", nargs=-1, type=str)
@click.option("--agents", "-a", multiple=True, help="Agent names to invoke in parallel")
@click.option(
    "--model",
    "-m",
    default="anthropic:claude-sonnet-4-20250514",
    help="LLM model to use",
)
def parallel(task, agents, model):
    """Run multiple agents in parallel on the same task."""
    if not task:
        click.echo("Error: No task provided", err=True)
        sys.exit(1)

    if not agents:
        click.echo("Error: No agents specified. Use --agents agent1,agent2,...", err=True)
        sys.exit(1)

    task_str = " ".join(task)

    click.echo(f"Task: {task_str}")
    click.echo(f"Agents: {', '.join(agents)}")

    subagents = []
    for agent_name in agents:
        subagents.append(
            {
                "name": agent_name,
                "description": f"Agent: {agent_name}",
                "system_prompt": f"You are {agent_name}. Execute the task.",
            }
        )

    DeepAgentOrchestrator(
        model_name=model,
        subagents=subagents,
    )

    click.echo("\nExecuting in parallel...")

    system_prompt = f"""
    You are a coordinator. Execute the following task by invoking these agents in parallel:
    {task_str}

    Available agents: {", ".join(agents)}

    Use the task() tool to invoke multiple agents simultaneously.
    """

    async def run_parallel():
        orchestrator2 = DeepAgentOrchestrator(
            model_name=model,
            system_prompt=system_prompt,
            subagents=subagents,
        )

        async for chunk in orchestrator2.stream(task_str):
            if "messages" in chunk:
                print(chunk["messages"][-1].content, end="", flush=True)

    asyncio.run(run_parallel())


@cli.command()
def version():
    """Show version information."""
    from . import __version__

    click.echo(f"DeepAgent Orchestrator v{__version__}")

    try:
        import deepagents

        version = getattr(deepagents, "__version__", "unknown")
        click.echo(f"DeepAgents v{version}")
    except ImportError:
        pass


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
