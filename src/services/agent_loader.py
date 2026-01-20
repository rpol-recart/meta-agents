"""
Agent loader for DeepAgent Orchestrator.

This module provides the AgentLoader class for loading and processing sub-agents
from YAML configuration files, eliminating duplication between CLI and API.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from ..agent_registry import SubAgentRegistry, SubAgentSpec

logger = logging.getLogger(__name__)


BUILTIN_TOOLS = {
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


class AgentLoader:
    """Loader for sub-agent configurations from YAML files."""

    def __init__(self, agents_dir: str | None = None):
        """
        Initialize the agent loader.

        Args:
            agents_dir: Optional directory path containing agent YAML files
        """
        self.agents_dir = Path(agents_dir) if agents_dir else None
        self._tool_registry: dict[str, Any] = {}
        self._registry: SubAgentRegistry | None = None

    def register_tool(self, name: str, func, description: str = ""):
        """
        Register a tool function.

        Args:
            name: Tool name
            func: Tool function
            description: Tool description
        """
        from langchain.tools import tool as langchain_tool

        wrapped_func = langchain_tool(func)
        wrapped_func.name = name
        wrapped_func.description = (
            description or func.__doc__.strip() if func.__doc__ else f"Tool: {name}"
        )

        self._tool_registry[name] = wrapped_func
        logger.debug(f"Registered tool: {name}")

    def load_from_directory(self, agents_dir: str) -> list[dict[str, Any]]:
        """
        Load all agent YAML files from a directory.

        Args:
            agents_dir: Directory path containing agent YAML files

        Returns:
            List of loaded sub-agent configurations
        """
        agent_dir = Path(agents_dir)

        if not agent_dir.exists():
            logger.warning(f"Agent directory not found: {agents_dir}")
            return []

        subagents = []

        for yaml_file in agent_dir.glob("*.yaml"):
            try:
                agent_config = self.load_from_file(yaml_file)
                if agent_config:
                    subagents.append(agent_config)
            except Exception as e:
                logger.error(f"Error loading {yaml_file.name}: {e}")

        logger.info(f"Loaded {len(subagents)} agents from {agents_dir}")
        return subagents

    def load_from_file(self, file_path: str) -> dict[str, Any] | None:
        """
        Load a single agent from a YAML file.

        Args:
            file_path: Path to the YAML file

        Returns:
            Agent configuration dict or None if invalid
        """
        yaml_file = Path(file_path)

        if not yaml_file.exists():
            logger.error(f"Agent file not found: {file_path}")
            return None

        try:
            with open(yaml_file) as f:
                subagent_data = yaml.safe_load(f)

            if not subagent_data:
                logger.warning(f"Empty agent file: {file_path}")
                return None

            processed_agent = self._process_agent_config(subagent_data, yaml_file.name)
            return processed_agent

        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading agent from {file_path}: {e}")
            return None

    def _process_agent_config(
        self, agent_data: dict[str, Any], source_name: str
    ) -> dict[str, Any]:
        """
        Process and validate an agent configuration.

        Args:
            agent_data: Raw agent configuration
            source_name: Source file/path for logging

        Returns:
            Processed agent configuration
        """
        processed = agent_data.copy()

        if "model" in processed:
            logger.debug(
                f"Removing model from {source_name} (using orchestrator default)"
            )
            del processed["model"]

        if "tools" in processed:
            processed["tools"] = self._process_tools(processed["tools"], source_name)

        logger.debug(f"Processed agent: {processed.get('name', source_name)}")
        return processed

    def _process_tools(self, tool_names: list[str], source_name: str) -> list[Any]:
        """
        Process tool names and convert to LangChain BaseTool objects.

        Args:
            tool_names: List of tool names
            source_name: Source file/path for logging

        Returns:
            List of LangChain BaseTool objects
        """
        custom_tool_names = [t for t in tool_names if t not in BUILTIN_TOOLS]

        if not custom_tool_names:
            logger.debug(f"Agent {source_name} uses only built-in tools")
            return []

        tools = []
        for tool_name in custom_tool_names:
            tool = self._get_tool(tool_name)
            if tool:
                tools.append(tool)
            else:
                logger.warning(f"Tool '{tool_name}' not found in registry, skipping")

        if tools:
            logger.debug(f"Agent {source_name} loaded with {len(tools)} custom tools")

        return tools

    def _get_tool(self, tool_name: str) -> Any | None:
        """
        Get a tool by name from the registry.

        Args:
            tool_name: Tool name

        Returns:
            LangChain BaseTool or None
        """
        return self._tool_registry.get(tool_name)

    def load_with_registry(
        self, agents_dir: str, registry: SubAgentRegistry | None = None
    ) -> list[dict[str, Any]]:
        """
        Load agents from directory and optionally register them.

        Args:
            agents_dir: Directory containing agent YAML files
            registry: Optional registry to register loaded agents

        Returns:
            List of processed sub-agent configurations
        """
        self._registry = registry

        agents = self.load_from_directory(agents_dir)

        if registry and self._registry:
            for agent in agents:
                self._registry.register(
                    SubAgentSpec(
                        name=agent.get("name", "unknown"),
                        description=agent.get("description", ""),
                        system_prompt=agent.get("system_prompt", ""),
                        tools=agent.get("tools", []),
                        model=agent.get("model"),
                    )
                )

        return agents

    def validate_agent(self, agent_data: dict[str, Any]) -> list[str]:
        """
        Validate an agent configuration.

        Args:
            agent_data: Agent configuration to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not isinstance(agent_data, dict):
            return ["Agent must be a dictionary"]

        if "name" not in agent_data:
            errors.append("Agent must have a 'name' field")

        if "system_prompt" not in agent_data:
            errors.append("Agent must have a 'system_prompt' field")

        if "name" in agent_data:
            name = agent_data["name"]
            if not isinstance(name, str) or not name.strip():
                errors.append("Agent name must be a non-empty string")
            elif len(name) > 100:
                errors.append("Agent name must be 100 characters or less")

        if "tools" in agent_data:
            if not isinstance(agent_data["tools"], list):
                errors.append("'tools' must be a list")

        return errors

    def get_registered_tools(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tool_registry.keys())

    def clear_registry(self):
        """Clear the tool registry."""
        self._tool_registry.clear()
        logger.debug("Cleared tool registry")


def create_agent_loader(agents_dir: str | None = None) -> AgentLoader:
    """
    Factory function to create an AgentLoader with default tools.

    Args:
        agents_dir: Optional directory containing agent YAML files

    Returns:
        Configured AgentLoader instance
    """
    loader = AgentLoader(agents_dir)

    return loader


def load_agents_from_dir(
    agents_dir: str, tool_loader: AgentLoader | None = None
) -> list[dict[str, Any]]:
    """
    Convenience function to load agents from a directory.

    Args:
        agents_dir: Directory containing agent YAML files
        tool_loader: Optional AgentLoader with registered tools

    Returns:
        List of loaded and processed agent configurations
    """
    loader = tool_loader or create_agent_loader()

    if tool_loader is None:
        _init_default_tools(loader)

    return loader.load_from_directory(agents_dir)


def _init_default_tools(loader: AgentLoader):
    """Initialize loader with default tools from the tools registry."""
    try:
        from .tools.registry import get_default_tool_registry
        from .tools.search import get_search_tools

        registry = get_default_tool_registry()
        registry = get_search_tools(registry)

        for tool_def in registry.list():
            loader.register_tool(
                name=tool_def.name,
                func=tool_def.func,
                description=tool_def.description,
            )

        logger.debug(f"Initialized {loader.get_registered_tools()} default tools")

    except ImportError as e:
        logger.warning(f"Could not initialize default tools: {e}")
