"""
Tool Registry - Manage custom tools for agents.
"""

import builtins
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


@dataclass
class ToolDef:
    """Definition for a tool."""

    name: str
    func: Callable
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    return_type: str = "str"


class ToolRegistry:
    """
    Registry for managing custom tools.

    Tools can be registered and later retrieved as LangChain BaseTool instances.
    """

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}

    def register(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: dict[str, Any] | None = None,
        return_type: str = "str",
    ) -> ToolDef:
        """
        Register a tool function.

        Args:
            name: Unique identifier for the tool
            func: The function to call
            description: Human-readable description
            parameters: JSON Schema for parameters
            return_type: Return type description

        Returns:
            ToolDef instance
        """
        tool_def = ToolDef(
            name=name,
            func=func,
            description=description,
            parameters=parameters or {},
            return_type=return_type,
        )
        self._tools[name] = tool_def
        logger.debug(f"Registered tool: {name}")
        return tool_def

    def get(self, name: str) -> ToolDef | None:
        """Get tool definition by name."""
        return self._tools.get(name)

    def list(self) -> list[ToolDef]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_names(self) -> builtins.list[str]:
        """List all tool names."""
        return list(self._tools.keys())

    def unregister(self, name: str) -> bool:
        """Unregister a tool by name."""
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")
            return True
        return False

    def clear(self):
        """Clear all registered tools."""
        self._tools.clear()
        logger.debug("Cleared all tools")

    def to_langchain_tools(self) -> builtins.list[BaseTool]:
        """
        Convert all registered tools to LangChain BaseTool instances.

        Returns:
            List of BaseTool instances
        """
        from langchain_core.tools import StructuredTool

        langchain_tools = []

        for tool_def in self._tools.values():
            func = tool_def.func
            description = tool_def.description

            langchain_tool = StructuredTool.from_function(
                func=func,
                name=tool_def.name,
                description=description,
            )

            langchain_tools.append(langchain_tool)

        return langchain_tools

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self.list_names())


def get_default_tool_registry() -> ToolRegistry:
    """Create and populate default tool registry."""
    registry = ToolRegistry()

    def get_weather(city: str) -> str:
        """Get the weather in a city."""
        return f"Weather in {city}: Sunny, 72°F"

    def calculate(a: int, b: int, operation: str = "add") -> str:
        """Perform a calculation."""
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            result = a / b if b != 0 else "Error: Division by zero"
        else:
            return f"Unknown operation: {operation}"
        return f"{a} {operation} {b} = {result}"

    def current_time() -> str:
        """Get the current time."""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def random_number(min_val: int = 1, max_val: int = 100) -> str:
        """Generate a random number."""
        import random

        return str(random.randint(min_val, max_val))

    registry.register(
        name="get_weather",
        func=get_weather,
        description="Get the weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    )

    registry.register(
        name="calculate",
        func=calculate,
        description="Perform basic math operations",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"},
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "Operation to perform",
                },
            },
            "required": ["a", "b"],
        },
    )

    registry.register(
        name="current_time",
        func=current_time,
        description="Get the current date and time",
    )

    registry.register(
        name="random_number",
        func=random_number,
        description="Generate a random number in a range",
        parameters={
            "type": "object",
            "properties": {
                "min_val": {
                    "type": "integer",
                    "description": "Minimum value",
                    "default": 1,
                },
                "max_val": {
                    "type": "integer",
                    "description": "Maximum value",
                    "default": 100,
                },
            },
        },
    )

    return registry
