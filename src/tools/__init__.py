"""
Tool Registry - Management module for agent tools.

This module provides tools registration and management for the multi-agent system.
"""

from .registry import ToolRegistry, get_default_tool_registry
from .analysis import get_analysis_tools
from .pattern_tools import get_pattern_tools
from .dependency_tools import get_dependency_tools

__all__ = [
    "ToolRegistry",
    "get_default_tool_registry",
    "get_analysis_tools",
    "get_pattern_tools",
    "get_dependency_tools",
]
