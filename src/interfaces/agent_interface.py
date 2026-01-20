"""
Agent interface for DeepAgent Orchestrator.

This module defines the interface for agent creation and management.
"""

from abc import ABC, abstractmethod
from typing import Any

from langgraph.graph.state import CompiledStateGraph


class AgentInterface(ABC):
    """Abstract interface for agent creation and management."""

    @abstractmethod
    def create_agent(
        self,
        model: Any,
        system_prompt: str | None = None,
        subagents: list[dict[str, Any]] | None = None,
        custom_tools: list[Any] | None = None,
        interrupt_on: dict[str, Any] | None = None,
        backend: Any | None = None,
        middleware: list[Any] | None = None,
        **kwargs
    ) -> CompiledStateGraph:
        """
        Create an agent.

        Args:
            model: Initialized chat model
            system_prompt: Optional system prompt
            subagents: List of sub-agent configurations
            custom_tools: List of custom tools
            interrupt_on: HITL configuration
            backend: Backend protocol instance
            middleware: Additional middleware
            **kwargs: Additional parameters

        Returns:
            Compiled DeepAgent graph

        Raises:
            AgentCreationError: If agent creation fails
        """
        pass

    @abstractmethod
    def add_subagent(
        self, agent: CompiledStateGraph, subagent: dict[str, Any]
    ) -> CompiledStateGraph:
        """
        Add a sub-agent to an existing agent.

        Args:
            agent: The existing agent
            subagent: Sub-agent configuration

        Returns:
            Updated agent
        """
        pass

    @abstractmethod
    def remove_subagent(
        self, agent: CompiledStateGraph, subagent_name: str
    ) -> CompiledStateGraph:
        """
        Remove a sub-agent from an existing agent.

        Args:
            agent: The existing agent
            subagent_name: Name of the sub-agent to remove

        Returns:
            Updated agent
        """
        pass

    @abstractmethod
    def get_subagents(self, agent: CompiledStateGraph) -> list[dict[str, Any]]:
        """
        Get list of sub-agents from an agent.

        Args:
            agent: The agent to inspect

        Returns:
            List of sub-agent configurations
        """
        pass
