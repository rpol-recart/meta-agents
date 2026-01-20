"""
Agent service for DeepAgent Orchestrator.

This module provides the AgentService implementation for agent creation.
"""

import logging
from typing import Any

from deepagents import create_deep_agent
from langgraph.graph.state import CompiledStateGraph

from ..exceptions import AgentCreationError
from ..interfaces import AgentInterface

logger = logging.getLogger(__name__)


class AgentService(AgentInterface):
    """Service for agent creation and management."""

    def create_agent(
        self,
        model: Any,
        system_prompt: str | None = None,
        subagents: list[dict[str, Any]] | None = None,
        custom_tools: list[Any] | None = None,
        interrupt_on: dict[str, Any] | None = None,
        backend: Any | None = None,
        middleware: list[Any] | None = None,
        **kwargs,
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
        try:
            agent = create_deep_agent(
                model=model,
                system_prompt=system_prompt,
                tools=custom_tools or [],
                subagents=subagents or [],
                backend=backend,
                interrupt_on=interrupt_on,
                middleware=middleware or [],
            )

            logger.debug(f"Created agent with {len(subagents or [])} subagents")
            return agent

        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise AgentCreationError(message=f"Failed to create agent: {e}", cause=e)

    def add_subagent(
        self, agent: CompiledStateGraph, subagent: dict[str, Any]
    ) -> CompiledStateGraph:
        """
        Add a sub-agent to an existing agent.

        Note: This is a placeholder. In practice, you'd need to recreate the agent
        with the new subagent included.

        Args:
            agent: The existing agent (not used directly)
            subagent: Sub-agent configuration

        Returns:
            Agent (currently unchanged - would need recreation)
        """
        logger.info(f"Adding sub-agent: {subagent.get('name', 'unknown')}")
        # In a full implementation, you would recreate the agent with the new subagent
        return agent

    def remove_subagent(
        self, agent: CompiledStateGraph, subagent_name: str
    ) -> CompiledStateGraph:
        """
        Remove a sub-agent from an existing agent.

        Note: This is a placeholder. In practice, you'd need to recreate the agent
        without the subagent.

        Args:
            agent: The existing agent (not used directly)
            subagent_name: Name of the sub-agent to remove

        Returns:
            Agent (currently unchanged - would need recreation)
        """
        logger.info(f"Removing sub-agent: {subagent_name}")
        return agent

    def get_subagents(self, agent: CompiledStateGraph) -> list[dict[str, Any]]:
        """
        Get list of sub-agents from an agent.

        Args:
            agent: The agent to inspect

        Returns:
            List of sub-agent configurations
        """
        # In a full implementation, you'd extract subagents from the agent graph
        return []
