"""
Basic Agent Orchestration Framework for Project Analysis Tool

This module implements the orchestration layer for coordinating
code analysis agents in the multi-LLM agent system.
"""

import logging
from typing import Any

from src.agent_registry import SubAgentRegistry, SubAgentSpec
from src.orchestrator import DeepAgentOrchestrator
from src.tools.analysis import get_analysis_tools
from src.tools.dependency_tools import get_dependency_tools
from src.tools.pattern_tools import get_pattern_tools

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Orchestration layer for coordinating code analysis agents.

    This class manages the execution of specialized agents for code analysis
    and coordinates the flow of data between them.
    """

    def __init__(self, orchestrator: DeepAgentOrchestrator | None = None):
        """
        Initialize the agent orchestrator.

        Args:
            orchestrator: Optional DeepAgentOrchestrator instance. If not provided,
                          creates one with all analysis tools registered.
        """
        all_tools = []

        analysis_tools_registry = get_analysis_tools()
        analysis_tools = analysis_tools_registry.to_langchain_tools()
        all_tools.extend(analysis_tools)

        pattern_tools_registry = get_pattern_tools()
        pattern_tools = pattern_tools_registry.to_langchain_tools()
        all_tools.extend(pattern_tools)

        dependency_tools_registry = get_dependency_tools()
        dependency_tools = dependency_tools_registry.to_langchain_tools()
        all_tools.extend(dependency_tools)

        if orchestrator is None:
            self.orchestrator = DeepAgentOrchestrator(custom_tools=all_tools)
        else:
            self.orchestrator = orchestrator

        self.registry = SubAgentRegistry(self.orchestrator)
        logger.info(f"Registered {len(all_tools)} tools for agents")

    def register_analysis_agents(self, agent_definitions: list[dict[str, Any]]):
        """
        Register code analysis agents with the orchestrator.

        Args:
            agent_definitions: List of agent specification dictionaries
        """
        for agent_def in agent_definitions:
            spec = SubAgentSpec.from_dict(agent_def)
            self.registry.register(spec)
            logger.info(f"Registered analysis agent: {agent_def['name']}")

    async def coordinate_analysis(self, project_path: str) -> dict[str, Any]:
        """
        Coordinate the execution of analysis agents on a project.

        Args:
            project_path: Path to the project to analyze

        Returns:
            Dictionary containing analysis results
        """
        results = {}

        # Step 1: Run semantic analysis
        semantic_task = f"Analyze the semantics of the project at {project_path}"
        semantic_result = await self.orchestrator.run(
            task=semantic_task, thread_id=f"semantic_{project_path}"
        )
        results["semantic"] = semantic_result

        # Step 2: Run pattern recognition
        pattern_task = f"Identify design patterns in the project at {project_path}"
        pattern_result = await self.orchestrator.run(
            task=pattern_task, thread_id=f"pattern_{project_path}"
        )
        results["patterns"] = pattern_result

        # Step 3: Run dependency analysis
        dependency_task = f"Analyze dependencies in the project at {project_path}"
        dependency_result = await self.orchestrator.run(
            task=dependency_task, thread_id=f"dependency_{project_path}"
        )
        results["dependencies"] = dependency_result

        return results

    def get_agent_status(self) -> list[dict[str, Any]]:
        """
        Get the status of all registered agents.

        Returns:
            List of agent status dictionaries
        """
        agents = self.registry.list()
        return [
            {"name": agent.name, "description": agent.description, "tools": agent.tools}
            for agent in agents
        ]
