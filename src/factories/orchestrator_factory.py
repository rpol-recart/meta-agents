"""
Orchestrator factory for DeepAgent Orchestrator.

This module provides the OrchestratorFactory class for creating orchestrator
instances with dependency injection and service composition.
"""

import logging
from typing import Any

from ..exceptions import OrchestratorCreationError
from ..interfaces import AgentInterface, ModelInterface, TaskInterface
from ..orchestrator import DeepAgentOrchestrator, OrchestratorConfig
from ..services import AgentService, ModelService, TaskService
from ..services.agent_loader import AgentLoader

logger = logging.getLogger(__name__)


class OrchestratorFactory:
    """
    Factory for creating DeepAgentOrchestrator instances.

    This factory encapsulates the creation logic and supports dependency injection
    for testing and customization.

    Example:
        >>> factory = OrchestratorFactory()
        >>> orchestrator = factory.create(
        ...     model_name="anthropic:claude-sonnet-4-20250514",
        ...     agents_dir="agents/",
        ... )
        >>> result = await orchestrator.run("Analyze this data")
    """

    def __init__(
        self,
        model_service: ModelInterface | None = None,
        agent_service: AgentInterface | None = None,
        task_service: TaskInterface | None = None,
    ):
        """
        Initialize the factory with optional services.

        Args:
            model_service: Optional model service for DI
            agent_service: Optional agent service for DI
            task_service: Optional task service for DI
        """
        self._model_service = model_service or ModelService()
        self._agent_service = agent_service or AgentService()
        self._task_service = task_service or TaskService()

        logger.debug("OrchestratorFactory initialized with services")

    def create(
        self,
        model_name: str = "anthropic:claude-sonnet-4-20250514",
        system_prompt: str | None = None,
        subagents: list[dict[str, Any]] | None = None,
        interrupt_on: dict | None = None,
        custom_tools: list | None = None,
        middleware: list | None = None,
        enable_memory: bool = False,
        memory_namespace: str = "memories",
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        workspace_dir: str = "/tmp/agent-workspace",
        agents_dir: str | None = None,
    ) -> DeepAgentOrchestrator:
        """
        Create a DeepAgentOrchestrator instance.

        Args:
            model_name: LLM model identifier
            system_prompt: Custom system instructions
            subagents: List of sub-agent specifications
            interrupt_on: HITL configuration
            custom_tools: Additional tools
            middleware: Additional middleware
            enable_memory: Enable persistent memory
            memory_namespace: Namespace for memories
            openai_api_key: API key for OpenAI-compatible API
            openai_base_url: Base URL for OpenAI-compatible API
            workspace_dir: Workspace directory for file operations
            agents_dir: Optional directory to load sub-agents from

        Returns:
            Configured DeepAgentOrchestrator instance

        Raises:
            OrchestratorCreationError: If creation fails
        """
        try:
            if agents_dir:
                loader = AgentLoader(agents_dir)
                _init_default_tools(loader)
                dir_subagents = loader.load_from_directory(agents_dir)
                subagents = (subagents or []) + dir_subagents

            config = OrchestratorConfig(
                model_name=model_name,
                system_prompt=system_prompt,
                subagents=subagents or [],
                interrupt_on=interrupt_on,
                custom_tools=custom_tools or [],
                middleware=middleware or [],
                enable_memory=enable_memory,
                memory_namespace=memory_namespace,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
                workspace_dir=workspace_dir,
            )

            orchestrator = DeepAgentOrchestrator(
                model_name=model_name,
                system_prompt=system_prompt,
                subagents=subagents or [],
                interrupt_on=interrupt_on,
                custom_tools=custom_tools or [],
                middleware=middleware or [],
                enable_memory=enable_memory,
                memory_namespace=memory_namespace,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
            )

            logger.info(f"Created orchestrator with model: {model_name}")
            return orchestrator

        except Exception as e:
            logger.error(f"Failed to create orchestrator: {e}")
            raise OrchestratorCreationError(
                message=f"Failed to create orchestrator: {e}",
                model_name=model_name,
                cause=e,
            )

    def create_with_config(
        self,
        config: OrchestratorConfig,
        agents_dir: str | None = None,
    ) -> DeepAgentOrchestrator:
        """
        Create orchestrator from OrchestratorConfig.

        Args:
            config: OrchestratorConfig instance
            agents_dir: Optional directory to load sub-agents from

        Returns:
            Configured DeepAgentOrchestrator instance
        """
        subagents = list(config.subagents)

        if agents_dir:
            loader = AgentLoader(agents_dir)
            _init_default_tools(loader)
            dir_subagents = loader.load_from_directory(agents_dir)
            subagents.extend(dir_subagents)

        config.subagents = subagents

        return self.create(
            model_name=config.model_name,
            system_prompt=config.system_prompt,
            subagents=subagents,
            interrupt_on=config.interrupt_on,
            custom_tools=config.custom_tools,
            middleware=config.middleware,
            enable_memory=config.enable_memory,
            memory_namespace=config.memory_namespace,
            openai_api_key=config.openai_api_key,
            openai_base_url=config.openai_base_url,
            workspace_dir=config.workspace_dir,
        )

    def get_model_service(self) -> ModelInterface:
        """Get the model service instance."""
        return self._model_service

    def get_agent_service(self) -> AgentInterface:
        """Get the agent service instance."""
        return self._agent_service

    def get_task_service(self) -> TaskInterface:
        """Get the task service instance."""
        return self._task_service

    def with_model_service(self, service: ModelInterface) -> "OrchestratorFactory":
        """
        Create a new factory with a custom model service.

        Args:
            service: Model service to use

        Returns:
            New OrchestratorFactory instance
        """
        return OrchestratorFactory(
            model_service=service,
            agent_service=self._agent_service,
            task_service=self._task_service,
        )

    def with_agent_service(self, service: AgentInterface) -> "OrchestratorFactory":
        """
        Create a new factory with a custom agent service.

        Args:
            service: Agent service to use

        Returns:
            New OrchestratorFactory instance
        """
        return OrchestratorFactory(
            model_service=self._model_service,
            agent_service=service,
            task_service=self._task_service,
        )

    def with_task_service(self, service: TaskInterface) -> "OrchestratorFactory":
        """
        Create a new factory with a custom task service.

        Args:
            service: Task service to use

        Returns:
            New OrchestratorFactory instance
        """
        return OrchestratorFactory(
            model_service=self._model_service,
            agent_service=self._agent_service,
            task_service=service,
        )


def create_orchestrator(
    model_name: str = "anthropic:claude-sonnet-4-20250514",
    agents_dir: str | None = None,
    **kwargs,
) -> DeepAgentOrchestrator:
    """
    Convenience function to create an orchestrator.

    Args:
        model_name: LLM model identifier
        agents_dir: Optional directory to load sub-agents from
        **kwargs: Additional arguments passed to factory.create()

    Returns:
        Configured DeepAgentOrchestrator instance
    """
    factory = OrchestratorFactory()
    return factory.create(model_name=model_name, agents_dir=agents_dir, **kwargs)


def _init_default_tools(loader: AgentLoader):
    """Initialize loader with default tools."""
    try:
        from ..tools.registry import get_default_tool_registry
        from ..tools.search import get_search_tools

        registry = get_default_tool_registry()
        registry = get_search_tools(registry)

        for tool_def in registry.list():
            loader.register_tool(
                name=tool_def.name,
                func=tool_def.func,
                description=tool_def.description,
            )

        logger.debug("Initialized default tools in loader")

    except ImportError as e:
        logger.warning(f"Could not initialize default tools: {e}")
