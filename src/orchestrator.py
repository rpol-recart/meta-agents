"""
DeepAgent Orchestrator - Core orchestrator using LangChain DeepAgent.

Features:
- Built-in planning (write_todos/read_todos)
- Built-in file operations (ls, read_file, write_file, edit_file, glob, grep)
- Built-in sub-agent delegation (task tool)
- Built-in human-in-the-loop approval (interrupt_on)
- Built-in context summarization
"""

import logging
import os
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deepagents.backends import (
    CompositeBackend,
    FilesystemBackend,
    StoreBackend,
)
from deepagents.backends.protocol import BackendProtocol
from langgraph.prebuilt.tool_node import ToolRuntime

from .interfaces import ModelInterface, AgentInterface, TaskInterface
from .services import ModelService, AgentService, TaskService


logger = logging.getLogger(__name__)

DEFAULT_MODEL_FALLBACK = "anthropic:claude-sonnet-4-20250514"
DEFAULT_WORKSPACE_DIR = "/tmp/agent-workspace"


@dataclass
class OrchestratorConfig:
    """Configuration for the DeepAgent orchestrator."""

    model_name: str = DEFAULT_MODEL_FALLBACK
    system_prompt: str | None = None
    subagents: list[dict[str, Any]] = field(default_factory=list)
    interrupt_on: dict[str, dict[str, list[str]]] | None = None
    custom_tools: list = field(default_factory=list)
    middleware: list = field(default_factory=list)
    enable_memory: bool = False
    memory_namespace: str = "memories"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    workspace_dir: str = DEFAULT_WORKSPACE_DIR

    @classmethod
    def from_settings(cls, settings: "Settings") -> "OrchestratorConfig":
        """
        Create OrchestratorConfig from centralized Settings.

        This method serves as an adapter between the centralized Settings
        configuration and the OrchestratorConfig used by DeepAgentOrchestrator.

        Args:
            settings: Centralized Settings instance from config.py

        Returns:
            OrchestratorConfig instance configured from Settings

        Example:
            >>> from src.config import load_settings, apply_env_overrides
            >>> settings = load_settings()
            >>> settings = apply_env_overrides(settings)
            >>> config = OrchestratorConfig.from_settings(settings)
        """
        model_name = getattr(settings.model, "name", None) or cls.model_name

        api_key = getattr(settings.model, "api_key", None) or os.environ.get(
            "OPENAI_API_KEY"
        )
        base_url = getattr(settings.model, "base_url", None) or os.environ.get(
            "OPENAI_BASE_URL"
        )

        workspace_dir = (
            getattr(settings.backend, "root_dir", None) or DEFAULT_WORKSPACE_DIR
        )
        store_namespace = (
            getattr(settings.backend, "store_namespace", None) or "memories"
        )

        hitl_enabled = (
            getattr(settings.hitl, "enabled", None) if settings.hitl else False
        )

        Path(workspace_dir).mkdir(parents=True, exist_ok=True)

        return cls(
            model_name=model_name,
            system_prompt=None,
            subagents=[],
            interrupt_on=None,
            custom_tools=[],
            middleware=[],
            enable_memory=hitl_enabled,
            memory_namespace=store_namespace,
            openai_api_key=api_key,
            openai_base_url=base_url,
            workspace_dir=workspace_dir,
        )


class DeepAgentOrchestrator:
    """
    Multi-agent orchestrator using LangChain DeepAgent.

    This orchestrator wraps the DeepAgent framework to provide:
    - Built-in planning with todo lists
    - Built-in file operations
    - Sub-agent delegation capabilities
    - Human-in-the-loop approval workflows
    - Persistent memory across conversations

    This class supports dependency injection for testing and customization.

    Attributes:
        config: OrchestratorConfig instance
        model: LangChain chat model
        agent: Compiled DeepAgent graph
        model_service: Model initialization service
        agent_service: Agent creation service
        task_service: Task execution service
    """

    def __init__(
        self,
        model_name: str | None = None,
        system_prompt: str | None = None,
        subagents: list[dict[str, Any]] | None = None,
        interrupt_on: dict | None = None,
        custom_tools: list | None = None,
        middleware: list | None = None,
        enable_memory: bool = True,
        memory_namespace: str = "memories",
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        model_service: ModelInterface | None = None,
        agent_service: AgentInterface | None = None,
        task_service: TaskInterface | None = None,
    ):
        """
        Initialize the DeepAgent orchestrator.

        Args:
            model_name: LLM model identifier. Reads from DEFAULT_MODEL env var.
            system_prompt: Custom system instructions (appended to defaults)
            subagents: List of sub-agent specifications
            interrupt_on: Human-in-the-loop configuration for sensitive tools
            custom_tools: Additional tools beyond built-in DeepAgent tools
            middleware: Additional middleware instances
            enable_memory: Enable persistent memory across conversations
            memory_namespace: Namespace for stored memories
            openai_api_key: API key for OpenAI-compatible API
            openai_base_url: Base URL for OpenAI-compatible API
            model_service: Optional model service for dependency injection
            agent_service: Optional agent service for dependency injection
            task_service: Optional task service for dependency injection
        """
        env_model = os.environ.get("DEFAULT_MODEL")
        effective_model = model_name or env_model or DEFAULT_MODEL_FALLBACK

        self.config = OrchestratorConfig(
            model_name=effective_model,
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

        self._model_service = model_service or ModelService()
        self._agent_service = agent_service or AgentService()
        self._task_service = task_service or TaskService()

        self.model = self._init_model()
        self.backend = self._create_backend()
        self.agent = self._create_agent()

        logger.info("Initialized DeepAgent orchestrator with model: %s",
                    self.config.model_name)

    def _init_model(self):
        """Initialize the LangChain chat model using the model service."""
        api_key = self.config.openai_api_key or os.environ.get("OPENAI_API_KEY")
        base_url = self.config.openai_base_url or os.environ.get("OPENAI_BASE_URL")

        try:
            self.model = self._model_service.initialize(
                model_name=self.config.model_name,
                api_key=api_key,
                base_url=base_url,
            )
            logger.debug("Initialized model: %s", self.config.model_name)
            return self.model
        except Exception as e:
            logger.error("Failed to initialize model: %s", e)
            raise

    def _create_backend(self) -> Callable[[ToolRuntime], BackendProtocol]:
        """Create the backend factory for file/memory management."""
        def create_backend_with_runtime(runtime: ToolRuntime) -> BackendProtocol:
            routes = {}

            if self.config.enable_memory:
                routes[f"/{self.config.memory_namespace}/"] = StoreBackend(runtime)

            workspace_dir = self.config.workspace_dir
            os.makedirs(workspace_dir, exist_ok=True)

            return CompositeBackend(
                default=FilesystemBackend(root_dir=workspace_dir),
                routes=routes,
            )

        return create_backend_with_runtime

    def _create_agent(self):
        """Create the DeepAgent with all configurations using the agent service."""
        try:
            self.agent = self._agent_service.create_agent(
                model=self.model,
                system_prompt=self.config.system_prompt,
                tools=self.config.custom_tools,
                subagents=self.config.subagents,
                backend=self.backend,
                interrupt_on=self.config.interrupt_on,
                middleware=self.config.middleware,
            )
            logger.debug("Created DeepAgent successfully")
            return self.agent
        except Exception as e:
            logger.error("Failed to create DeepAgent: %s", e)
            raise

    async def run(self, task: str, thread_id: str | None = None) -> dict[str, Any]:
        """
        Execute a task through the orchestrator using the task service.

        Args:
            task: The task to execute
            thread_id: Optional thread ID for state persistence

        Returns:
            Dict containing the execution result with messages
        """
        try:
            result = await self._task_service.execute(
                task=task,
                agent=self.agent,
                thread_id=thread_id,
            )
            return result
        except Exception as e:
            logger.error("Task execution failed: %s", e)
            raise

    async def stream(
        self, task: str, thread_id: str | None = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream execution for real-time output using the task service.

        Args:
            task: The task to execute
            thread_id: Optional thread ID for state persistence

        Yields:
            Dict containing execution chunks
        """
        try:
            async for chunk in self._task_service.stream(
                task=task,
                agent=self.agent,
                thread_id=thread_id,
            ):
                yield chunk
        except Exception as e:
            logger.error("Streaming execution failed: %s", e)
            raise

    async def astream_events(
        self, task: str, thread_id: str | None = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream execution events for debugging/monitoring using the task service.

        Args:
            task: The task to execute
            thread_id: Optional thread ID for state persistence

        Yields:
            LangGraph execution events
        """
        try:
            async for event in self._task_service.stream_events(
                task=task,
                agent=self.agent,
                thread_id=thread_id,
            ):
                yield event
        except Exception as e:
            logger.error("Event streaming failed: %s", e)
            raise

    def add_subagent(
        self,
        name: str,
        description: str,
        system_prompt: str,
        tools: list | None = None,
        model: str | None = None,
    ):
        """
        Add a new sub-agent dynamically.

        Args:
            name: Unique identifier for the sub-agent
            description: Human-readable description
            system_prompt: System instructions for the sub-agent
            tools: List of tool names available to the sub-agent
            model: Optional model override for the sub-agent
        """
        subagent_spec = {
            "name": name,
            "description": description,
            "system_prompt": system_prompt,
            "tools": tools or [],
        }
        if model:
            subagent_spec["model"] = model

        self.config.subagents.append(subagent_spec)
        logger.info("Added sub-agent: %s", name)

        self.agent = self._create_agent()

    def remove_subagent(self, name: str) -> bool:
        """
        Remove a sub-agent by name.

        Args:
            name: Name of the sub-agent to remove

        Returns:
            True if removed, False if not found
        """
        for i, subagent in enumerate(self.config.subagents):
            if subagent.get("name") == name:
                self.config.subagents.pop(i)
                logger.info("Removed sub-agent: %s", name)
                self.agent = self._create_agent()
                return True
        return False

    def list_subagents(self) -> list[dict[str, Any]]:
        """List all registered sub-agents."""
        return self.config.subagents

    def configure_hitl(self, tool_name: str, allowed_decisions: list[str]):
        """
        Configure human-in-the-loop for a specific tool.

        Args:
            tool_name: Name of the tool to protect
            allowed_decisions: List of allowed decisions
        """
        if self.config.interrupt_on is None:
            self.config.interrupt_on = {}

        self.config.interrupt_on[tool_name] = {"allowed_decisions": allowed_decisions}

        logger.info("Configured HITL for tool: %s", tool_name)
        self.agent = self._create_agent()

    def get_state(self, thread_id: str) -> dict[str, Any] | None:
        """
        Get the current state for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            State dict or None if not found
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            return self.agent.get_state(config)
        except Exception as e:
            logger.error("Failed to get state: %s", e)
            return None

    async def get_state_async(self, thread_id: str) -> dict[str, Any] | None:
        """Async version of get_state."""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            return await self.agent.aget_state(config)
        except Exception as e:
            logger.error("Failed to get async state: %s", e)
            return None

    def clear_state(self, thread_id: str):
        """
        Clear state for a thread.

        Args:
            thread_id: Thread identifier
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            self.agent.update_state(config, {"messages": []})
            logger.info("Cleared state for thread: %s", thread_id)
        except Exception as e:
            logger.error("Failed to clear state: %s", e)

    @property
    def model_service(self) -> ModelInterface:
        """Get the model service instance."""
        return self._model_service

    @property
    def agent_service(self) -> AgentInterface:
        """Get the agent service instance."""
        return self._agent_service

    @property
    def task_service(self) -> TaskInterface:
        """Get the task service instance."""
        return self._task_service
