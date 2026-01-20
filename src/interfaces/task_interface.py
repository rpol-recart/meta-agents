"""
Task interface for DeepAgent Orchestrator.

This module defines the interface for task execution and state management.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any


class TaskInterface(ABC):
    """Abstract interface for task execution."""

    @abstractmethod
    async def execute(
        self, task: str, agent: Any, thread_id: str | None = None
    ) -> dict[str, Any]:
        """
        Execute a task.

        Args:
            task: The task description
            agent: The agent to execute the task
            thread_id: Optional thread ID for state persistence

        Returns:
            Execution result

        Raises:
            TaskExecutionError: If task execution fails
        """
        pass

    @abstractmethod
    async def stream(
        self, task: str, agent: Any, thread_id: str | None = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream task execution.

        Args:
            task: The task description
            agent: The agent to execute the task
            thread_id: Optional thread ID for state persistence

        Yields:
            Execution chunks

        Raises:
            TaskExecutionError: If task execution fails
        """
        pass

    @abstractmethod
    async def stream_events(
        self, task: str, agent: Any, thread_id: str | None = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream execution events for debugging/monitoring.

        Args:
            task: The task description
            agent: The agent to execute the task
            thread_id: Optional thread ID for state persistence

        Yields:
            LangGraph execution events

        Raises:
            TaskExecutionError: If event streaming fails
        """
        pass

    @abstractmethod
    async def get_state(self, agent: Any, thread_id: str) -> dict[str, Any] | None:
        """
        Get the current state for a thread.

        Args:
            agent: The agent
            thread_id: Thread identifier

        Returns:
            Current state or None
        """
        pass

    @abstractmethod
    async def clear_state(self, agent: Any, thread_id: str) -> bool:
        """
        Clear the state for a thread.

        Args:
            agent: The agent
            thread_id: Thread identifier

        Returns:
            True if cleared, False if not found
        """
        pass
