"""
Task service for DeepAgent Orchestrator.

This module provides the TaskService implementation for task execution.
"""

import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from ..exceptions import TaskExecutionError
from ..interfaces import TaskInterface

logger = logging.getLogger(__name__)


class TaskService(TaskInterface):
    """Service for task execution and state management."""

    async def execute(
        self, task: str, agent: CompiledStateGraph, thread_id: str | None = None
    ) -> dict[str, Any]:
        """
        Execute a task synchronously.

        Args:
            task: The task description
            agent: The agent to execute the task
            thread_id: Optional thread ID for state persistence

        Returns:
            Execution result

        Raises:
            TaskExecutionError: If task execution fails
        """
        thread_id = thread_id or str(uuid.uuid4())
        inputs = {"messages": [{"role": "user", "content": task}]}
        config = {"configurable": {"thread_id": thread_id}}

        try:
            result = await agent.ainvoke(inputs, config=config)
            logger.info(f"Task completed successfully for thread: {thread_id}")
            return result
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            raise TaskExecutionError(
                message=f"Task execution failed: {e}",
                task=task,
                thread_id=thread_id,
                cause=e,
            )

    async def stream(
        self, task: str, agent: CompiledStateGraph, thread_id: str | None = None
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
        thread_id = thread_id or str(uuid.uuid4())
        inputs = {"messages": [{"role": "user", "content": task}]}
        config = {"configurable": {"thread_id": thread_id}}

        try:
            async for chunk in agent.astream(inputs, config=config):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming execution failed: {e}")
            raise TaskExecutionError(
                message=f"Streaming execution failed: {e}",
                task=task,
                thread_id=thread_id,
                cause=e,
            )

    async def stream_events(
        self, task: str, agent: CompiledStateGraph, thread_id: str | None = None
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
        thread_id = thread_id or str(uuid.uuid4())
        inputs = {"messages": [{"role": "user", "content": task}]}
        config = {"configurable": {"thread_id": thread_id}}

        try:
            async for event in agent.astream_events(inputs, config=config):
                yield event
        except Exception as e:
            logger.error(f"Event streaming failed: {e}")
            raise TaskExecutionError(
                message=f"Event streaming failed: {e}",
                task=task,
                thread_id=thread_id,
                cause=e,
            )

    async def get_state(
        self, agent: CompiledStateGraph, thread_id: str
    ) -> dict[str, Any] | None:
        """
        Get the current state for a thread.

        Args:
            agent: The agent
            thread_id: Thread identifier

        Returns:
            Current state or None
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            return await agent.aget_state(config)
        except Exception as e:
            logger.error(f"Failed to get state: {e}")
            return None

    async def clear_state(self, agent: CompiledStateGraph, thread_id: str) -> bool:
        """
        Clear the state for a thread.

        Args:
            agent: The agent
            thread_id: Thread identifier

        Returns:
            True if cleared, False if not found
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            await agent.aupdate_state(config, {"messages": []})
            logger.info(f"Cleared state for thread: {thread_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear state: {e}")
            return False
