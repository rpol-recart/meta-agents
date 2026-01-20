"""
Backend interface for DeepAgent Orchestrator.

This module defines the interface for backend management.
"""

from abc import ABC, abstractmethod
from typing import Any

from langgraph.prebuilt.tool_node import ToolRuntime


class BackendInterface(ABC):
    """Abstract interface for backend management."""

    @abstractmethod
    def create_backend(self, runtime: ToolRuntime) -> Any:
        """
        Create a backend instance.

        Args:
            runtime: The tool runtime

        Returns:
            Backend protocol instance
        """
        pass

    @abstractmethod
    def configure_filesystem(
        self,
        root_dir: str,
        memory_namespace: str = "memories",
        enable_memory: bool = True,
    ) -> None:
        """
        Configure filesystem backend.

        Args:
            root_dir: Root directory for file operations
            memory_namespace: Namespace for memory storage
            enable_memory: Whether to enable memory
        """
        pass
