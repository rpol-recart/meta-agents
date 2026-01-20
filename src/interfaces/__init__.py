"""
Interfaces for DeepAgent Orchestrator.

This module defines abstract base classes (interfaces) for all major components,
enabling loose coupling and easy testing.
"""

from .agent_interface import AgentInterface
from .backend_interface import BackendInterface
from .model_interface import ModelInterface
from .storage_interface import StorageInterface
from .task_interface import TaskInterface

__all__ = [
    "ModelInterface",
    "AgentInterface",
    "TaskInterface",
    "BackendInterface",
    "StorageInterface",
]
