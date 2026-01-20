"""
Storage interface for DeepAgent Orchestrator.

This module defines the interface for persistent storage.
"""

from abc import ABC, abstractmethod
from typing import Any


class StorageInterface(ABC):
    """Abstract interface for persistent storage."""

    @abstractmethod
    def save(self, key: str, data: dict[str, Any], namespace: str = "default") -> bool:
        """
        Save data to storage.

        Args:
            key: Unique key for the data
            data: Data to save
            namespace: Storage namespace

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def load(self, key: str, namespace: str = "default") -> dict[str, Any] | None:
        """
        Load data from storage.

        Args:
            key: Unique key for the data
            namespace: Storage namespace

        Returns:
            Loaded data or None
        """
        pass

    @abstractmethod
    def delete(self, key: str, namespace: str = "default") -> bool:
        """
        Delete data from storage.

        Args:
            key: Unique key for the data
            namespace: Storage namespace

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def list_keys(self, namespace: str = "default") -> list:
        """
        List all keys in a namespace.

        Args:
            namespace: Storage namespace

        Returns:
            List of keys
        """
        pass

    @abstractmethod
    def clear_namespace(self, namespace: str = "default") -> bool:
        """
        Clear all data in a namespace.

        Args:
            namespace: Storage namespace

        Returns:
            True if successful
        """
        pass
