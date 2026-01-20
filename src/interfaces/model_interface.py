"""
Model interface for DeepAgent Orchestrator.

This module defines the interface for model initialization and management.
"""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel


class ModelInterface(ABC):
    """Abstract interface for model initialization and management."""

    @abstractmethod
    def initialize(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        **kwargs
    ) -> BaseChatModel:
        """
        Initialize a chat model.

        Args:
            model_name: Name of the model (e.g., "anthropic:claude-sonnet-4-20250514")
            api_key: Optional API key
            base_url: Optional base URL for the API
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters

        Returns:
            Initialized chat model

        Raises:
            ModelInitializationError: If model initialization fails
        """
        pass

    @abstractmethod
    def validate_model_name(self, model_name: str) -> bool:
        """
        Validate a model name format.

        Args:
            model_name: The model name to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    @abstractmethod
    def get_available_models(self) -> dict[str, Any]:
        """
        Get list of available models.

        Returns:
            Dictionary mapping model IDs to their metadata
        """
        pass
