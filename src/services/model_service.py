"""
Model service for DeepAgent Orchestrator.

This module provides the ModelService implementation for model initialization.
"""

import logging
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ..exceptions import ModelInitializationError
from ..interfaces import ModelInterface

logger = logging.getLogger(__name__)


class ModelService(ModelInterface):
    """Service for model initialization and management."""

    def initialize(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        **kwargs,
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
        try:
            # Check if custom base URL is provided (for Ollama, LM Studio, etc.)
            if base_url:
                # Use ChatOpenAI with custom base URL
                # Keep the full model name including provider prefix
                # Some APIs (like the one at foundation-models.api.cloud.ru) require the full name
                actual_model = model_name

                model = ChatOpenAI(
                    model=actual_model,
                    api_key=api_key or "not-needed",
                    base_url=base_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                logger.debug(
                    f"Initialized model with custom base URL: {base_url}, "
                    f"model: {actual_model}"
                )
            else:
                # Use standard LangChain initialization
                model = init_chat_model(model_name)
                logger.debug(f"Initialized model: {model_name}")

            return model

        except Exception as e:
            logger.error(f"Failed to initialize model '{model_name}': {e}")
            raise ModelInitializationError(
                message=f"Failed to initialize model: {e}",
                model_name=model_name,
                cause=e,
            )

    def validate_model_name(self, model_name: str) -> bool:
        """
        Validate a model name format.

        Args:
            model_name: The model name to validate

        Returns:
            True if valid, False otherwise
        """
        if not model_name or not isinstance(model_name, str):
            return False

        # Model name should not be empty and should not contain whitespace
        if not model_name.strip():
            return False

        # Should contain a provider prefix or be a valid model identifier
        # Examples: "anthropic:claude-sonnet-4-20250514", "openai:gpt-4o", "qwen2.5"
        if len(model_name) > 256:
            return False

        return True

    def get_available_models(self) -> dict[str, Any]:
        """
        Get list of available models.

        Returns:
            Dictionary mapping model IDs to their metadata
        """
        return {
            "anthropic:claude-sonnet-4-20250514": {
                "provider": "anthropic",
                "name": "Claude Sonnet 4",
                "description": "Anthropic's latest Sonnet model",
            },
            "anthropic:claude-3-5-sonnet-latest": {
                "provider": "anthropic",
                "name": "Claude 3.5 Sonnet",
                "description": "Anthropic's Claude 3.5 Sonnet",
            },
            "openai:gpt-4o": {
                "provider": "openai",
                "name": "GPT-4o",
                "description": "OpenAI's GPT-4 Omni model",
            },
            "openai:gpt-4o-mini": {
                "provider": "openai",
                "name": "GPT-4o-mini",
                "description": "OpenAI's GPT-4o Mini model",
            },
        }
